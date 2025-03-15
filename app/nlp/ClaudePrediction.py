import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import re
import logging
from dotenv import load_dotenv
import anthropic
from langchain.memory import ConversationBufferMemory
import traceback

from data.api_fetcher import DataCollector
from models.fbprophet import ProphetModel
from models.viz import PredictionVisualizer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finn_assistant.log')
    ]
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

class ClaudeHandler:
    def __init__(self):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Clé ANTHROPIC_API_KEY introuvable dans .env")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.retry_delay = 60
        self.max_retries = 3
        self.max_tokens = 1000
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", max_len=20, return_messages=True)
        self.reference_date = datetime.now()
        self.data_cache = {'crypto': {}, 'stock': {}}
        self.cache_ttl = 300
        
        # Initialisation des composants
        self.collector = DataCollector()
        self.chat_history = []
        self.prophet_model = ProphetModel()
        self.visualizer = PredictionVisualizer()

    def _cache_key(self, symbol, timeframe):
        return f"{symbol.lower()}_{timeframe}"

    def _get_from_cache(self, asset_type, symbol, timeframe):
        key = self._cache_key(symbol, timeframe)
        cache = self.data_cache[asset_type].get(key, {})
        if not cache or (time.time() - cache.get('timestamp', 0) > (60 if timeframe == "current" else self.cache_ttl * 5)):
            return None
        return cache.get('data')

    def _throttled_api_call(self, func, *args, **kwargs):
        initial_delay = 2
        max_delay = 30
        max_retries = 5
        last_call_time = getattr(self, '_last_api_call_time', 0)
        current_time = time.time()
        if current_time - last_call_time < 1:
            time.sleep(1 - (current_time - last_call_time))
        
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                self._last_api_call_time = time.time()
                return result
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "Too Many Requests" in error_str:
                    if attempt < max_retries - 1:
                        delay = min(initial_delay * (2 ** attempt), max_delay)
                        logger.warning(f"Limite de taux atteinte. Attente de {delay}s ({attempt+1}/{max_retries})")
                        time.sleep(delay)
                        continue
                logger.error(f"Erreur API après {attempt+1} tentatives: {error_str}")
                raise

    def _save_to_cache(self, asset_type, symbol, timeframe, data):
        key = self._cache_key(symbol, timeframe)
        self.data_cache[asset_type][key] = {'timestamp': time.time(), 'data': data}

    def fetch_price_data(self, symbol, timeframes=None):
        """Récupère les données de prix pour n'importe quel actif (crypto ou action)"""
        if timeframes is None:
            timeframes = ["current"]
        elif isinstance(timeframes, str):
            timeframes = [timeframes]
            
        result = {"symbol": symbol, "name": None, "asset_type": None, "data": {}, "error": None}
        
        try:
            asset_found = False
            resolved_symbol = None
            name = None
            asset_type = None
            
            # Essayer d'abord comme une action
            try:
                stock_info = self._throttled_api_call(self.collector.mapper.get_stock_info, symbol)
                if stock_info:
                    asset_type = "stock"
                    resolved_symbol, name = stock_info
                    asset_found = True
                    logger.info(f"Actif identifié comme action: {resolved_symbol} ({name})")
            except Exception as e:
                logger.debug(f"Échec identification comme action: {str(e)}")
            
            # Si ce n'est pas une action, essayer comme une crypto
            if not asset_found:
                try:
                    crypto_info = self._throttled_api_call(self.collector.mapper.get_crypto_info, symbol)
                    if crypto_info:
                        asset_type = "crypto"
                        resolved_symbol, name = crypto_info
                        asset_found = True
                        logger.info(f"Actif identifié comme crypto: {resolved_symbol} ({name})")
                except Exception as e:
                    logger.debug(f"Échec identification comme crypto: {str(e)}")
            
            # Correction pour Bitcoin (BTC)
            if not asset_found and symbol.lower() in ['btc', 'bitcoin']:
                asset_type = "crypto"
                resolved_symbol = "BTC"
                name = "Bitcoin"
                asset_found = True
                logger.info(f"Forçage de BTC comme crypto: {resolved_symbol} ({name})")
            
            # Si l'actif n'est pas trouvé
            if not asset_found:
                result["error"] = f"Actif '{symbol}' non trouvé dans nos sources de données."
                return result
            
            # Mettre à jour les informations de résultat
            result["symbol"] = resolved_symbol
            result["name"] = name
            result["asset_type"] = asset_type
            
            # Récupérer les données historiques si nécessaire
            hist_data = None
            if any(tf not in ["current"] for tf in timeframes):
                if asset_type == "crypto":
                    hist_data = self._throttled_api_call(self.collector.get_crypto_historical, resolved_symbol)
                else:
                    hist_data = self._throttled_api_call(self.collector.get_stock_historical, resolved_symbol)
                
                if hist_data is not None and not hist_data.empty:
                    hist_data.index = pd.to_datetime(hist_data.index).tz_localize(None)
                else:
                    hist_data = None
            
            # Récupérer les données pour chaque timeframe demandé
            for timeframe in timeframes:
                cached_data = self._get_from_cache(asset_type, resolved_symbol, timeframe)
                if cached_data and timeframe != "current":
                    result["data"][timeframe] = cached_data
                    continue
                
                try:
                    if timeframe == "current":
                        if asset_type == "crypto":
                            raw_data = self._throttled_api_call(self.collector.get_crypto_current, resolved_symbol)
                            if not raw_data or "price" not in raw_data:
                                result["data"][timeframe] = {"error": "Données actuelles indisponibles"}
                                continue
                            price = float(raw_data.get("price", 0))
                            data = {
                                "price": round(price, 2),
                                "date": self.reference_date.strftime("%d %B %Y"),
                                "change": round(float(raw_data.get("change_24h", 0)), 2),
                                "volume": raw_data.get("volume", 0),
                                "market_cap": raw_data.get("market_cap", None)
                            }
                            result["data"][timeframe] = data
                        else:
                            raw_data = self._throttled_api_call(self.collector.get_stock_current, resolved_symbol)
                            if not raw_data or "price" not in raw_data:
                                result["data"][timeframe] = {"error": "Données actuelles indisponibles"}
                                continue
                            price = float(raw_data.get("price", 0))
                            data = {
                                "price": round(price, 2),
                                "date": self.reference_date.strftime("%d %B %Y"),
                                "change": round(float(raw_data.get("change", 0)), 2),
                                "volume": raw_data.get("volume", 0),
                                "market_cap": raw_data.get("market_cap", None)
                            }
                            result["data"][timeframe] = data
                        self._save_to_cache(asset_type, resolved_symbol, timeframe, result["data"][timeframe])
                    
                    elif hist_data is not None:
                        if timeframe == "week":
                            week_ago = (self.reference_date - timedelta(days=7)).date()
                            week_data = hist_data[hist_data.index.date >= week_ago]
                            if not week_data.empty:
                                data = {
                                    "start_date": week_data.index[0].strftime("%d %B %Y"),
                                    "end_date": week_data.index[-1].strftime("%d %B %Y"),
                                    "high": float(week_data["Close"].max()),
                                    "low": float(week_data["Close"].min()),
                                    "change": float(((week_data["Close"].iloc[-1] / week_data["Close"].iloc[0]) - 1) * 100)
                                }
                                result["data"][timeframe] = data
                        elif timeframe == "month":
                            month_ago = (self.reference_date - timedelta(days=30)).date()
                            month_data = hist_data[hist_data.index.date >= month_ago]
                            if not month_data.empty:
                                data = {
                                    "start_date": month_data.index[0].strftime("%d %B %Y"),
                                    "end_date": month_data.index[-1].strftime("%d %B %Y"),
                                    "high": float(month_data["Close"].max()),
                                    "low": float(month_data["Close"].min()),
                                    "change": float(((month_data["Close"].iloc[-1] / month_data["Close"].iloc[0]) - 1) * 100)
                                }
                                result["data"][timeframe] = data
                        self._save_to_cache(asset_type, resolved_symbol, timeframe, result["data"][timeframe])
                    
                    else:
                        result["data"][timeframe] = {"error": "Aucune donnée historique disponible"}
                
                except Exception as e:
                    result["data"][timeframe] = {"error": f"Erreur récupération: {str(e)}"}
        
        except Exception as e:
            result["error"] = f"Erreur globale: {str(e)}"
        
        return result

    def _extract_query_info(self, query):
        """Extrait les informations de la requête sans coder en dur des actifs spécifiques"""
        # Analyse sémantique de la requête
        analysis_prompt = f"""
        Analyse cette requête en français: "{query}"
        Détermine si l'utilisateur demande:
        1. Une question sur les prix actuels/historiques (price)
        2. Une prédiction de prix futur (future_price)
        3. Une visualisation/graphique (visualization)
        4. Une analyse ou tendance (analysis)
        5. Une comparaison d'actifs (comparison)
        6. Autre (other)
        
        Identifie:
        - Actifs mentionnés (ex: Bitcoin, BTC, Ethereum, ETH, Apple, AAPL, TSLA, Cardano, ADA, etc.)
        - Type ("stock" si actions ou "crypto" si crypto-monnaies)
        - Période en jours pour prédictions (extrais un nombre de jours ou mois, défaut 30 si non spécifié)
        - Timeframes pour prix ("current", "week", "month", etc.)
        
        Réponds au format JSON:
        {{
            "intent": "price/future_price/visualization/analysis/comparison/other",
            "assets": ["symbole1", "symbole2"],
            "asset_type": "crypto/stock",
            "timeframe": nombre_jours,
            "price_timeframes": ["current/week/month"],
            "language": "fr"
        }}
        """
        try:
            response = self.client.messages.create(model=self.model, max_tokens=200, messages=[{"role": "user", "content": analysis_prompt}])
            json_text = response.content[0].text.strip()
            logger.info(f"Résultat brut de l'extraction: {json_text}")
            
            # Trouver et extraire le JSON uniquement
            json_match = re.search(r'({[^}]*})', json_text.replace('\n', ''))
            if json_match:
                json_text = json_match.group(1)
            
            info = json.loads(json_text)
            
            # Vérifier et compléter les informations manquantes
            if not info.get("intent") or info["intent"] == "other":
                if info.get("assets"):
                    info["intent"] = "price"
            
            if not info.get("price_timeframes"):
                info["price_timeframes"] = ["current"]
                
            if not info.get("timeframe"):
                info["timeframe"] = 30
                
            if not info.get("language"):
                info["language"] = "fr"
                
            return info
            
        except Exception as e:
            logger.error(f"Échec du parsing JSON: {e}")
            traceback.print_exc()
            
            # Format par défaut si l'analyse échoue
            # Essayer d'extraire manuellement les actifs mentionnés
            actifs_potentiels = re.findall(r'\b(btc|bitcoin|eth|ethereum|ada|cardano|sol|solana|aapl|apple|tsla|tesla|amzn|amazon|nvda|nvidia)\b', query.lower())
            actifs_uniques = list(set(actifs_potentiels))
            
            # Déterminer le type d'actif de manière simple
            asset_type = "crypto"  # Par défaut
            if any(x in actifs_uniques for x in ["aapl", "apple", "tsla", "tesla", "amzn", "amazon", "nvda", "nvidia"]):
                asset_type = "stock"
                
            # Détecter si c'est une demande de prédiction
            is_prediction = any(word in query.lower() for word in ["prédi", "futur", "sera", "devenir", "évolu", "tendance"])
            intent = "future_price" if is_prediction else "price"
            
            # Détecter si c'est une demande de visualisation
            if any(word in query.lower() for word in ["graph", "visual", "courbe", "chart", "affich"]):
                intent = "visualization"
                
            return {
                "intent": intent,
                "assets": actifs_uniques if actifs_uniques else [],
                "asset_type": asset_type,
                "timeframe": 30,
                "price_timeframes": ["current"],
                "language": "fr"
            }

    def make_prediction(self, asset, asset_type, timeframe):
        """Génère une prédiction pour n'importe quel actif financier"""
        logger.info(f"Demande de prédiction pour {asset} ({asset_type}) sur {timeframe} jours")
        try:
            # Nettoyer les variables pour éviter des problèmes d'état
            self.prophet_model = ProphetModel()  # Réinitialiser pour éviter les conflits
            
            # Chargement des données
            logger.info(f"Chargement des données pour {asset}")
            prophet_data = self.prophet_model.load_data(asset, asset_type)
            
            # Construction du modèle
            logger.info(f"Construction du modèle pour {asset}")
            self.prophet_model.build_model()
            
            # Génération des prédictions
            logger.info(f"Génération des prédictions pour {asset} sur {timeframe} jours")
            forecast = self.prophet_model.forecast_specific_period(days=timeframe)
            
            if isinstance(forecast, pd.DataFrame) and not forecast.empty:
                # Extraction des informations pertinentes
                last_price = forecast['Prix prédit'].iloc[-1]
                last_date = forecast['Date'].iloc[-1].strftime('%Y-%m-%d')
                current_price = self.prophet_model.prophet_data['current_price']
                change_pct = ((last_price / current_price) - 1) * 100
                
                logger.info(f"Prédiction réussie pour {asset}: {last_price}$ le {last_date}")
                
                # Retourner les résultats structurés
                return {
                    "asset": asset,
                    "asset_name": self.prophet_model.asset_name,
                    "current_price": current_price,
                    "prediction": {
                        "price": last_price,
                        "date": last_date,
                        "change_percent": change_pct
                    }
                }
            else:
                logger.error(f"Aucune prédiction générée pour {asset}")
                return {"asset": asset, "error": "Impossible de générer des prédictions"}
        except Exception as e:
            logger.error(f"Erreur dans make_prediction pour {asset}: {e}")
            traceback.print_exc()
            return {"asset": asset, "error": str(e)}

    def create_visualization(self, asset, asset_type, timeframe):
        """Crée une visualisation pour un actif financier"""
        logger.info(f"Création de visualisation pour {asset} ({asset_type}) sur {timeframe} jours")
        try:
            # Réinitialiser le visualiseur
            self.visualizer = PredictionVisualizer()
            
            # Création du graphique avec gestion explicite de l'affichage
            self.visualizer.create_single_asset_plot(asset, asset_type, timeframe)
            
            # Vérification supplémentaire pour forcer l'affichage
            import plotly.io as pio
            pio.renderers.default = "browser"  # Forcer l'affichage dans le navigateur
            
            return {
                "success": True,
                "message": f"Visualisation créée pour {asset} sur {timeframe} jours - regarde dans ton navigateur !",
                "asset": asset,
                "asset_type": asset_type,
                "timeframe": timeframe
            }
        except Exception as e:
            logger.error(f"Erreur dans create_visualization pour {asset}: {e}")
            traceback.print_exc()
            return {"asset": asset, "error": str(e)}

    def analyze_asset(self, asset, asset_type):
        """Analyse n'importe quel actif financier"""
        logger.info(f"Analyse demandée pour {asset} ({asset_type})")
        try:
            # Réinitialiser le modèle Prophet
            self.prophet_model = ProphetModel()
            
            # Chargement des données
            self.prophet_model.load_data(asset, asset_type)
            
            # Construction du modèle
            self.prophet_model.build_model()
            
            # Évaluation des performances
            metrics = self.prophet_model.evaluate_model()
            
            # Analyse de la tendance future (30 jours par défaut)
            forecast = self.prophet_model.forecast_specific_period(days=30)
            
            if not forecast.empty:
                trend = "hausse" if forecast['Prix prédit'].iloc[-1] > self.prophet_model.prophet_data['current_price'] else "baisse"
                trend_pct = ((forecast['Prix prédit'].iloc[-1] / self.prophet_model.prophet_data['current_price']) - 1) * 100
                
                logger.info(f"Analyse réussie pour {asset}: tendance en {trend} de {trend_pct:.2f}%")
                
                return {
                    "asset": asset,
                    "asset_name": self.prophet_model.asset_name,
                    "current_price": self.prophet_model.prophet_data['current_price'],
                    "metrics": metrics,
                    "trend": trend,
                    "trend_percent": trend_pct
                }
            else:
                logger.error(f"Aucune prédiction générée pour l'analyse de {asset}")
                return {"asset": asset, "error": "Impossible de générer l'analyse de tendance"}
        except Exception as e:
            logger.error(f"Erreur dans analyze_asset pour {asset}: {e}")
            traceback.print_exc()
            return {"asset": asset, "error": str(e)}

    def _generate_system_prompt(self, query, financial_data=None, prediction_data=None, viz_data=None, analysis_data=None):
        """Génère le prompt système pour Claude basé sur les données collectées"""
        date_str = self.reference_date.strftime('%d %B %Y')
        history_data = self.memory.load_memory_variables({})['chat_history']
        history_str = "Contexte récent :\n" + "\n".join([f"{'Utilisateur' if i % 2 == 0 else 'Assistant'}: {msg.content}" for i, msg in enumerate(history_data[-2:])]) if history_data else "Aucun échange récent."

        financial_data_str = ""
        if financial_data:
            financial_data_str = "\n===== DONNÉES FINANCIÈRES =====\n"
            for asset_info in financial_data:
                if asset_info.get('error'):
                    financial_data_str += f"\nErreur pour {asset_info.get('symbol', 'actif inconnu')}: {asset_info.get('error')}\n"
                    continue
                    
                asset_name = asset_info.get('name', asset_info.get('symbol', ''))
                financial_data_str += f"\n--- {asset_name} ({asset_info.get('symbol', '')}) ---\n"
                for timeframe, data in asset_info.get('data', {}).items():
                    if isinstance(data, dict) and "error" not in data:
                        if timeframe == 'current':
                            financial_data_str += f"Prix actuel: {data.get('price', 'N/A')} $ (var: {data.get('change', 0):+.2f}%)\n"
                        elif timeframe in ['week', 'month']:
                            financial_data_str += f"{timeframe.capitalize()}: Haut: {data.get('high', 'N/A')} $ | Bas: {data.get('low', 'N/A')} $ | Var: {data.get('change', 0):+.2f}%\n"

        prediction_data_str = ""
        if prediction_data:
            prediction_data_str = "\n===== PRÉDICTIONS =====\n"
            for pred in prediction_data:
                if "error" not in pred:
                    prediction_data_str += f"{pred.get('asset_name', pred['asset'])}: Prix actuel {pred.get('current_price', 0):.2f} $ → Prix prédit {pred['prediction']['price']:.2f} $ le {pred['prediction']['date']} (var: {pred['prediction']['change_percent']:+.2f}%)\n"
                else:
                    prediction_data_str += f"{pred['asset']}: Erreur lors de la prédiction - {pred['error']}\n"

        viz_data_str = ""
        if viz_data and viz_data.get("success"):
            asset_name = viz_data.get("asset", "l'actif")
            timeframe = viz_data.get("timeframe", "30")
            viz_data_str = f"\n===== VISUALISATION =====\nUn graphique a été généré pour {asset_name} avec l'historique et les prédictions sur {timeframe} jours.\n"
        elif viz_data and "error" in viz_data:
            viz_data_str = f"\n===== VISUALISATION =====\nErreur: {viz_data['error']}\n"

        analysis_data_str = ""
        if analysis_data:
            analysis_data_str = "\n===== ANALYSE =====\n"
            for analysis in analysis_data:
                if "error" not in analysis:
                    asset_name = analysis.get('asset_name', analysis['asset'])
                    analysis_data_str += f"{asset_name}: Tendance en {analysis['trend']} de {analysis.get('trend_percent', 0):+.2f}%\n"
                    analysis_data_str += f"Prix actuel: {analysis.get('current_price', 0):.2f} $\n"
                    analysis_data_str += f"Précision du modèle (MAPE): {analysis['metrics'].get('MAPE', 0):.2f}%\n"
                else:
                    analysis_data_str += f"{analysis['asset']}: Erreur - {analysis['error']}\n"

        system_prompt = f"""
        Tu es Finn, un assistant financier conversationnel et précis.
        - Réponds directement en 1-2 phrases pour les prix ou prédictions.
        - Sois naturel et chaleureux ("tu vois", "en fait").
        - Ne mentionne jamais les processus techniques.
        - Date actuelle: {date_str}
        
        {history_str}
        {financial_data_str}
        {prediction_data_str}
        {viz_data_str}
        {analysis_data_str}
        
        Question: {query}
        Réponds en t'appuyant uniquement sur les données fournies ci-dessus, sans inventer.
        """
        return system_prompt

    def process_query(self, query):
        """Traite une requête utilisateur et génère une réponse adaptée"""
        try:
            # Extraire les informations de la requête
            query_info = self._extract_query_info(query)
            intent = query_info.get("intent")
            assets = query_info.get("assets", [])
            
            logger.info(f"Intention détectée: {intent}, Actifs: {assets}, Type: {query_info.get('asset_type')}")
            
            # Si aucun actif n'est détecté mais qu'il s'agit d'une intention liée à un actif
            if not assets and intent in ["price", "future_price", "visualization", "analysis"]:
                return "Je n'ai pas pu identifier l'actif financier dont tu parles. Peux-tu préciser le nom ou le symbole de l'actif qui t'intéresse?"

            financial_data = []
            prediction_data = []
            viz_data = None
            analysis_data = []

            # Traitement selon l'intention détectée
            if intent == "price" and assets:
                logger.info(f"Traitement de l'intention 'price' pour {assets}")
                for asset in assets:
                    asset_data = self.fetch_price_data(asset, query_info.get("price_timeframes"))
                    financial_data.append(asset_data)

            elif intent == "future_price" and assets:
                logger.info(f"Traitement de l'intention 'future_price' pour {assets}")
                for asset in assets:
                    # Récupérer les données actuelles d'abord pour le contexte
                    asset_data = self.fetch_price_data(asset, ["current"])
                    financial_data.append(asset_data)
                    
                    # Déterminer le type d'actif à partir des données récupérées
                    asset_type = asset_data.get("asset_type", query_info.get("asset_type", "crypto"))
                    
                    # Générer la prédiction
                    pred_data = self.make_prediction(asset, asset_type, query_info.get("timeframe", 30))
                    prediction_data.append(pred_data)

            elif intent == "visualization" and assets:
                logger.info(f"Traitement de l'intention 'visualization' pour {assets[0]}")
                asset = assets[0]
                
                # Récupérer les données actuelles pour le contexte
                asset_data = self.fetch_price_data(asset, ["current"])
                financial_data.append(asset_data)
                
                # Déterminer le type d'actif à partir des données récupérées
                asset_type = asset_data.get("asset_type", query_info.get("asset_type", "crypto"))
                
                # Utiliser le symbole résolu si disponible
                resolved_symbol = asset_data.get("symbol", asset)
                
                # S'assurer que timeframe est un entier
                timeframe = query_info.get("timeframe", 30)
                if not isinstance(timeframe, int):
                    try:
                        timeframe = int(timeframe)
                    except:
                        timeframe = 30
                
                # Créer la visualisation
                viz_data = self.create_visualization(resolved_symbol, asset_type, timeframe)

            elif intent == "analysis" and assets:
                logger.info(f"Traitement de l'intention 'analysis' pour {assets}")
                for asset in assets:
                    # Récupérer les données actuelles pour le contexte
                    asset_data = self.fetch_price_data(asset, ["current"])
                    financial_data.append(asset_data)
                    
                    # Déterminer le type d'actif à partir des données récupérées
                    asset_type = asset_data.get("asset_type", query_info.get("asset_type", "crypto"))
                    
                    # Générer l'analyse
                    analysis_result = self.analyze_asset(asset, asset_type)
                    analysis_data.append(analysis_result)

            elif intent == "comparison" and len(assets) > 1:
                logger.info(f"Traitement de l'intention 'comparison' pour {assets}")
                for asset in assets:
                    asset_data = self.fetch_price_data(asset, query_info.get("price_timeframes"))
                    financial_data.append(asset_data)

            else:
                # Si l'intention n'est pas reconnue ou pas supportée
                if not assets:
                    return "Je n'ai pas pu déterminer l'actif financier dont tu parles. Peux-tu reformuler ta question en précisant le nom de l'actif?"
                else:
                    # Traiter comme une question sur le prix actuel par défaut
                    logger.info(f"Traitement par défaut (prix actuel) pour {assets}")
                    for asset in assets:
                        asset_data = self.fetch_price_data(asset, ["current"])
                        financial_data.append(asset_data)

            # Vérifier si des données ont été collectées
            if not financial_data and not prediction_data and not viz_data and not analysis_data:
                logger.warning(f"Aucune donnée collectée pour la requête: {query}")
                return "Je n'ai pas pu récupérer les informations demandées. Peux-tu reformuler ta question ou préciser l'actif qui t'intéresse?"

            # Générer le prompt système
            system_prompt = self._generate_system_prompt(query, financial_data, prediction_data, viz_data, analysis_data)
            
            logger.info(f"Génération de la réponse via Claude")
            # Appeler Claude avec plusieurs tentatives en cas d'erreur
            for attempt in range(self.max_retries):
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content": system_prompt}]
                    )
                    resp_text = response.content[0].text.strip()
                    
                    # Sauvegarder le contexte de conversation
                    self.memory.save_context({"input": query}, {"output": resp_text})
                    self.chat_history.extend([query, resp_text])
                    
                    logger.info(f"Réponse générée avec succès")
                    return resp_text
                    
                except Exception as e:
                    logger.error(f"Erreur lors de l'appel à Claude (tentative {attempt+1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        return "Désolé, j'ai eu un problème pour générer une réponse. Peux-tu réessayer ta question dans quelques instants?"
                    time.sleep(self.retry_delay)
                    
        except Exception as e:
            logger.error(f"Erreur générale dans process_query: {e}")
            traceback.print_exc()
            return f"Désolé, j'ai rencontré une erreur lors du traitement de ta demande. Détail: {str(e)}"

if __name__ == "__main__":
    handler = ClaudeHandler()
    print("Finn estyo prêt ! Posez vos questions (ou 'exit' pour quitter) :")
    while True:
        try:
            query = input("\n> ")
            if query.lower() in ['exit', 'quit', 'q', 'sortie', 'quitter']:
                print("À bientôt !")
                break
            response = handler.process_query(query)
            print(f"\nA: {response}")
        except KeyboardInterrupt:
            print("\nProgramme interrompu par l'utilisateur.")
            break
        except Exception as e:
            print(f"\nErreur inattendue: {e}")
            traceback.print_exc()