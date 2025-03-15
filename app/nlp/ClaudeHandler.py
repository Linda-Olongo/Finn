import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

import pandas as pd
from datetime import datetime, timedelta
import time
import json
import re
import logging
from dotenv import load_dotenv
import anthropic
from langchain.memory import ConversationBufferMemory

# Import des modules
from data.api_fetcher import DataCollector

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

class ClaudeHandler:
    def __init__(self):
        """Initialisation de l'assistant avec Claude."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Clé ANTHROPIC_API_KEY introuvable dans .env")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.retry_delay = 60
        self.max_retries = 3
        self.max_tokens = 1000
        
        # Mémoire LangChain avec capacité augmentée
        try:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                max_len=20,
                return_messages=True
            )
        except Exception as e:
            logger.warning(f"Avertissement lors de l'initialisation de la mémoire: {e}")
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        # Date dynamique
        self.reference_date = datetime.now()
        
        # Cache pour les données financières
        self.data_cache = {
            'crypto': {},
            'stock': {}
        }
        
        # Durée de validité du cache
        self.cache_ttl = 300
        
        # Collector d'API
        self.collector = DataCollector()
    
        # Historique de conversation
        self.chat_history = []

    def _cache_key(self, symbol, timeframe):
        """Génère une clé de cache pour les données financières."""
        return f"{symbol.lower()}_{timeframe}"
    
    def _get_from_cache(self, asset_type, symbol, timeframe):
        key = self._cache_key(symbol, timeframe)
        cache = self.data_cache[asset_type].get(key, {})
        
        if not cache:
            return None
        
        cache_time = cache.get('timestamp', 0)
        ttl_to_use = 60 if timeframe == "current" else self.cache_ttl * 5
        
        if time.time() - cache_time > ttl_to_use:
            return None
        
        return cache.get('data')
    
    def _throttled_api_call(self, func, *args, **kwargs):
        """Effectue un appel API avec gestion intelligente des limites de taux."""
        initial_delay = 2
        max_delay = 30
        max_retries = 5  # Plus de tentatives
        
        last_call_time = getattr(self, '_last_api_call_time', 0)
        current_time = time.time()
        
        # Espacer les appels d'au moins 1 seconde pour éviter les rate limits
        if current_time - last_call_time < 1:
            time.sleep(1 - (current_time - last_call_time))
        
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                # Mise à jour du timestamp du dernier appel
                self._last_api_call_time = time.time()
                return result
            except Exception as e:
                error_str = str(e)
                
                # Gestion des erreurs de limite de taux
                if "429" in error_str or "Too Many Requests" in error_str:
                    if attempt < max_retries - 1:
                        # Backoff exponentiel
                        delay = min(initial_delay * (2 ** attempt), max_delay)
                        logger.warning(f"Limite de taux atteinte. Attente de {delay}s avant nouvel essai ({attempt+1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                
                # Autres types d'erreurs ou échec après tous les essais
                logger.error(f"Erreur API après {attempt+1} tentatives: {error_str}")
                raise
    
    def _save_to_cache(self, asset_type, symbol, timeframe, data):
        """Sauvegarde des données dans le cache."""
        key = self._cache_key(symbol, timeframe)
        self.data_cache[asset_type][key] = {
            'timestamp': time.time(),
            'data': data
        }
    
    def fetch_price_data(self, symbol, timeframes=None):
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
            
            # Vérifier si c'est une action
            try:
                stock_info = self._throttled_api_call(self.collector.mapper.get_stock_info, symbol)
                if stock_info:
                    asset_type = "stock"
                    resolved_symbol, name = stock_info
                    asset_found = True
            except Exception as e:
                logger.debug(f"Échec action: {str(e)}")
            
            # Si pas une action, tester crypto
            if not asset_found:
                try:
                    crypto_info = self._throttled_api_call(self.collector.mapper.get_crypto_info, symbol)
                    if crypto_info:
                        asset_type = "crypto"
                        resolved_symbol, name = crypto_info
                        asset_found = True
                except Exception as e:
                    logger.debug(f"Échec crypto: {str(e)}")
            
            if not asset_found:
                result["error"] = f"Actif '{symbol}' non trouvé."
                return result
            
            result["symbol"] = resolved_symbol
            result["name"] = name
            result["asset_type"] = asset_type
            
            # Charger les données historiques si nécessaire
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
            
            for timeframe in timeframes:
                # Forcer la récupération pour "current" à chaque fois
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
                            if price <= 0 or (resolved_symbol.lower() == "bitcoin" and (price < 10000 or price > 200000)):
                                logger.warning(f"Prix aberrant pour {resolved_symbol}: {price}")
                                result["data"][timeframe] = {"error": "Prix incohérent détecté"}
                            else:
                                change = float(raw_data.get("change_24h", 0)) if raw_data.get("change_24h") else 0
                                data = {
                                    "price": round(price, 2),
                                    "date": self.reference_date.strftime("%d %B %Y"),
                                    "change": round(change, 2),
                                    "volume": raw_data.get("volume", 0),
                                    "market_cap": raw_data.get("market_cap", None),
                                    "high": None,
                                    "low": None,
                                    "open": None
                                }
                                result["data"][timeframe] = data
                        else:  # Stock
                            raw_data = self._throttled_api_call(self.collector.get_stock_current, resolved_symbol)
                            if not raw_data or "price" not in raw_data:
                                result["data"][timeframe] = {"error": "Données actuelles indisponibles"}
                                continue
                            price = float(raw_data.get("price", 0))
                            if price <= 0 or price > 10000:  # Limite réaliste pour actions
                                logger.warning(f"Prix aberrant pour {resolved_symbol}: {price}")
                                result["data"][timeframe] = {"error": "Prix incohérent détecté"}
                            else:
                                change = float(raw_data.get("change", 0)) if raw_data.get("change") else 0
                                data = {
                                    "price": round(price, 2),
                                    "date": self.reference_date.strftime("%d %B %Y"),
                                    "change": round(change, 2),
                                    "volume": raw_data.get("volume", 0),
                                    "market_cap": raw_data.get("market_cap", None),
                                    "high": raw_data.get("high", None),
                                    "low": raw_data.get("low", None),
                                    "open": raw_data.get("open", None)
                                }
                                result["data"][timeframe] = data
                        self._save_to_cache(asset_type, resolved_symbol, timeframe, result["data"][timeframe])
                    
                    elif hist_data is not None:
                        if timeframe == "yesterday":
                            last_date = hist_data.index[-1]
                            row_data = hist_data.loc[last_date]
                            data = {
                                "price": float(row_data["Close"]),
                                "date": last_date.strftime("%d %B %Y"),
                                "high": float(row_data["High"]),
                                "low": float(row_data["Low"]),
                                "open": float(row_data["Open"]),
                                "volume": float(row_data["Volume"])
                            }
                            result["data"][timeframe] = data
                        
                        elif timeframe == "week":
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
                        
                        elif timeframe.isdigit():
                            year = int(timeframe)
                            year_data = hist_data[hist_data.index.year == year]
                            if not year_data.empty:
                                data = {
                                    "start_date": year_data.index[0].strftime("%d %B %Y"),
                                    "end_date": year_data.index[-1].strftime("%d %B %Y"),
                                    "max_price": float(year_data["Close"].max()),
                                    "min_price": float(year_data["Close"].min()),
                                    "change": float(((year_data["Close"].iloc[-1] / year_data["Close"].iloc[0]) - 1) * 100)
                                }
                                result["data"][timeframe] = data
                            else:
                                result["data"][timeframe] = {"error": f"Aucune donnée pour l'année {year}"}
                        
                        else:
                            specific_date = pd.to_datetime(timeframe, errors='coerce')
                            if not pd.isna(specific_date):
                                if specific_date not in hist_data.index:
                                    available_dates = hist_data.index[hist_data.index >= specific_date]
                                    if not available_dates.empty:
                                        closest_date = available_dates[0]
                                        row_data = hist_data.loc[closest_date]
                                        data = {
                                            "price": float(row_data["Close"]),
                                            "date": closest_date.strftime("%d %B %Y"),
                                            "high": float(row_data["High"]),
                                            "low": float(row_data["Low"]),
                                            "open": float(row_data["Open"]),
                                            "volume": float(row_data["Volume"]),
                                            "note": f"Date exacte non disponible, prochaine date: {closest_date.strftime('%d %B %Y')}"
                                        }
                                        result["data"][timeframe] = data
                                    else:
                                        result["data"][timeframe] = {"error": f"Aucune donnée après {timeframe}"}
                                else:
                                    row_data = hist_data.loc[specific_date]
                                    data = {
                                        "price": float(row_data["Close"]),
                                        "date": specific_date.strftime("%d %B %Y"),
                                        "high": float(row_data["High"]),
                                        "low": float(row_data["Low"]),
                                        "open": float(row_data["Open"]),
                                        "volume": float(row_data["Volume"])
                                    }
                                    result["data"][timeframe] = data
                            else:
                                result["data"][timeframe] = {"error": f"Format de date non reconnu: {timeframe}"}
                        
                        self._save_to_cache(asset_type, resolved_symbol, timeframe, result["data"][timeframe])
                    
                    else:
                        result["data"][timeframe] = {"error": "Aucune donnée historique disponible"}
                
                except Exception as e:
                    result["data"][timeframe] = {"error": f"Erreur récupération: {str(e)}"}
        
        except Exception as e:
            result["error"] = f"Erreur globale: {str(e)}"
        
        return result

    def _compare_assets(self, assets_data, timeframes=None):
        """
        Compare plusieurs actifs financiers avec une gestion robuste des données manquantes.
        """
        if not assets_data or len(assets_data) < 2:
            return {"error": "Pas assez d'actifs pour faire une comparaison"}
        
        comparison = {
            "assets": [],
            "current_prices": [],
            "performances": {},
            "winner": {},
            "error": None
        }
        
        # Récupérer les noms et prix actuels
        valid_assets = []
        for asset in assets_data:
            asset_name = asset.get("name", asset.get("symbol", ""))
            
            # Vérifier si nous avons des données utilisables pour cet actif
            has_valid_data = False
            if "data" in asset and asset["data"]:
                for timeframe, data in asset["data"].items():
                    if isinstance(data, dict) and "error" not in data:
                        has_valid_data = True
                        break
            
            if has_valid_data:
                valid_assets.append(asset)
                comparison["assets"].append(asset_name)
                
                if "data" in asset and "current" in asset["data"]:
                    current_data = asset["data"]["current"]
                    # Vérification des données avant inclusion
                    if isinstance(current_data, dict) and "error" not in current_data and "price" in current_data and current_data["price"] is not None:
                        change_value = current_data.get("change", current_data.get("change_24h", 0))
                        if change_value is None:
                            change_value = 0
                            
                        comparison["current_prices"].append({
                            "name": asset_name,
                            "price": current_data["price"],
                            "change": change_value
                        })
        
        # S'il reste moins de 2 actifs avec des données valides, on ne peut pas faire de comparaison
        if len(valid_assets) < 2:
            comparison["error"] = "Pas assez d'actifs avec des données valides pour faire une comparaison"
            return comparison
        
        # Déterminer les périodes à comparer de manière dynamique
        available_periods = set()
        
        # Identifier toutes les périodes disponibles dans les données valides
        for asset in valid_assets:
            if "data" in asset:
                for timeframe, data in asset["data"].items():
                    if isinstance(data, dict) and "error" not in data:
                        available_periods.add(timeframe)
        
        # Exclure "current" et "historical" qui sont traités différemment
        if "current" in available_periods:
            available_periods.remove("current")
        if "historical" in available_periods:
            available_periods.remove("historical")
            
        # Si aucune période spécifique n'est demandée, utiliser toutes celles disponibles
        if not timeframes:
            periods_to_compare = list(available_periods)
        else:
            # Sinon utiliser uniquement les périodes demandées qui sont disponibles
            periods_to_compare = [period for period in timeframes if period in available_periods]
        
        # Comparer chaque période
        for period in periods_to_compare:
            period_name = period
            comparison["performances"][period_name] = []
            max_change = -float('inf')
            winner = None
            
            for asset in valid_assets:
                asset_name = asset.get("name", asset.get("symbol", ""))
                change = None
                
                if "data" in asset and period in asset["data"]:
                    period_data = asset["data"][period]
                    
                    # Vérifier que les données sont valides
                    if isinstance(period_data, dict) and "error" not in period_data:
                        # Extraire la variation en fonction du type de période
                        if period in ["week", "month"]:
                            change = period_data.get("change")
                        elif period.isdigit():  # Années
                            change = period_data.get("change")
                        elif period == "yesterday" and "current" in asset["data"]:
                            # Calcul de la variation par rapport à hier si disponible
                            current_data = asset["data"]["current"]
                            yesterday_data = period_data
                            
                            if isinstance(current_data, dict) and isinstance(yesterday_data, dict):
                                current_price = current_data.get("price")
                                yesterday_price = yesterday_data.get("price")
                                
                                if current_price is not None and yesterday_price is not None and yesterday_price != 0:
                                    change = ((current_price / yesterday_price) - 1) * 100
                
                if change is not None:
                    comparison["performances"][period_name].append({
                        "name": asset_name,
                        "change": change
                    })
                    
                    # Déterminer le gagnant pour cette période
                    if change > max_change:
                        max_change = change
                        winner = asset_name
            
            # Enregistrer le gagnant pour cette période s'il y en a un
            if winner:
                comparison["winner"][period_name] = winner
        
        # Déterminer le gagnant du jour si les données actuelles sont disponibles
        day_changes = [(price.get("name"), price.get("change", 0)) 
                    for price in comparison["current_prices"] 
                    if price.get("change") is not None]
        
        if day_changes:
            day_winner = max(day_changes, key=lambda x: x[1])
            comparison["winner"]["day"] = day_winner[0]
        
        return comparison
        
    def _extract_query_info(self, query):
        """
        Utilise Claude pour extraire les informations de la requête avec une meilleure détection.
        
        Returns:
            dict: Informations de la requête
        """
        analysis_prompt = f"""
        En tant qu'assistant financier, analyse précisément la requête suivante:
        
        "{query}"
        
        Ton rôle est d'extraire les informations suivantes:
        1. S'agit-il d'une question sur les prix ou les données financières? (is_price_query) - Réponds true uniquement s'il s'agit d'une demande liée à un actif financier.
        2. Quel(s) actif(s) financier(s) est/sont mentionné(s)? (assets) - Liste de symboles/noms exacts tels que mentionnés dans la question
        3. Quelle(s) période(s) est/sont concernée(s)? (timeframes) - Liste parmi:
        - current (aujourd'hui, actuel, maintenant)
        - yesterday (hier, veille)
        - week (semaine, 7 derniers jours)
        - month (mois, 30 derniers jours)
        - year/YYYY (année spécifique comme 2023, 2024)
        - [date spécifique au format YYYY-MM-DD]
        4. Quel type d'analyse est demandé? (analysis_type) - Un ou plusieurs parmi:
        - current_price (prix actuel simple)
        - historical_price (prix historique à une date)
        - comparison (comparaison entre périodes ou actifs)
        - trend (tendance sur une période)
        - volatility (volatilité)
        - high_low (plus haut/plus bas)
        - average (moyenne)
        - threshold (dépassement d'un seuil)
        5. Thresholds/valeurs mentionnés? (specific_values) - Liste des valeurs numériques mentionnées à analyser
        6. Dans quelle langue est la requête? (language) - "fr" pour français, "en" pour anglais, etc.
        
        IMPORTANT: Si un actif est mentionné, même sans demande explicite de prix, considère qu'il s'agit probablement d'une requête sur le prix actuel.
        
        Fournis une réponse au format JSON exact:
        {{
            "is_price_query": true/false,
            "assets": ["btc", "aapl", ...],
            "timeframes": ["current", "yesterday", ...],
            "analysis_type": ["comparison", "trend", ...],
            "specific_values": [150, 30000, ...],
            "language": "fr"
        }}
        """
        
        try:
            # Gestion plus robuste des erreurs et des tentatives multiples
            for attempt in range(self.max_retries):
                try:
                    analysis_response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content": analysis_prompt}]
                    )
                    
                    if analysis_response and analysis_response.content:
                        json_text = re.search(r'({.*})', analysis_response.content[0].text.replace('\n', ' '), re.DOTALL)
                        if json_text:
                            try:
                                query_info = json.loads(json_text.group(1))
                                logger.info(f"Informations extraites de la requête: {query_info}")
                                
                                # Si un asset est détecté mais is_price_query est False, le corriger
                                if "assets" in query_info and query_info["assets"] and not query_info.get("is_price_query", False):
                                    query_info["is_price_query"] = True
                                    if "analysis_type" not in query_info or not query_info["analysis_type"]:
                                        query_info["analysis_type"] = ["current_price"]
                                
                                # Si aucune période spécifiée mais qu'il y a une requête de prix, ajouter "current"
                                if query_info.get("is_price_query", False) and (
                                    "timeframes" not in query_info or not query_info["timeframes"]
                                ):
                                    query_info["timeframes"] = ["current"]
                                
                                return query_info
                            except json.JSONDecodeError:
                                logger.error("Erreur de décodage JSON dans la réponse")
                                if attempt == self.max_retries - 1:
                                    break
                    time.sleep(1)  # Petite pause avant de réessayer
                except Exception as e:
                    logger.error(f"Erreur à la tentative {attempt+1} d'analyse de la requête: {str(e)}")
                    if attempt == self.max_retries - 1:
                        break
                    time.sleep(self.retry_delay / self.max_retries)
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la requête: {str(e)}")
            
        # Détection manuelle simplifiée en cas d'échec de l'analyse
        language = "fr"  # Par défaut en français
        english_patterns = r'^(hi|hello|good|what|how|why|when|which|where|is|can|should|who|tell|explain)'
        if re.match(english_patterns, query.strip().lower()):
            language = "en"
            
        return {
            "is_price_query": False,
            "assets": [],
            "timeframes": ["current"],
            "analysis_type": [],
            "specific_values": [],
            "language": language
        }
    
    def _generate_system_prompt(self, query, financial_data=None):
        """
        Générateur de prompt système amélioré pour des réponses plus naturelles et concises.
        """
        date_str = self.reference_date.strftime('%d %B %Y')
        
        # Historique de conversation limité pour contexte
        history_data = self.memory.load_memory_variables({})['chat_history']
        history_str = ""
        
        if history_data:
            history_str = "Contexte des échanges récents :\n"
            for i, msg in enumerate(history_data[-2:]):  # Limité aux 2 derniers échanges pour plus de concision
                role = "Utilisateur" if i % 2 == 0 else "Assistant"
                history_str += f"{role} : {msg.content}\n"
        
        # Formatage des données financières de façon concise
        financial_data_str = ""
        if financial_data:
            financial_data_str = "\n===== DONNÉES FINANCIÈRES =====\n"
            
            for asset_info in financial_data:
                asset_name = asset_info.get('name', asset_info.get('symbol', ''))
                financial_data_str += f"\n--- {asset_name} ({asset_info.get('symbol', '')}) ---\n"
                
                for timeframe, data in asset_info.get('data', {}).items():
                    if isinstance(data, dict):
                        if 'error' in data:
                            financial_data_str += f"⚠️ {timeframe}: {data['error']}\n"
                            continue
                            
                        if timeframe == 'current':
                            financial_data_str += f"Prix actuel: {data.get('price', 'N/A')} $ "
                            if 'change_24h' in data:
                                financial_data_str += f"(var 24h: {data.get('change_24h', 0):+.2f}%)\n"
                            elif 'change' in data:
                                financial_data_str += f"(var: {data.get('change', 0):+.2f}%)\n"
                            else:
                                financial_data_str += "\n"
                                
                        elif timeframe == 'yesterday':
                            financial_data_str += f"Prix ({data.get('date', 'hier')}): {data.get('price', 'N/A')} $\n"
                            
                        elif timeframe == 'week':
                            financial_data_str += f"Semaine: +Haut: {data.get('high', 'N/A')} $ | +Bas: {data.get('low', 'N/A')} $ | Var: {data.get('change', 0):+.2f}%\n"
                                
                        elif timeframe == 'month':
                            financial_data_str += f"Mois: +Haut: {data.get('high', 'N/A')} $ | +Bas: {data.get('low', 'N/A')} $ | Var: {data.get('change', 0):+.2f}%\n"
                        
                        elif timeframe.isdigit():
                            financial_data_str += f"Année {timeframe}: +Haut: {data.get('max_price', 'N/A')} $ | +Bas: {data.get('min_price', 'N/A')} $ | Var: {data.get('change', 0):+.2f}%\n"
                            
                        elif timeframe == 'historical':
                            financial_data_str += f"Données historiques: De {data.get('available_from', 'N/A')} à {data.get('available_to', 'N/A')}\n"
                            
                        else:
                            financial_data_str += f"Prix ({data.get('date', timeframe)}): {data.get('price', 'N/A')} $\n"
                            if data.get('note'):
                                financial_data_str += f"Note: {data.get('note')}\n"
                
                if asset_info.get('error'):
                    financial_data_str += f"⚠️ Erreur: {asset_info['error']}\n"
            
            financial_data_str += "\n==============================\n"
        
        # Extraction de la langue de la requête
        query_info = self._extract_query_info(query)
        language = query_info.get("language", "fr")
        
        # Personnalité et instructions - adaptées selon la langue
        if language == "en":
            system_prompt = f"""
            You are Finn, a financial assistant with the following characteristics:
            
            ## Core behaviors
            - Answer financial questions directly and concisely
            - Use a warm, natural conversational style (but keep it brief)
            - Vary your greetings and responses to sound human and engaging
            - For price requests, answer in 1-2 sentences maximum
            - NEVER mention technical processes (APIs, databases, etc.)
            - Speak conversationally using occasional informal expressions ("you know", "actually", etc.)
            
            ## Response style
            - For simple price questions: 1-2 sentences max
            - For analysis: Brief, accessible insights (3-4 sentences max)
            - For comparisons: Short, clear statements with key figures
            - Use varied sentence structures that sound natural
            - Start with the most important information
            - Show personality but prioritize brevity
            
            ## Current date: {date_str}
            
            ## Conversation context
            {history_str if history_str else "No recent exchanges to consider"}
            
            ## Available financial data
            {financial_data_str if financial_data_str else "No financial data is available yet for this query."}
            
            ## Current question
            {query}
            """
        else:  # Default to French
            system_prompt = f"""
            Tu es Finn, un assistant financier avec les caractéristiques suivantes:
            
            ## Comportements essentiels
            - Réponds aux questions financières directement et de façon concise
            - Utilise un style conversationnel chaleureux mais bref
            - Varie tes salutations et réponses pour paraître humain et engageant
            - Pour les demandes de prix, réponds en 1-2 phrases maximum
            - Ne mentionne JAMAIS les processus techniques (APIs, bases de données, etc.)
            - Parle de façon conversationnelle en utilisant occasionnellement des expressions informelles ("tu vois", "en fait", etc.)
            
            ## Style de réponse
            - Pour les questions de prix simples: 1-2 phrases maximum
            - Pour les analyses: Aperçus brefs et accessibles (3-4 phrases max)
            - Pour les comparaisons: Déclarations courtes et claires avec chiffres clés
            - Utilise des structures de phrases variées qui sonnent naturelles
            - Commence par l'information la plus importante
            - Montre de la personnalité mais privilégie la brièveté
            
            ## Date actuelle: {date_str}
            
            ## Contexte de la conversation
            {history_str if history_str else "Aucun échange récent à considérer"}
            
            ## Données financières disponibles
            {financial_data_str if financial_data_str else "Aucune donnée financière n'est encore disponible pour cette requête."}
            
            ## Question actuelle
            {query}
            """
        
        return system_prompt

    def _optimize_response(self, response_text, query_info):
        """
        Optimise la réponse pour la rendre plus concise et naturelle selon le type de requête.
        
        Args:
            response_text (str): Texte de réponse brut
            query_info (dict): Informations extraites de la requête
            
        Returns:
            str: Réponse optimisée
        """
        # Si c'est une demande simple de prix, s'assurer que la réponse est concise
        if len(query_info.get("assets", [])) == 1 and "current_price" in query_info.get("analysis_type", []):
            # Diviser la réponse en phrases
            sentences = re.split(r'(?<=[.!?])\s+', response_text)
            
            # Pour une requête de prix simple, limiter à 1-2 phrases maximum
            if len(sentences) > 2:
                # Conserver les phrases avec des données numériques
                meaningful_sentences = []
                for sentence in sentences:
                    if re.search(r'\d', sentence):
                        meaningful_sentences.append(sentence)
                        if len(meaningful_sentences) >= 2:
                            break
                
                # S'assurer d'avoir au moins une phrase
                if not meaningful_sentences and sentences:
                    meaningful_sentences.append(sentences[0])
                
                optimized_response = ' '.join(meaningful_sentences)
                
                # S'assurer que le style reste conversationnel
                if not re.search(r'\b(eh bien|tu sais|en fait|d\'ailleurs|vraiment)\b', optimized_response, re.IGNORECASE):
                    conversation_starters = [
                        "Eh bien, ", "Tu sais, ", "En fait, ", "Actuellement, ", ""
                    ]
                    if not optimized_response.startswith(tuple(conversation_starters)):
                        import random
                        optimized_response = random.choice(conversation_starters) + optimized_response
                
                return optimized_response
        
        # Pour les comparaisons, limiter à environ 3-4 phrases
        elif "comparison" in query_info.get("analysis_type", []) and len(response_text.split()) > 80:
            sentences = re.split(r'(?<=[.!?])\s+', response_text)
            
            # Sélectionner les phrases les plus importantes
            if len(sentences) > 4:
                # Prioriser les phrases contenant des comparaisons directes
                comparison_sentences = []
                for sentence in sentences:
                    if re.search(r'(plus|moins|mieux|meilleur|pire|supérieur|inférieur|gagnant|surperform|performant)', sentence.lower()):
                        comparison_sentences.append(sentence)
                
                # Ajouter les phrases avec des pourcentages
                percentage_sentences = []
                for sentence in sentences:
                    if re.search(r'\d+(\.\d+)?%', sentence) and sentence not in comparison_sentences:
                        percentage_sentences.append(sentence)
                
                # Combiner avec un maximum de 4 phrases
                meaningful_sentences = comparison_sentences + percentage_sentences
                if len(meaningful_sentences) > 4:
                    meaningful_sentences = meaningful_sentences[:4]
                elif len(meaningful_sentences) < 2 and sentences:
                    # Si pas assez de phrases significatives, ajouter la première phrase
                    if sentences[0] not in meaningful_sentences:
                        meaningful_sentences.insert(0, sentences[0])
                
                return ' '.join(meaningful_sentences)
        
        # Pour les autres types de requêtes, limiter en longueur si nécessaire
        elif len(response_text.split()) > 75:
            sentences = re.split(r'(?<=[.!?])\s+', response_text)
            
            # Prioriser les phrases avec des données numériques
            numerical_sentences = []
            for sentence in sentences:
                if re.search(r'\d', sentence):
                    numerical_sentences.append(sentence)
            
            # Ajouter les phrases avec des termes significatifs
            significant_sentences = []
            for sentence in sentences:
                if re.search(r'(tendance|évolue|analyse|baisse|hausse|progression|recul|performance)', sentence.lower()) and sentence not in numerical_sentences:
                    significant_sentences.append(sentence)
            
            # Combiner en conservant l'ordre original
            original_indices = {}
            for i, sentence in enumerate(sentences):
                if sentence in numerical_sentences or sentence in significant_sentences:
                    original_indices[sentence] = i
            
            # Trier selon l'ordre original et limiter à 4-5 phrases
            meaningful_sentences = sorted(
                list(numerical_sentences) + list(significant_sentences),
                key=lambda s: original_indices.get(s, len(sentences))
            )
            
            if len(meaningful_sentences) > 5:
                meaningful_sentences = meaningful_sentences[:5]
            elif len(meaningful_sentences) < 2 and sentences:
                # Si pas assez de phrases significatives, ajouter la première phrase
                if sentences[0] not in meaningful_sentences:
                    meaningful_sentences.insert(0, sentences[0])
                # Et éventuellement la dernière phrase
                if len(sentences) > 1 and sentences[-1] not in meaningful_sentences:
                    meaningful_sentences.append(sentences[-1])
            
            return ' '.join(meaningful_sentences)
        
        return response_text
    
    def process_query(self, query):
        """
        Traite la requête utilisateur et génère une réponse pertinente avec une gestion améliorée
        des erreurs et des réponses plus concises.
        
        Args:
            query (str): Requête utilisateur
        
        Returns:
            str: Réponse de l'assistant
        """
        # Étape 1: Extraire les informations de la requête
        query_info = self._extract_query_info(query)
        
        # Étape 2: Si c'est une requête financière, récupérer les données nécessaires
        financial_data = []
        comparison_data = None
        
        if query_info.get("is_price_query", False) and query_info.get("assets"):
            for asset in query_info.get("assets", []):
                # Déterminer tous les timeframes nécessaires en fonction de l'analyse demandée
                timeframes = query_info.get("timeframes", ["current"])
                
                # Pour les comparaisons, s'assurer d'avoir hier si on a aujourd'hui
                if "comparison" in query_info.get("analysis_type", []) and len(financial_data) > 1:
                    try:
                        comparison_data = self._compare_assets(financial_data, query_info.get("timeframes"))
                        
                        # Vérifier si la comparaison a échoué
                        if comparison_data.get("error"):
                            error_msg = comparison_data.get("error")
                            if query_info.get("language") == "en":
                                system_prompt += f"\n\nNOTE: Comparison failed: {error_msg}. Please inform the user you couldn't compare the assets right now due to data limitations, but you can still discuss individual assets. Be conversational and natural."
                            else:
                                system_prompt += f"\n\nNOTE: Comparaison échouée: {error_msg}. Veuillez informer l'utilisateur que vous ne pouvez pas comparer les actifs pour le moment en raison de limitations des données, mais que vous pouvez toujours discuter des actifs individuellement. Soyez conversationnel et naturel."
                        else:
                            # Ajouter les données de comparaison au prompt
                            if query_info.get("language") == "en":
                                system_prompt += "\n\n===== COMPARISON DATA =====\n"
                                system_prompt += f"Assets compared: {', '.join(comparison_data.get('assets', []))}\n"
                                
                                # Ajouter les performances par période
                                for period, performances in comparison_data.get("performances", {}).items():
                                    system_prompt += f"\nPerformance over {period}:\n"
                                    for perf in sorted(performances, key=lambda x: x.get("change", 0), reverse=True):
                                        system_prompt += f"- {perf.get('name')}: {perf.get('change', 0):+.2f}%\n"
                                
                                # Ajouter les gagnants par période
                                system_prompt += "\nBest performance:\n"
                                for period, winner in comparison_data.get("winner", {}).items():
                                    if winner:
                                        system_prompt += f"- {period}: {winner}\n"
                            else:
                                system_prompt += "\n\n===== DONNÉES DE COMPARAISON =====\n"
                                system_prompt += f"Actifs comparés: {', '.join(comparison_data.get('assets', []))}\n"
                                
                                # Ajouter les performances par période
                                for period, performances in comparison_data.get("performances", {}).items():
                                    system_prompt += f"\nPerformances sur {period}:\n"
                                    for perf in sorted(performances, key=lambda x: x.get("change", 0), reverse=True):
                                        system_prompt += f"- {perf.get('name')}: {perf.get('change', 0):+.2f}%\n"
                                
                                # Ajouter les gagnants par période
                                system_prompt += "\nMeilleure performance:\n"
                                for period, winner in comparison_data.get("winner", {}).items():
                                    if winner:
                                        system_prompt += f"- {period}: {winner}\n"
                    except Exception as e:
                        logger.error(f"Erreur lors de la comparaison des actifs: {str(e)}")
                        if query_info.get("language") == "en":
                            system_prompt += "\n\nNOTE: Asset comparison failed due to technical limitations. Please inform the user naturally that you don't have enough data to compare these assets right now."
                        else:
                            system_prompt += "\n\nNOTE: La comparaison des actifs a échoué en raison de limitations techniques. Veuillez informer l'utilisateur de façon naturelle que vous n'avez pas assez de données pour comparer ces actifs actuellement."
                                
                # Pour les tendances, ajouter la semaine/mois si non spécifié
                if "trend" in query_info.get("analysis_type", []) and "week" not in timeframes and "month" not in timeframes:
                    timeframes.append("week")
                
                # Pour la volatilité, s'assurer d'avoir au moins une semaine de données
                if "volatility" in query_info.get("analysis_type", []) and "week" not in timeframes:
                    timeframes.append("week")
                
                # Pour high/low, s'assurer d'avoir la période appropriée
                if "high_low" in query_info.get("analysis_type", []) and not any(t.isdigit() for t in timeframes):
                    current_year = str(self.reference_date.year)
                    if current_year not in timeframes:
                        timeframes.append(current_year)
                
                # Récupérer les données pour cet actif avec tous les timeframes nécessaires
                try:
                    asset_data = self.fetch_price_data(asset, timeframes)
                    financial_data.append(asset_data)
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des données pour {asset}: {str(e)}")
                    financial_data.append({
                        "symbol": asset,
                        "error": f"Impossible de récupérer les données: {str(e)}"
                    })
            
            # Si c'est une comparaison entre actifs, préparer ces données
            if "comparison" in query_info.get("analysis_type", []) and len(financial_data) > 1:
                comparison_data = self._compare_assets(financial_data, query_info.get("timeframes"))
        
        # Étape 3: Générer le prompt système avec les données
        system_prompt = self._generate_system_prompt(query, financial_data if financial_data else None)
        
        # Ajouter des instructions spécifiques selon le type de requête
        if len(query_info.get("assets", [])) == 1 and "current_price" in query_info.get("analysis_type", []):
            # Pour les demandes de prix simples, demander une réponse concise
            if query_info.get("language") == "en":
                system_prompt += "\n\nThis is a simple price request. Answer in a very concise but conversational way (1-2 sentences)."
            else:
                system_prompt += "\n\nCette requête porte sur un prix actuel simple. Réponds de façon très concise mais conversationnelle (1-2 phrases)."
        
        if comparison_data:
            # Ajouter les données de comparaison au prompt
            if query_info.get("language") == "en":
                system_prompt += "\n\n===== COMPARISON DATA =====\n"
                system_prompt += f"Assets compared: {', '.join(comparison_data.get('assets', []))}\n"
                
                # Ajouter les performances par période
                for period, performances in comparison_data.get("performances", {}).items():
                    system_prompt += f"\nPerformance over {period}:\n"
                    for perf in sorted(performances, key=lambda x: x.get("change", 0), reverse=True):
                        system_prompt += f"- {perf.get('name')}: {perf.get('change', 0):+.2f}%\n"
                
                # Ajouter les gagnants par période
                system_prompt += "\nBest performance:\n"
                for period, winner in comparison_data.get("winner", {}).items():
                    if winner:
                        system_prompt += f"- {period}: {winner}\n"
            else:
                system_prompt += "\n\n===== DONNÉES DE COMPARAISON =====\n"
                system_prompt += f"Actifs comparés: {', '.join(comparison_data.get('assets', []))}\n"
                
                # Ajouter les performances par période
                for period, performances in comparison_data.get("performances", {}).items():
                    system_prompt += f"\nPerformances sur {period}:\n"
                    for perf in sorted(performances, key=lambda x: x.get("change", 0), reverse=True):
                        system_prompt += f"- {perf.get('name')}: {perf.get('change', 0):+.2f}%\n"
                
                # Ajouter les gagnants par période
                system_prompt += "\nMeilleure performance:\n"
                for period, winner in comparison_data.get("winner", {}).items():
                    if winner:
                        system_prompt += f"- {period}: {winner}\n"
        
        # Étape 4: Appeler Claude pour obtenir une réponse
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "user", "content": system_prompt}
                    ]
                )
                if response and response.content:
                    resp_text = response.content[0].text.strip()
                    
                    # Post-traitement pour rendre les réponses plus concises
                    resp_text = self._optimize_response(resp_text, query_info)
                    
                    # Enregistrer l'échange dans la mémoire
                    self.memory.save_context({"input": query}, {"output": resp_text})
                    self.chat_history.extend([query, resp_text])
                    return resp_text
                
                return "Je n'ai pas pu analyser complètement cette requête. Pourriez-vous la reformuler?"
            
            except anthropic.RateLimitError:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Quota Claude atteint, nouvelle tentative dans {self.retry_delay}s")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Quota Claude définitivement atteint")
                    return "Je rencontre des limitations techniques temporaires. Pourriez-vous réessayer plus tard?"
            
            except Exception as e:
                logger.error(f"Erreur lors de l'appel à Claude: {str(e)}")
                if financial_data and any(data.get("error") for data in financial_data):
                    return "Je n'ai pas pu récupérer toutes les données financières nécessaires. Certaines informations pourraient ne pas être disponibles actuellement."
                return "Je rencontre des difficultés pour traiter cette requête. Pourriez-vous la reformuler différemment?"
        
        return "Je ne peux pas répondre à cette question pour le moment. Veuillez réessayer plus tard."


if __name__ == "__main__":
    handler = ClaudeHandler()
    print("Finn est prêt ! Pose tes questions (ou 'exit' pour quitter) :")
    
    while True:
        query = input("\n> ")
        if query.lower() == 'exit':
            print("À bientôt !")
            break
        response = handler.process_query(query)
        print(f"\n{response}")