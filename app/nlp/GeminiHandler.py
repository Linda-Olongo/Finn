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
import google.generativeai as genai
from google.api_core import exceptions
from langchain.memory import ConversationBufferMemory

# Import des modules
from data.api_fetcher import DataCollector

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

class GeminiHandler:
    def __init__(self):
        """Initialisation de l'assistant avec Gemini."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Clé GEMINI_API_KEY introuvable dans .env")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.chat = self.model.start_chat(history=[])
        self.retry_delay = 60
        self.max_retries = 3
        
        # Mémoire LangChain avec capacité augmentée
        try:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                max_len=20,  # Plus d'échanges pour un meilleur contexte
                return_messages=True
            )
        except Exception as e:
            logger.warning(f"Avertissement lors de l'initialisation de la mémoire: {e}")
            # Fallback pour éviter l'erreur de dépréciation
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        # Date dynamique
        self.reference_date = datetime.now()
        
        # Cache pour les données financières (réduit les appels API)
        self.data_cache = {
            'crypto': {},  # Format: {'btc': {'current': {...}, 'yesterday': {...}}}
            'stock': {}    # Format: {'aapl': {'current': {...}, 'yesterday': {...}}}
        }
        
        # Durée de validité du cache (en secondes)
        self.cache_ttl = 300  # 5 minutes
        
        # Collector d'API (unique pour éviter trop de connexions)
        self.collector = DataCollector()
    
    def _cache_key(self, symbol, timeframe):
        """Génère une clé de cache pour les données financières."""
        return f"{symbol.lower()}_{timeframe}"
    
    def _get_from_cache(self, asset_type, symbol, timeframe):
        """Récupère des données du cache si disponibles et valides."""
        key = self._cache_key(symbol, timeframe)
        cache = self.data_cache[asset_type].get(key, {})
        
        if not cache:
            return None
        
        # Vérifier si le cache est encore valide
        cache_time = cache.get('timestamp', 0)
        if time.time() - cache_time > self.cache_ttl:
            return None
            
        return cache.get('data')
    
    def _save_to_cache(self, asset_type, symbol, timeframe, data):
        """Sauvegarde des données dans le cache."""
        key = self._cache_key(symbol, timeframe)
        self.data_cache[asset_type][key] = {
            'timestamp': time.time(),
            'data': data
        }
    
    def fetch_price_data(self, symbol, timeframes=None):
        """
        Récupère les données de prix pour un ou plusieurs timeframes.
        
        Args:
            symbol (str): Nom ou symbole de l'actif
            timeframes (list/str, optional): Liste de périodes ou période unique
                
        Returns:
            dict: Données de prix avec un champ pour chaque timeframe demandé
        """
        if timeframes is None:
            timeframes = ["current"]
        elif isinstance(timeframes, str):
            timeframes = [timeframes]
            
        result = {"symbol": symbol, "name": None, "asset_type": None, "data": {}, "error": None}
        
        try:
            # Optimisation: d'abord chercher dans les actions pour les symboles courts comme "AAPL"
            stock_first = len(symbol) <= 5 and symbol.isalpha() and symbol.isupper()
            
            if stock_first:
                # Chercher d'abord dans les actions
                stock_info = self.collector.mapper.get_stock_info(symbol)
                if stock_info:
                    asset_type = "stock"
                    resolved_symbol, name = stock_info
                else:
                    # Puis dans les cryptos
                    crypto_info = self.collector.mapper.get_crypto_info(symbol)
                    if crypto_info:
                        asset_type = "crypto"
                        resolved_symbol, name = crypto_info
                    else:
                        result["error"] = f"Actif '{symbol}' non trouvé."
                        return result
            else:
                # Chercher d'abord dans les cryptos
                crypto_info = self.collector.mapper.get_crypto_info(symbol)
                if crypto_info:
                    asset_type = "crypto"
                    resolved_symbol, name = crypto_info
                else:
                    # Puis dans les actions
                    stock_info = self.collector.mapper.get_stock_info(symbol)
                    if stock_info:
                        asset_type = "stock"
                        resolved_symbol, name = stock_info
                    else:
                        result["error"] = f"Actif '{symbol}' non trouvé."
                        return result
            
            result["symbol"] = resolved_symbol
            result["name"] = name
            result["asset_type"] = asset_type
            
            # Récupérer les données pour chaque timeframe demandé
            for timeframe in timeframes:
                # Vérifier le cache d'abord
                cached_data = self._get_from_cache(asset_type, resolved_symbol, timeframe)
                if cached_data:
                    logger.info(f"Données récupérées du cache pour {resolved_symbol}, {timeframe}")
                    result["data"][timeframe] = cached_data
                    continue
                    
                # Si pas dans le cache, récupérer depuis les APIs
                try:
                    if timeframe == "current":
                        if asset_type == "crypto":
                            data = self.collector.get_crypto_current(resolved_symbol)
                            data["date"] = self.reference_date.strftime("%d %B %Y")
                        else:
                            data = self.collector.get_stock_current(resolved_symbol)
                            data["date"] = self.reference_date.strftime("%d %B %Y")
                        
                        result["data"][timeframe] = data
                        self._save_to_cache(asset_type, resolved_symbol, timeframe, data)
                        
                    elif timeframe == "yesterday":
                        hist_data = (self.collector.get_crypto_historical if asset_type == "crypto" 
                                else self.collector.get_stock_historical)(resolved_symbol)
                        
                        yesterday = self.reference_date - timedelta(days=1)
                        yesterday_str = yesterday.strftime("%Y-%m-%d")
                        
                        if yesterday_str in hist_data.index:
                            close_price = hist_data.loc[yesterday_str, "Close"]
                            data = {
                                "price": close_price,
                                "date": yesterday.strftime("%d %B %Y"),
                                "high": hist_data.loc[yesterday_str, "High"] if "High" in hist_data.columns else None,
                                "low": hist_data.loc[yesterday_str, "Low"] if "Low" in hist_data.columns else None,
                                "open": hist_data.loc[yesterday_str, "Open"] if "Open" in hist_data.columns else None,
                                "volume": hist_data.loc[yesterday_str, "Volume"] if "Volume" in hist_data.columns else None,
                            }
                            result["data"][timeframe] = data
                            self._save_to_cache(asset_type, resolved_symbol, timeframe, data)
                        else:
                            # Essayer de trouver le jour ouvré précédent
                            # Trier l'index par date décroissante et prendre la première date avant aujourd'hui
                            business_days = hist_data.index[hist_data.index < self.reference_date.strftime("%Y-%m-%d")]
                            if len(business_days) > 0:
                                last_business_day = business_days[-1]
                                close_price = hist_data.loc[last_business_day, "Close"]
                                last_date = pd.to_datetime(last_business_day)
                                data = {
                                    "price": close_price,
                                    "date": last_date.strftime("%d %B %Y"),
                                    "high": hist_data.loc[last_business_day, "High"] if "High" in hist_data.columns else None,
                                    "low": hist_data.loc[last_business_day, "Low"] if "Low" in hist_data.columns else None,
                                    "open": hist_data.loc[last_business_day, "Open"] if "Open" in hist_data.columns else None,
                                    "volume": hist_data.loc[last_business_day, "Volume"] if "Volume" in hist_data.columns else None,
                                }
                                result["data"][timeframe] = data
                                self._save_to_cache(asset_type, resolved_symbol, timeframe, data)
                            else:
                                result["error"] = f"Données indisponibles pour {timeframe}."
                        
                    elif timeframe == "week":
                        hist_data = (self.collector.get_crypto_historical if asset_type == "crypto" 
                                else self.collector.get_stock_historical)(resolved_symbol)
                        
                        week_ago = self.reference_date - timedelta(days=7)
                        # Utiliser date directement pour éviter l'erreur str accessor
                        week_data = hist_data[hist_data.index >= week_ago.strftime("%Y-%m-%d")]
                        
                        if not week_data.empty:
                            # Format de données amélioré
                            data = {
                                "start_date": pd.to_datetime(week_data.index[0]).strftime("%d %B %Y"),
                                "end_date": pd.to_datetime(week_data.index[-1]).strftime("%d %B %Y"),
                                "prices": week_data["Close"].round(2).tolist(),
                                "dates": [pd.to_datetime(date).strftime("%d %B %Y") for date in week_data.index],
                                "high": week_data["Close"].max(),
                                "low": week_data["Close"].min(),
                                "high_date": pd.to_datetime(week_data["Close"].idxmax()).strftime("%d %B %Y"),
                                "low_date": pd.to_datetime(week_data["Close"].idxmin()).strftime("%d %B %Y"),
                                "average": week_data["Close"].mean(),
                                "change": ((week_data["Close"].iloc[-1] / week_data["Close"].iloc[0]) - 1) * 100,
                                "volatility": week_data["Close"].pct_change().std() * 100  # Volatilité en %
                            }
                            result["data"][timeframe] = data
                            self._save_to_cache(asset_type, resolved_symbol, timeframe, data)
                        else:
                            result["error"] = f"Données indisponibles pour {timeframe}."
                            
                    elif timeframe == "month":
                        hist_data = (self.collector.get_crypto_historical if asset_type == "crypto" 
                                else self.collector.get_stock_historical)(resolved_symbol)
                        
                        month_ago = self.reference_date - timedelta(days=30)
                        month_data = hist_data[hist_data.index >= month_ago.strftime("%Y-%m-%d")]
                        
                        if not month_data.empty:
                            data = {
                                "start_date": pd.to_datetime(month_data.index[0]).strftime("%d %B %Y"),
                                "end_date": pd.to_datetime(month_data.index[-1]).strftime("%d %B %Y"),
                                "prices": month_data["Close"].round(2).tolist(),
                                "dates": [pd.to_datetime(date).strftime("%d %B %Y") for date in month_data.index],
                                "high": month_data["Close"].max(),
                                "low": month_data["Close"].min(),
                                "high_date": pd.to_datetime(month_data["Close"].idxmax()).strftime("%d %B %Y"),
                                "low_date": pd.to_datetime(month_data["Close"].idxmin()).strftime("%d %B %Y"),
                                "average": month_data["Close"].mean(),
                                "change": ((month_data["Close"].iloc[-1] / month_data["Close"].iloc[0]) - 1) * 100,
                                "volatility": month_data["Close"].pct_change().std() * 100
                            }
                            result["data"][timeframe] = data
                            self._save_to_cache(asset_type, resolved_symbol, timeframe, data)
                        else:
                            result["error"] = f"Données indisponibles pour {timeframe}."
                    
                    elif timeframe.isdigit():  # Année spécifique
                        hist_data = (self.collector.get_crypto_historical if asset_type == "crypto" 
                                else self.collector.get_stock_historical)(resolved_symbol)
                        
                        # Vérifier si les indices sont des strings ou des timestamps
                        if isinstance(hist_data.index[0], str):
                            year_data = hist_data[hist_data.index.str.startswith(timeframe)]
                        else:
                            # Si les indices sont des timestamps, convertir au format string pour filtrer
                            hist_data.index = hist_data.index.strftime("%Y-%m-%d")
                            year_data = hist_data[hist_data.index.str.startswith(timeframe)]
                        
                        if not year_data.empty:
                            data = {
                                "start_date": pd.to_datetime(year_data.index[0]).strftime("%d %B %Y") if len(year_data) > 0 else None,
                                "end_date": pd.to_datetime(year_data.index[-1]).strftime("%d %B %Y") if len(year_data) > 0 else None,
                                "max_price": year_data["Close"].max(),
                                "max_date": pd.to_datetime(year_data["Close"].idxmax()).strftime("%d %B %Y"),
                                "min_price": year_data["Close"].min(),
                                "min_date": pd.to_datetime(year_data["Close"].idxmin()).strftime("%d %B %Y"),
                                "average": year_data["Close"].mean(),
                                "median": year_data["Close"].median(),
                                "change": ((year_data["Close"].iloc[-1] / year_data["Close"].iloc[0]) - 1) * 100 if len(year_data) > 1 else 0,
                                "volatility": year_data["Close"].pct_change().std() * 100 if len(year_data) > 1 else 0
                            }
                            result["data"][timeframe] = data
                            self._save_to_cache(asset_type, resolved_symbol, timeframe, data)
                        else:
                            result["error"] = f"Données indisponibles pour {timeframe}."
                    
                    else:
                        # Pour une date spécifique ou autre période plus complexe
                        hist_data = (self.collector.get_crypto_historical if asset_type == "crypto" 
                                else self.collector.get_stock_historical)(resolved_symbol)
                        
                        # Convertir le timeframe en date si possible (format "YYYY-MM-DD" ou "DD/MM/YYYY")
                        specific_date = None
                        date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]
                        
                        for fmt in date_formats:
                            try:
                                specific_date = datetime.strptime(timeframe, fmt)
                                break
                            except ValueError:
                                continue
                                
                        if specific_date:
                            # Formater la date pour correspondre au format d'index
                            date_str = specific_date.strftime("%Y-%m-%d")
                            if date_str in hist_data.index:
                                close_price = hist_data.loc[date_str, "Close"]
                                data = {
                                    "price": close_price,
                                    "date": specific_date.strftime("%d %B %Y"),
                                    "high": hist_data.loc[date_str, "High"] if "High" in hist_data.columns else None,
                                    "low": hist_data.loc[date_str, "Low"] if "Low" in hist_data.columns else None,
                                    "open": hist_data.loc[date_str, "Open"] if "Open" in hist_data.columns else None,
                                    "volume": hist_data.loc[date_str, "Volume"] if "Volume" in hist_data.columns else None
                                }
                                result["data"][timeframe] = data
                                self._save_to_cache(asset_type, resolved_symbol, timeframe, data)
                            else:
                                result["error"] = f"Données indisponibles pour {timeframe}."
                        else:
                            # S'il ne s'agit pas d'une date spécifique, fournir l'historique général
                            data = {
                                "available_from": pd.to_datetime(hist_data.index[0]).strftime("%d %B %Y") if len(hist_data) > 0 else None,
                                "available_to": pd.to_datetime(hist_data.index[-1]).strftime("%d %B %Y") if len(hist_data) > 0 else None,
                                "prices": hist_data["Close"].round(2).tolist()[-30:],  # Derniers 30 jours seulement pour limiter la taille
                                "dates": [pd.to_datetime(date).strftime("%d %B %Y") for date in hist_data.index][-30:]
                            }
                            result["data"]["historical"] = data
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des données pour {timeframe}: {str(e)}")
                    # Continuer avec les autres timeframes malgré l'erreur
                    result["data"][timeframe] = {"error": str(e)}
                    
                # Pause entre les requêtes pour éviter de dépasser les limites d'API
                time.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Erreur globale lors de la récupération des données: {str(e)}")
            result["error"] = f"Erreur lors de la récupération des données: {str(e)}"
            
        return result
        
    def _extract_query_info(self, query):
        """
        Utilise Gemini pour extraire les informations de la requête.
        
        Returns:
            dict: Informations de la requête
        """
        analysis_prompt = f"""
        En tant qu'assistant financier, analyse la requête suivante:
        
        "{query}"
        
        Ton rôle est d'extraire les informations suivantes:
        1. S'agit-il d'une question sur les prix ou les données financières? (is_price_query)
        2. Quel(s) actif(s) financier(s) est/sont mentionné(s)? (assets) - Liste de symboles/noms
        3. Quelle(s) période(s) est/sont concernée(s)? (timeframes) - Liste parmi:
        - current (aujourd'hui, actuel)
        - yesterday (hier, veille)
        - week (semaine, 7 derniers jours)
        - month (mois, 30 derniers jours)
        - year/YYYY (année spécifique comme 2023, 2024)
        - [date spécifique au format YYYY-MM-DD]
        4. Quel type d'analyse est demandé? (analysis_type) - Un ou plusieurs parmi:
        - current_price (prix actuel simple)
        - historical_price (prix historique à une date)
        - comparison (comparaison entre périodes)
        - trend (tendance sur une période)
        - volatility (volatilité)
        - high_low (plus haut/plus bas)
        - average (moyenne)
        - threshold (dépassement d'un seuil)
        5. Thresholds/valeurs mentionnés? (specific_values) - Liste des valeurs numériques mentionnées à analyser
        
        Fournis une réponse au format JSON exact:
        {{
            "is_price_query": true/false,
            "assets": ["btc", "aapl", ...],
            "timeframes": ["current", "yesterday", ...],
            "analysis_type": ["comparison", "trend", ...],
            "specific_values": [150, 30000, ...]
        }}
        """
        
        try:
            analysis_response = self.model.generate_content(analysis_prompt)
            if analysis_response and analysis_response.text:
                json_text = re.search(r'({.*})', analysis_response.text.replace('\n', ' '), re.DOTALL)
                if json_text:
                    try:
                        query_info = json.loads(json_text.group(1))
                        logger.info(f"Informations extraites de la requête: {query_info}")
                        return query_info
                    except json.JSONDecodeError:
                        logger.error("Erreur de décodage JSON dans la réponse")
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la requête: {str(e)}")
            
        # Valeurs par défaut
        return {
            "is_price_query": False,
            "assets": [],
            "timeframes": ["current"],
            "analysis_type": [],
            "specific_values": []
        }
    
    def _generate_system_prompt(self, query, financial_data=None):
        """
        Génère le prompt système pour l'assistant financier.
        
        Args:
            query (str): Requête utilisateur
            financial_data (dict, optional): Données financières récupérées
            
        Returns:
            str: Prompt système
        """
        date_str = self.reference_date.strftime('%d %B %Y')  # Format "10 mars 2025"
        
        # Historique de conversation amélioré via LangChain
        history_data = self.memory.load_memory_variables({})['chat_history']
        history_str = ""
        
        if history_data:
            history_str = "Contexte des échanges récents :\n"
            for i, msg in enumerate(history_data[-6:]):  # Plus de contexte (3 tours)
                role = "Utilisateur" if i % 2 == 0 else "Assistant"
                history_str += f"{role} : {msg.content}\n"
                
        # Données financières formatées
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
                            financial_data_str += f"Prix actuel ({data.get('date', date_str)}): {data.get('price', 'N/A')} $\n"
                            # Changement sur 24h si disponible
                            if 'change_24h' in data:
                                financial_data_str += f"Variation 24h: {data.get('change_24h', 0):+.2f}%\n"
                            elif 'change' in data:
                                financial_data_str += f"Variation: {data.get('change', 0):+.2f}%\n"
                                
                        elif timeframe == 'yesterday':
                            financial_data_str += f"Prix ({data.get('date', 'hier')}): {data.get('price', 'N/A')} $\n"
                            
                        elif timeframe == 'week':
                            financial_data_str += f"Semaine ({data.get('start_date', '')}-{data.get('end_date', '')}):\n"
                            financial_data_str += f"- Plus haut: {data.get('high', 'N/A')} $ ({data.get('high_date', '')})\n"
                            financial_data_str += f"- Plus bas: {data.get('low', 'N/A')} $ ({data.get('low_date', '')})\n"
                            financial_data_str += f"- Moyenne: {data.get('average', 'N/A'):.2f} $\n"
                            financial_data_str += f"- Variation: {data.get('change', 0):+.2f}%\n"
                            financial_data_str += f"- Volatilité: {data.get('volatility', 0):.2f}%\n"
                            
                        elif timeframe == 'month':
                            financial_data_str += f"Mois ({data.get('start_date', '')}-{data.get('end_date', '')}):\n"
                            financial_data_str += f"- Plus haut: {data.get('high', 'N/A')} $ ({data.get('high_date', '')})\n"
                            financial_data_str += f"- Plus bas: {data.get('low', 'N/A')} $ ({data.get('low_date', '')})\n"
                            financial_data_str += f"- Moyenne: {data.get('average', 'N/A'):.2f} $\n"
                            financial_data_str += f"- Variation: {data.get('change', 0):+.2f}%\n"
                            financial_data_str += f"- Volatilité: {data.get('volatility', 0):.2f}%\n"
                            
                        elif timeframe.isdigit():
                            financial_data_str += f"Année {timeframe}:\n"
                            financial_data_str += f"- Plus haut: {data.get('max_price', 'N/A')} $ ({data.get('max_date', '')})\n"
                            financial_data_str += f"- Plus bas: {data.get('min_price', 'N/A')} $ ({data.get('min_date', '')})\n"
                            financial_data_str += f"- Moyenne: {data.get('average', 'N/A'):.2f} $\n"
                            financial_data_str += f"- Médiane: {data.get('median', 'N/A'):.2f} $\n"
                            financial_data_str += f"- Variation: {data.get('change', 0):+.2f}%\n"
                            
                        elif timeframe == 'historical':
                            financial_data_str += "Données historiques disponibles:\n"
                            financial_data_str += f"- De {data.get('available_from', 'N/A')} à {data.get('available_to', 'N/A')}\n"
                            
                        else:
                            # Date spécifique
                            financial_data_str += f"Prix ({data.get('date', timeframe)}): {data.get('price', 'N/A')} $\n"
                            if 'high' in data and data['high'] is not None:
                                financial_data_str += f"- Plus haut: {data.get('high', 'N/A')} $\n"
                            if 'low' in data and data['low'] is not None:
                                financial_data_str += f"- Plus bas: {data.get('low', 'N/A')} $\n"
                
                if asset_info.get('error'):
                    financial_data_str += f"⚠️ Erreur: {asset_info['error']}\n"
            
            financial_data_str += "\n==============================\n"

        # Personnalité et instructions
        system_prompt = f"""
        Tu es Finn, un assistant financier expert. Tu aides les utilisateurs à comprendre et analyser les données financières.
        
        ## Directives pour ton comportement
        - Tu t'adaptes à la langue de l'utilisateur (français par défaut, anglais si on te parle en anglais)
        - Tu ne mentionnes JAMAIS les procédés techniques utilisés pour obtenir les données
        - Tu ne dis JAMAIS que tu utilises une API, fais des requêtes ou accèdes à des bases de données
        - Tu parles comme un expert financier qui a les informations directement en tête
        - Tu réponds toujours de manière fluide et naturelle, sans formulations robotiques
        - Tu réponds aux questions par des phrases complètes et non par des listings de données
        - Tu refuses poliment de répondre aux questions sans rapport avec la finance
        - Tu adoptes un ton confiant, expert mais accessible
        
        #        # Analyse des données
        - Réponds directement à la question sans reformuler la demande
        - Si tu as des données comparatives (aujourd'hui vs hier, par exemple), fais automatiquement la comparaison
        - Calcule les variations en pourcentage et en valeur absolue quand c'est pertinent
        - Interprète les tendances et donne du contexte quand c'est possible
        - Fais preuve d'esprit critique sur la volatilité des marchés et la fiabilité des données
        
        ## Date de référence
        - La date actuelle est le {date_str}
        - Interprète toutes les références temporelles (hier, demain, la semaine dernière) par rapport à cette date
        
        ## Contexte de la conversation
        {history_str if history_str else "Aucun échange récent à considérer"}
        
        ## Données financières disponibles
        {financial_data_str if financial_data_str else "Aucune donnée financière n'est encore disponible pour cette requête."}
        
        ## Question actuelle
        {query}
        """
        return system_prompt

    def process_query(self, query):
        """
        Traite la requête utilisateur et génère une réponse pertinente.
        
        Args:
            query (str): Requête utilisateur
        
        Returns:
            str: Réponse de l'assistant
        """
        # Étape 1: Extraire les informations de la requête
        query_info = self._extract_query_info(query)
        
        # Étape 2: Si c'est une requête financière, récupérer les données nécessaires
        financial_data = []
        if query_info.get("is_price_query", False) and query_info.get("assets"):
            for asset in query_info.get("assets", []):
                # Déterminer tous les timeframes nécessaires en fonction de l'analyse demandée
                timeframes = query_info.get("timeframes", ["current"])
                
                # Ajouter des timeframes supplémentaires en fonction du type d'analyse
                analysis_types = query_info.get("analysis_type", [])
                
                # Pour les comparaisons, s'assurer d'avoir hier si on a aujourd'hui
                if "comparison" in analysis_types and "current" in timeframes and "yesterday" not in timeframes:
                    timeframes.append("yesterday")
                
                # Pour les tendances, ajouter la semaine/mois si non spécifié
                if "trend" in analysis_types and "week" not in timeframes and "month" not in timeframes:
                    timeframes.append("week")
                
                # Pour la volatilité, s'assurer d'avoir au moins une semaine de données
                if "volatility" in analysis_types and "week" not in timeframes:
                    timeframes.append("week")
                
                # Pour high/low, s'assurer d'avoir la période appropriée
                if "high_low" in analysis_types and not any(t.isdigit() for t in timeframes):
                    # Si aucune année spécifique n'est demandée, ajouter l'année en cours
                    timeframes.append(str(self.reference_date.year))
                
                # Récupérer les données pour cet actif avec tous les timeframes nécessaires
                try:
                    asset_data = self.fetch_price_data(asset, timeframes)
                    financial_data.append(asset_data)
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des données pour {asset}: {str(e)}")
                    # Ajouter un résultat d'erreur pour informer l'utilisateur
                    financial_data.append({
                        "symbol": asset,
                        "error": f"Impossible de récupérer les données: {str(e)}"
                    })
        
        # Étape 3: Générer le prompt système avec les données
        system_prompt = self._generate_system_prompt(query, financial_data if financial_data else None)
        
        # Étape 4: Appeler Gemini pour obtenir une réponse
        for attempt in range(self.max_retries):
            try:
                response = self.chat.send_message(system_prompt)
                if response and response.text:
                    resp_text = response.text.strip()
                    # Enregistrer l'échange dans la mémoire
                    self.memory.save_context({"input": query}, {"output": resp_text})
                    return resp_text
                
                # En cas de réponse vide, message par défaut
                return "Je n'ai pas pu analyser complètement cette requête. Pourriez-vous la reformuler?"
            
            except exceptions.ResourceExhausted:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Quota Gemini atteint, nouvelle tentative dans {self.retry_delay}s")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Quota Gemini définitivement atteint")
                    return "Je rencontre des limitations techniques temporaires. Pourriez-vous réessayer plus tard?"
            
            except Exception as e:
                logger.error(f"Erreur lors de l'appel à Gemini: {str(e)}")
                # Message d'erreur personnalisé selon le contexte
                if financial_data and any(data.get("error") for data in financial_data):
                    return "Je n'ai pas pu récupérer toutes les données financières nécessaires. Certaines informations pourraient ne pas être disponibles actuellement."
                return "Je rencontre des difficultés pour traiter cette requête. Pourriez-vous la reformuler différemment?"
        
        # Si toutes les tentatives échouent
        return "Je ne peux pas répondre à cette question pour le moment. Veuillez réessayer plus tard."

if __name__ == "__main__":
    handler = GeminiHandler()
    print("Finn est prêt ! Pose tes questions (ou 'exit' pour quitter) :")
    
    while True:
        query = input("\n> ")
        if query.lower() == 'exit':
            print("À bientôt !")
            break
        response = handler.process_query(query)
        print(f"\n{response}")