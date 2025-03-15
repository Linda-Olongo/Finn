import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

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
from prophet import Prophet
import plotly.graph_objects as go

# Import des modules locaux
from data.api_fetcher import DataCollector

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

class ProphetModel:
    def __init__(self):
        self.model = None
        self.forecast = None
        self.df = None
        self.current_price = None
        self.asset_name = None

    def load_data(self, symbol: str, asset_type: str, data_collector: DataCollector):
        logger.info(f"Chargement des données pour {symbol} ({asset_type})")
        if asset_type == "crypto":
            hist_data = data_collector.get_crypto_historical(symbol)
        else:
            hist_data = data_collector.get_stock_historical(symbol)
        
        if hist_data is None or hist_data.empty:
            raise ValueError(f"Aucune donnée historique pour {symbol}")
        
        hist_data.index = pd.to_datetime(hist_data.index).tz_localize(None)
        self.df = hist_data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        self.current_price = float(self.df['y'].iloc[-1])
        self.asset_name = symbol.upper()

    def build_model(self):
        if self.df is None:
            raise ValueError("Aucune donnée chargée.")
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        self.model.fit(self.df)
        logger.info("Modèle Prophet construit")

    def forecast_specific_period(self, days: int = 30):
        if self.model is None:
            raise ValueError("Modèle non construit.")
        future = self.model.make_future_dataframe(periods=days)
        self.forecast = self.model.predict(future)
        future_forecast = self.forecast[self.forecast['ds'] >= datetime.now()]
        result = pd.DataFrame({
            'Date': future_forecast['ds'],
            'Prix prédit': future_forecast['yhat'],
            'Intervalle bas': future_forecast['yhat_lower'],
            'Intervalle haut': future_forecast['yhat_upper']
        })
        return result

    def evaluate_model(self):
        if self.model is None or self.df is None:
            return {"error": "Modèle ou données non disponibles"}
        historical_forecast = self.model.predict(self.df[['ds']])
        y_true = self.df['y']
        y_pred = historical_forecast['yhat']
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {"MAPE": mape}

class PredictionVisualizer:
    def __init__(self, data_collector: DataCollector):
        self.prophet_model = ProphetModel()
        self.data_collector = data_collector

    def create_single_asset_plot(self, symbol: str, asset_type: str, periods: int = 30):
        self.prophet_model.load_data(symbol, asset_type, self.data_collector)
        self.prophet_model.build_model()
        forecast = self.prophet_model.forecast_specific_period(periods)
        df = self.prophet_model.df

        three_months_ago = datetime.now() - timedelta(days=90)
        df = df[df['ds'] >= three_months_ago]

        fig = go.Figure()
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Historique', line=dict(color='blue'), mode='lines'))

        future_forecast = forecast[forecast['Date'] >= current_date].copy()
        if not future_forecast.empty:
            volatility = df['y'].pct_change().std() * 100 if len(df) > 5 else 0.5
            volatility = max(min(volatility, 2.0), 0.2)
            np.random.seed(42)
            future_forecast['yhat_zigzag'] = future_forecast['Prix prédit'] * (1 + np.random.uniform(-volatility/100, volatility/100, size=len(future_forecast)))
            future_forecast['yhat_smooth'] = future_forecast['yhat_zigzag'].ewm(span=3).mean()
            fig.add_trace(go.Scatter(x=future_forecast['Date'], y=future_forecast['yhat_smooth'], name='Prédiction', line=dict(color='grey', dash='dash'), mode='lines'))

        chart_data = {
            "labels": [d.strftime('%Y-%m-%d') for d in df['ds'].tolist()] + [d.strftime('%Y-%m-%d') for d in future_forecast['Date'].tolist()],
            "datasets": [
                {"label": "Historique", "data": df['y'].tolist(), "borderColor": "blue", "fill": False},
                {"label": "Prédiction", "data": [None] * len(df) + future_forecast['yhat_smooth'].tolist(), "borderColor": "grey", "borderDash": [5, 5], "fill": False}
            ]
        }

        fig.update_layout(title=f"Prédiction pour {symbol}", xaxis_title="Date", yaxis_title="Prix ($)")
        fig.show()
        logger.info(f"Visualisation créée pour {symbol}")
        return chart_data

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
        self.collector = DataCollector()
        self.chat_history = []
        self.prophet_model = ProphetModel()
        self.visualizer = PredictionVisualizer(self.collector)

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
            
            try:
                stock_info = self._throttled_api_call(self.collector.mapper.get_stock_info, symbol)
                if stock_info:
                    asset_type = "stock"
                    resolved_symbol, name = stock_info
                    asset_found = True
            except Exception as e:
                logger.debug(f"Échec action: {str(e)}")
            
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
        - Actifs (ex: "BTC", "AAPL")
        - Type (stock/crypto, défaut "crypto")
        - Période en jours pour prédictions (défaut 30 si non spécifié)
        - Timeframes pour prix (current, week, month, etc.)
        - Langue ("fr" ou "en")
        
        Réponds au format JSON:
        {{
            "intent": "price/future_price/visualization/analysis/comparison/other",
            "assets": ["BTC"],
            "asset_type": "crypto",
            "timeframe": 30,
            "price_timeframes": ["current"],
            "language": "fr"
        }}
        """
        response = self.client.messages.create(model=self.model, max_tokens=200, messages=[{"role": "user", "content": analysis_prompt}])
        json_text = response.content[0].text.strip()
        logger.info(f"Résultat brut de l'extraction: {json_text}")
        try:
            info = json.loads(json_text)
            if not info.get("intent") or info["intent"] == "other":
                if info.get("assets"):
                    info["intent"] = "price"
                    info["price_timeframes"] = info.get("price_timeframes", ["current"])
            return info
        except:
            logger.error(f"Échec du parsing JSON: {json_text}")
            return {"intent": "other", "assets": [], "asset_type": "crypto", "timeframe": 30, "price_timeframes": ["current"], "language": "fr"}

    def make_prediction(self, asset, asset_type, timeframe):
        try:
            self.prophet_model.load_data(asset, asset_type, self.collector)
            self.prophet_model.build_model()
            forecast = self.prophet_model.forecast_specific_period(timeframe)
            last_price = forecast['Prix prédit'].iloc[-1]
            last_date = forecast['Date'].iloc[-1].strftime('%Y-%m-%d')
            change_pct = ((last_price / self.prophet_model.current_price) - 1) * 100
            return {
                "asset": asset,
                "asset_name": self.prophet_model.asset_name,
                "current_price": self.prophet_model.current_price,
                "prediction": {"price": last_price, "date": last_date, "change_percent": change_pct}
            }
        except Exception as e:
            logger.error(f"Erreur dans make_prediction: {e}")
            return {"error": str(e)}

    def create_visualization(self, asset, asset_type, timeframe):
        try:
            chart_data = self.visualizer.create_single_asset_plot(asset, asset_type, timeframe)
            return {"success": True, "message": "Visualisation créée", "chart_data": chart_data}
        except Exception as e:
            logger.error(f"Erreur dans create_visualization: {e}")
            return {"error": str(e)}

    def analyze_asset(self, asset, asset_type):
        try:
            self.prophet_model.load_data(asset, asset_type, self.collector)
            self.prophet_model.build_model()
            metrics = self.prophet_model.evaluate_model()
            forecast = self.prophet_model.forecast_specific_period(30)
            trend = "hausse" if forecast['Prix prédit'].iloc[-1] > self.prophet_model.current_price else "baisse"
            return {"asset": asset, "metrics": metrics, "trend": trend}
        except Exception as e:
            logger.error(f"Erreur dans analyze_asset: {e}")
            return {"error": str(e)}

    def _generate_system_prompt(self, query, financial_data=None, prediction_data=None, viz_data=None, analysis_data=None):
        date_str = self.reference_date.strftime('%d %B %Y')
        history_data = self.memory.load_memory_variables({})['chat_history']
        history_str = "Contexte des échanges récents :\n" + "\n".join([f"{'Utilisateur' if i % 2 == 0 else 'Assistant'}: {msg.content}" for i, msg in enumerate(history_data[-2:])]) if history_data else "Aucun échange récent."

        financial_data_str = ""
        if financial_data:
            financial_data_str = "\n===== DONNÉES FINANCIÈRES =====\n"
            for asset_info in financial_data:
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
                    prediction_data_str += f"{pred['asset_name']}: Prix prédit {pred['prediction']['price']:.2f} $ le {pred['prediction']['date']} (var: {pred['prediction']['change_percent']:+.2f}%)\n"

        viz_data_str = ""
        if viz_data and viz_data.get("success"):
            viz_data_str = "\n===== VISUALISATION =====\nGraphique généré pour l'actif demandé.\n"

        analysis_data_str = ""
        if analysis_data:
            analysis_data_str = "\n===== ANALYSE =====\n"
            for analysis in analysis_data:
                if "error" not in analysis:
                    analysis_data_str += f"{analysis['asset']}: Tendance {analysis['trend']}, MAPE {analysis['metrics']['MAPE']:.2f}%\n"

        system_prompt = f"""
        Tu es Finn, un assistant financier conversationnel et concis.
        - Réponds directement et en 1-2 phrases pour les prix ou prédictions.
        - Sois naturel et chaleureux, utilise des expressions comme "tu vois" ou "en fait".
        - Ne mentionne jamais les processus techniques.
        - Date actuelle: {date_str}
        
        {history_str}
        {financial_data_str}
        {prediction_data_str}
        {viz_data_str}
        {analysis_data_str}
        
        Question: {query}
        """
        return system_prompt

    def process_query(self, query):
        query_info = self._extract_query_info(query)
        intent = query_info.get("intent")
        logger.info(f"Intention détectée: {query_info}")

        financial_data = []
        prediction_data = []
        viz_data = None
        analysis_data = []

        if intent == "price" and query_info.get("assets"):
            for asset in query_info["assets"]:
                financial_data.append(self.fetch_price_data(asset, query_info.get("price_timeframes")))

        elif intent == "future_price" and query_info.get("assets"):
            for asset in query_info["assets"]:
                prediction_data.append(self.make_prediction(asset, query_info.get("asset_type", "crypto"), query_info.get("timeframe", 30)))

        elif intent == "visualization" and query_info.get("assets"):
            asset = query_info["assets"][0]
            viz_data = self.create_visualization(asset, query_info.get("asset_type", "crypto"), query_info.get("timeframe", 30))

        elif intent == "analysis" and query_info.get("assets"):
            for asset in query_info["assets"]:
                analysis_data.append(self.analyze_asset(asset, query_info.get("asset_type", "crypto")))

        elif intent == "comparison" and len(query_info.get("assets", [])) > 1:
            for asset in query_info["assets"]:
                financial_data.append(self.fetch_price_data(asset, query_info.get("price_timeframes")))
            # Ajouter éventuellement des prédictions pour comparaison future

        system_prompt = self._generate_system_prompt(query, financial_data, prediction_data, viz_data, analysis_data)
        response = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, messages=[{"role": "user", "content": system_prompt}])
        resp_text = response.content[0].text.strip()
        
        self.memory.save_context({"input": query}, {"output": resp_text})
        self.chat_history.extend([query, resp_text])
        return resp_text

if __name__ == "__main__":
    handler = ClaudeHandler()
    print("Finn est prêt ! Posez vos questions (ou 'exit' pour quitter) :")
    while True:
        query = input("\n> ")
        if query.lower() == 'exit':
            print("À bientôt !")
            break
        response = handler.process_query(query)
        print(f"\nA: {response}")