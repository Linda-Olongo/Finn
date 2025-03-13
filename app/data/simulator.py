import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

# Importez vos modules de la même façon que dans votre fichier original
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

# Importez le modèle Prophet - ajustez le chemin si nécessaire
from models.fbprophet import ProphetModel

# Chargement de la clé API Gemini depuis l'environnement
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Configuration du logging
logging.basicConfig(level=logging.WARNING)  
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "faible"
    MEDIUM = "modéré"
    HIGH = "élevé"

class RecommendationType(Enum):
    BUY = "ACHETER"
    SELL = "VENDRE"
    HOLD = "CONSERVER"
    WAIT = "ATTENDRE"

class TradingSimulator:
    """Simulateur de trading utilisant Prophet pour les prédictions."""
    
    def __init__(self):
        self.prophet_model = ProphetModel()
        self.current_data = None
        self.forecast = None
        self.prophet_data = None

    def new_investment(self, symbol, amount, horizon, risk_level, asset_type='crypto', use_ai=True):
        """Simule un nouvel investissement."""
        try:
            # Valider les entrées
            if amount <= 0:
                raise ValueError("Le montant à investir doit être positif")
                
            if horizon <= 0:
                raise ValueError("L'horizon doit être positif")
                
            # Charger les données
            self._load_asset_data(symbol, asset_type)
            
            # Générer les prédictions
            self._build_forecast_model(horizon)
            
            # Ajuster l'intervalle de confiance
            self._adjust_confidence_interval(risk_level)
            
            # Calculer les métriques
            current_price = self.current_data['price']
            quantity = amount / current_price
            
            # Arrondir la quantité selon le type d'actif
            if asset_type == 'crypto':
                quantity = round(quantity, 5)
            else:
                quantity = round(quantity, 2)
                
            # Calculer la valeur future estimée
            future_price = self.forecast['Prix prédit'].iloc[-1]
            future_value = quantity * future_price
            
            # Calculer le rendement attendu
            expected_return = ((future_value / amount) - 1) * 100
            
            # Déterminer le meilleur moment d'entrée
            days_to_wait, entry_date = self._find_optimal_entry_point()
            
            # Calculer la fiabilité de la prédiction
            reliability = self._calculate_prediction_reliability()
            
            # Calculer la force de la tendance
            trend_strength, is_uptrend = self._calculate_trend_strength()
            
            # Générer une recommandation
            forecast_data = {
                'horizon': horizon,
                'expected_return': expected_return,
                'final_price': future_price,
                'reliability': reliability,
                'days_to_wait': days_to_wait,
                'trend_strength': trend_strength,
                'is_uptrend': is_uptrend
            }
            
            if use_ai and ANTHROPIC_API_KEY:
                recommendation = self._get_ai_recommendation(
                    self.current_data, forecast_data, mode='new_investment')
            else:
                recommendation = self._generate_investment_recommendation(
                    expected_return, reliability, days_to_wait)
            
            # Préparer les résultats
            results = {
                'asset': {
                    'symbol': symbol,
                    'name': self.current_data.get('name', symbol),
                    'current_price': current_price,
                    'change_24h': self.current_data.get('change_24h', self.current_data.get('change', 0)),
                    'volume': self.current_data.get('volume', 0),
                    'market_cap': self.current_data.get('market_cap', 0)
                },
                'investment': {
                    'amount': amount,
                    'quantity': quantity,
                    'future_value': round(future_value, 2),
                    'expected_return': round(expected_return, 1),
                    'reliability': reliability,
                    'optimal_entry': {
                        'days_to_wait': days_to_wait,
                        'date': entry_date
                    }
                },
                'forecast': {
                    'horizon': horizon,
                    'prices': self.forecast.to_dict('records'),
                    'trend_strength': round(trend_strength, 1),
                    'is_uptrend': is_uptrend
                },
                'recommendation': recommendation
            }
            
            return results
            
        except Exception as e:
            raise ValueError(f"Erreur lors de la simulation: {e}")
    
    def analyze_portfolio(self, symbol, quantity, avg_purchase_price, horizon, asset_type='crypto', use_ai=True):
        """Analyse une position existante."""
        try:
            # Valider les entrées
            if quantity <= 0:
                raise ValueError("La quantité doit être positive")
                
            if avg_purchase_price <= 0:
                raise ValueError("Le prix d'achat moyen doit être positif")
                
            if horizon <= 0:
                raise ValueError("L'horizon doit être positif")
                
            # Charger les données
            self._load_asset_data(symbol, asset_type)
            
            # Générer les prédictions
            self._build_forecast_model(horizon)
            
            # Calculer la valeur de la position
            current_price = self.current_data['price']
            initial_value = quantity * avg_purchase_price
            current_value = quantity * current_price
            
            # Calculer les performances
            current_return = ((current_value / initial_value) - 1) * 100
            
            # Calculer la valeur future estimée
            future_price = self.forecast['Prix prédit'].iloc[-1]
            future_value = quantity * future_price
            
            # Calculer les rendements projetés
            future_return = ((future_price / avg_purchase_price) - 1) * 100
            future_return_from_now = ((future_price / current_price) - 1) * 100
            
            # Calculer la fiabilité de la prédiction
            reliability = self._calculate_prediction_reliability()
            
            # Calculer la force de la tendance
            trend_strength, is_uptrend = self._calculate_trend_strength()
            
            # Générer une recommandation
            forecast_data = {
                'horizon': horizon,
                'final_price': future_price,
                'reliability': reliability,
                'trend_strength': trend_strength,
                'is_uptrend': is_uptrend
            }
            
            position_data = {
                'quantity': quantity,
                'avg_purchase_price': avg_purchase_price,
                'current_return': current_return,
                'future_return': future_return,
                'future_return_from_now': future_return_from_now
            }
            
            if use_ai and ANTHROPIC_API_KEY:
                recommendation = self._get_ai_recommendation(
                    self.current_data, forecast_data, position_data, mode='portfolio')
            else:
                recommendation = self._generate_portfolio_recommendation(
                    current_return, future_return_from_now, reliability, trend_strength, is_uptrend)
            
            # Préparer les résultats
            results = {
                'asset': {
                    'symbol': symbol,
                    'name': self.current_data.get('name', symbol),
                    'current_price': current_price,
                    'change_24h': self.current_data.get('change_24h', self.current_data.get('change', 0)),
                    'volume': self.current_data.get('volume', 0),
                    'market_cap': self.current_data.get('market_cap', 0)
                },
                'position': {
                    'quantity': quantity,
                    'avg_purchase_price': avg_purchase_price,
                    'initial_value': round(initial_value, 2),
                    'current_value': round(current_value, 2),
                    'current_return': round(current_return, 1),
                    'future_value': round(future_value, 2),
                    'future_return': round(future_return, 1),
                    'future_return_from_now': round(future_return_from_now, 1)
                },
                'forecast': {
                    'horizon': horizon,
                    'prices': self.forecast.to_dict('records'),
                    'trend_strength': round(trend_strength, 1),
                    'is_uptrend': is_uptrend,
                    'reliability': reliability
                },
                'recommendation': recommendation
            }
            
            return results
            
        except Exception as e:
            raise ValueError(f"Erreur lors de l'analyse: {e}")
        
    def search_assets(self, query):
        """Recherche des actifs correspondant à la requête."""
        try:
            results = []
            
            # Rechercher dans les cryptos
            crypto_result = self.preprocessor.collector.mapper.get_crypto_info(query)
            if crypto_result:
                crypto_id, crypto_name = crypto_result
                results.append({
                    "name": crypto_name,
                    "symbol": crypto_id,
                    "type": "crypto"
                })
            
            # Rechercher dans les actions
            stock_result = self.preprocessor.collector.mapper.get_stock_info(query)
            if stock_result:
                stock_symbol, stock_name = stock_result
                results.append({
                    "name": stock_name,
                    "symbol": stock_symbol,
                    "type": "stock"
                })
                
            # Si aucun résultat exact, utiliser l'API de recherche de CoinGecko
            if not results and len(query) >= 2:
                try:
                    search_url = f"https://api.coingecko.com/api/v3/search?query={query}"
                    response = requests.get(search_url)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Ajouter les 5 premières cryptos trouvées
                        for coin in data.get('coins', [])[:5]:
                            results.append({
                                "name": coin['name'],
                                "symbol": coin['id'],
                                "type": "crypto"
                            })
                except Exception as e:
                    logger.error(f"Erreur lors de la recherche CoinGecko: {e}")
            
            # Recherche similaire pour les actions via Yahoo Finance si pas de résultats
            if not results and len(query) >= 2:
                try:
                    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Ajouter les 5 premières actions trouvées
                        for quote in data.get('quotes', [])[:5]:
                            if quote.get('quoteType') == 'EQUITY':
                                results.append({
                                    "name": quote.get('longname', quote['symbol']),
                                    "symbol": quote['symbol'],
                                    "type": "stock"
                                })
                except Exception as e:
                    logger.error(f"Erreur lors de la recherche Yahoo Finance: {e}")
                    
            return results
        except Exception as e:
            logger.error(f"Erreur de recherche d'actifs: {e}")
            return []
    
    def _load_asset_data(self, symbol, asset_type='crypto'):
        """Charge les données d'un actif."""
        try:
            # Charger les données via ProphetModel
            prophet_data = self.prophet_model.load_data(symbol, asset_type)
            
            # Conserver les données actuelles
            if asset_type == 'crypto':
                self.current_data = self.prophet_model.preprocessor.collector.get_crypto_current(symbol)
            else:
                self.current_data = self.prophet_model.preprocessor.collector.get_stock_current(symbol)
                
            return prophet_data
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement des données pour {symbol}: {e}")
    
    def _build_forecast_model(self, horizon):
        """Construit et entraîne le modèle de prédiction."""
        try:
            # Construire le modèle
            self.prophet_model.build_model()
            
            # Générer les prédictions
            self.forecast = self.prophet_model.forecast_specific_period(days=horizon)
            
            return self.forecast
        except Exception as e:
            raise ValueError(f"Erreur lors de la génération des prédictions: {e}")
    
    def _adjust_confidence_interval(self, risk_level):
        """Ajuste l'intervalle de confiance des prédictions selon le niveau de risque."""
        if self.forecast is None or self.forecast.empty:
            return
            
        # Obtenir l'écart-type des prédictions
        price_std = self.forecast['Prix prédit'].std()
        
        # Ajuster l'intervalle selon le niveau de risque
        risk_multipliers = {
            RiskLevel.LOW.value: 1.5,    # Plus conservateur
            RiskLevel.MEDIUM.value: 1.0,  # Normal
            RiskLevel.HIGH.value: 0.6     # Plus risqué
        }
        
        multiplier = risk_multipliers.get(risk_level.lower(), 1.0)
        
        # Recalculer les intervalles
        mean_prices = self.forecast['Prix prédit']
        self.forecast['Intervalle bas'] = mean_prices - (price_std * multiplier)
        self.forecast['Intervalle haut'] = mean_prices + (price_std * multiplier)
        
        # S'assurer que les valeurs ne sont pas négatives
        self.forecast['Intervalle bas'] = self.forecast['Intervalle bas'].clip(lower=0)
    
    def _calculate_prediction_reliability(self):
        """Calcule un score de fiabilité pour les prédictions."""
        if self.prophet_model.metrics is None:
            # Évaluer le modèle si ce n'est pas déjà fait
            metrics = self.prophet_model.evaluate_model()
        else:
            metrics = self.prophet_model.metrics
            
        # Calculer le score de fiabilité basé sur R2 et MAPE
        r2_score = metrics.get('R2', 0)
        mape = metrics.get('MAPE', 100)
        
        # Normaliser R2 entre 0 et 1
        r2_normalized = max(0, min(1, (r2_score + 1) / 2))
        
        # Normaliser MAPE inversé entre 0 et 1
        mape_normalized = max(0, min(1, 1 - (mape / 100)))
        
        # Combiner les scores (60% R2, 40% MAPE)
        reliability = (r2_normalized * 0.6) + (mape_normalized * 0.4)
        
        # Convertir en pourcentage et arrondir
        return round(reliability * 100)
    
    def _calculate_trend_strength(self):
        """Calcule la force de la tendance et sa direction."""
        if self.forecast is None or self.forecast.empty or len(self.forecast) < 2:
            return 0, True
            
        # Prix actuel
        current_price = self.forecast['Prix prédit'].iloc[0]
        
        # Prix à la fin de l'horizon
        final_price = self.forecast['Prix prédit'].iloc[-1]
        
        # Déterminer la direction de la tendance
        is_uptrend = final_price >= current_price
        
        # Calculer le changement en pourcentage
        percent_change = abs((final_price - current_price) / current_price) * 100
        
        # Normaliser la force de la tendance (max 100%)
        trend_strength = min(100, percent_change)
        
        return trend_strength, is_uptrend
    
    def _find_optimal_entry_point(self):
        """Identifie le meilleur moment pour entrer sur le marché."""
        if self.forecast is None or self.forecast.empty:
            return 0, "Maintenant"
            
        # Copie des prédictions avec tri par prix
        sorted_forecast = self.forecast.sort_values('Prix prédit')
        
        # Si la tendance est constamment à la hausse, entrer maintenant
        current_price = self.forecast['Prix prédit'].iloc[0]
        if sorted_forecast['Prix prédit'].iloc[0] == current_price:
            return 0, "Maintenant"
            
        # Trouver le point d'entrée optimal (prix le plus bas dans les 7 premiers jours)
        short_term = self.forecast.head(min(7, len(self.forecast)))
        lowest_price_idx = short_term['Prix prédit'].idxmin()
        
        # Obtenir le nombre de jours à attendre
        entry_date = self.forecast.loc[lowest_price_idx, 'Date']
        today = pd.Timestamp.now().normalize()
        days_to_wait = (entry_date - today).days
        
        if days_to_wait <= 0:
            return 0, "Maintenant"
            
        return days_to_wait, entry_date.strftime('%Y-%m-%d')
    
    def _generate_investment_recommendation(self, expected_return, reliability, days_to_wait):
        """Génère une recommandation pour un nouvel investissement."""
        # Initialiser les variables
        recommendation_type = None
        confidence = min(reliability, 100)
        explanation = ""
        
        # Déterminer la recommandation
        if expected_return <= -5:
            # Rendement attendu négatif significatif
            recommendation_type = RecommendationType.WAIT
            explanation = "Tendance baissière prévue. Attendre un meilleur point d'entrée."
            
        elif days_to_wait > 0 and days_to_wait <= 7:
            # Un meilleur point d'entrée à court terme
            recommendation_type = RecommendationType.WAIT
            explanation = f"Un meilleur prix est attendu dans {days_to_wait} jours."
            
        elif expected_return >= 10 and reliability >= 60:
            # Bon rendement attendu avec une fiabilité acceptable
            recommendation_type = RecommendationType.BUY
            explanation = f"La tendance est favorable avec une probabilité de hausse de {confidence}% sur l'horizon prévu."
            
        elif expected_return > 0:
            # Rendement positif mais modéré
            recommendation_type = RecommendationType.BUY
            explanation = "Tendance légèrement positive. Investissement potentiellement rentable."
            
        else:
            # Cas par défaut
            recommendation_type = RecommendationType.WAIT
            explanation = "Les conditions actuelles ne favorisent pas un investissement immédiat."
            
        return {
            'type': recommendation_type.value,
            'confidence': confidence,
            'explanation': explanation,
            'source': 'interne'
        }
    
    def _generate_portfolio_recommendation(self, current_return, future_return, reliability, trend_strength, is_uptrend):
        """Génère une recommandation pour une position existante."""
        # Initialiser les variables
        recommendation_type = None
        confidence = min(reliability, 100)
        explanation = ""
        
        # Déterminer la recommandation
        if not is_uptrend and future_return < -10 and reliability >= 60:
            # Tendance baissière significative
            recommendation_type = RecommendationType.SELL
            explanation = f"Une baisse significative est anticipée avec une confiance de {confidence}%."
            
        elif not is_uptrend and future_return < 0:
            # Tendance baissière légère
            if current_return > 20:
                # Position en gain significatif
                recommendation_type = RecommendationType.SELL
                explanation = "Envisagez de prendre vos bénéfices avant la baisse anticipée."
            else:
                # Position sans gain significatif
                recommendation_type = RecommendationType.HOLD
                explanation = "La tendance baissière pourrait être temporaire. Surveillez de près."
                
        elif is_uptrend and future_return > 15 and reliability >= 60:
            # Forte tendance haussière
            recommendation_type = RecommendationType.HOLD
            explanation = f"La tendance reste haussière à moyen terme avec une confiance de {confidence}%."
            
        elif is_uptrend and future_return > 0:
            # Tendance haussière modérée
            recommendation_type = RecommendationType.HOLD
            explanation = "La tendance reste haussière à moyen terme mais ralentit. Conservation recommandée."
            
        else:
            # Cas par défaut
            recommendation_type = RecommendationType.HOLD
            explanation = "Les conditions actuelles suggèrent de conserver votre position."
            
        return {
            'type': recommendation_type.value,
            'confidence': confidence,
            'explanation': explanation,
            'source': 'interne'
        }
    
    def _get_ai_recommendation(self, asset_data: Dict, forecast_data: Dict, position_data: Optional[Dict] = None, mode: str = 'new_investment') -> Dict:
        """
        Obtient une recommandation d'investissement de Claude API.
        
        Args:
            asset_data: Données de l'actif.
            forecast_data: Données de prévision.
            position_data: Données de la position actuelle (pour le mode suivi de portefeuille).
            mode: Mode de recommandation ('new_investment' ou 'portfolio').
            
        Returns:
            Dict: Recommandation et justification générées par Claude.
        """
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY non configurée. Utilisation de la logique interne.")
            if mode == 'new_investment':
                return self._generate_investment_recommendation(
                    forecast_data.get('expected_return', 0),
                    forecast_data.get('reliability', 0),
                    forecast_data.get('days_to_wait', 0)
                )
            else:
                return self._generate_portfolio_recommendation(
                    position_data.get('current_return', 0),
                    position_data.get('future_return_from_now', 0),
                    forecast_data.get('reliability', 0),
                    forecast_data.get('trend_strength', 0),
                    forecast_data.get('is_uptrend', True)
                )

        try:
            # Construction du prompt pour Claude
            if mode == 'new_investment':
                prompt = f"""
                Agis comme un conseiller financier expert en analyse technique et fondamentale. 
                Analyse les données suivantes pour un nouvel investissement potentiel et formule une recommandation concise :

                Actif : {asset_data.get('name', 'Inconnu')} ({asset_data.get('symbol', 'N/A')})
                Prix actuel : {asset_data.get('current_price', 'N/A')}$
                Variation sur 24h : {asset_data.get('change_24h', 'N/A')}%

                Projection sur {forecast_data.get('horizon', 0)} jours :
                - Prix final estimé : {forecast_data.get('final_price', 0)}$
                - Rendement attendu : {forecast_data.get('expected_return', 0)}%
                - Fiabilité de la prédiction : {forecast_data.get('reliability', 0)}%

                Point d'entrée optimal : dans {forecast_data.get('days_to_wait', 0)} jours

                Donne uniquement une recommandation parmi les suivantes : ACHETER, VENDRE, ATTENDRE, CONSERVER
                Puis fournis une explication brève (max 200 caractères) justifiant ta recommandation.
                Format de réponse : "RECOMMANDATION: [ta recommandation]\nRAISON: [ton explication]"
                """
            else:
                prompt = f"""
                Agis comme un conseiller financier expert en analyse technique et fondamentale.
                Analyse les données suivantes pour une position existante et formule une recommandation concise :

                Actif : {asset_data.get('name', 'Inconnu')} ({asset_data.get('symbol', 'N/A')})
                Prix actuel : {asset_data.get('current_price', 'N/A')}$
                Variation sur 24h : {asset_data.get('change_24h', 'N/A')}%

                Position actuelle :
                - Quantité détenue : {position_data.get('quantity', 0)}
                - Prix d'achat moyen : {position_data.get('avg_purchase_price', 0)}$
                - Performance actuelle : {position_data.get('current_return', 0)}%

                Projection sur {forecast_data.get('horizon', 0)} jours :
                - Prix final estimé : {forecast_data.get('final_price', 0)}$
                - Performance projetée : {position_data.get('future_return', 0)}%
                - Fiabilité de la prédiction : {forecast_data.get('reliability', 0)}%

                Donne uniquement une recommandation parmi les suivantes : ACHETER, VENDRE, ATTENDRE, CONSERVER
                Puis fournis une explication brève (max 200 caractères) justifiant ta recommandation.
                Format de réponse : "RECOMMANDATION: [ta recommandation]\nRAISON: [ton explication]"
                """
            # Appel à l'API Claude
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.3
            }

            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()

            response_data = response.json()
            text_response = response_data["content"][0]["text"]
            logger.info(f"Réponse brute de Claude: {text_response}")

            # Normalisation des retours de ligne et parsing robuste
            lines = text_response.strip().splitlines()  # Utilise splitlines() pour gérer tous les types de retours (\n, \r\n)
            logger.debug(f"Lignes parsées: {lines}")

            reco_type = "CONSERVER"  # Valeur par défaut
            explanation = "Aucune explication fournie par Claude."

            if lines:
                for line in lines:
                    if "RECOMMANDATION:" in line:
                        reco_type = line.replace('RECOMMANDATION:', '').strip()
                    elif "RAISON:" in line:
                        explanation = line.replace('RAISON:', '').strip()

                # Si aucune raison n'a été trouvée dans les lignes
                if explanation == "Aucune explication fournie par Claude.":
                    logger.warning("Aucune raison trouvée dans la réponse de Claude.")

            # Vérifier que la recommandation est valide
            valid_recos = [r.value for r in RecommendationType]
            if reco_type not in valid_recos:
                logger.warning(f"Recommandation invalide '{reco_type}', utilisant CONSERVER.")
                reco_type = RecommendationType.HOLD.value

            return {
                'type': reco_type,
                'confidence': forecast_data.get('reliability', 70),
                'explanation': explanation,  # Assurez-vous que l'explication est bien incluse ici
                'source': 'claude'
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de l'appel à l'API Claude: {e}")
            print(f"Erreur API Claude: {e}. Utilisation de la logique interne comme fallback.")
            if mode == 'new_investment':
                return self._generate_investment_recommendation(
                    forecast_data.get('expected_return', 0),
                    forecast_data.get('reliability', 0),
                    forecast_data.get('days_to_wait', 0)
                )
            else:
                return self._generate_portfolio_recommendation(
                    position_data.get('current_return', 0),
                    position_data.get('future_return_from_now', 0),
                    forecast_data.get('reliability', 0),
                    forecast_data.get('trend_strength', 0),
                    forecast_data.get('is_uptrend', True)
                )
        except Exception as e:
            logger.error(f"Erreur inattendue dans _get_ai_recommendation: {e}")
            print(f"Erreur inattendue: {e}. Utilisation de la logique interne comme fallback.")
            if mode == 'new_investment':
                return self._generate_investment_recommendation(
                    forecast_data.get('expected_return', 0),
                    forecast_data.get('reliability', 0),
                    forecast_data.get('days_to_wait', 0)
                )
            else:
                return self._generate_portfolio_recommendation(
                    position_data.get('current_return', 0),
                    position_data.get('future_return_from_now', 0),
                    forecast_data.get('reliability', 0),
                    forecast_data.get('trend_strength', 0),
                    forecast_data.get('is_uptrend', True)
                )
    
    def plot_forecast(self, include_history=True, save_path=None):
        """Affiche un graphique interactif des prévisions."""
        if self.forecast is None or self.forecast.empty:
            print("Aucune prévision disponible. Veuillez d'abord générer des prédictions.")
            return
            
        try:
            # Créer une figure interactive avec Plotly
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            # Obtenir la date actuelle
            current_date = pd.Timestamp.now().normalize()
            current_date_str = current_date.strftime('%Y-%m-%d')
            
            # Ajouter l'historique des prix si demandé
            if include_history and hasattr(self.prophet_model, 'prophet_data') and self.prophet_model.prophet_data:
                df = self.prophet_model.prophet_data['df']
                
                # Ne garder que les données des 90 derniers jours pour la lisibilité
                recent_cutoff = current_date - pd.Timedelta(days=90)
                df_recent = df[df['ds'] >= recent_cutoff]
                
                if not df_recent.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=df_recent['ds'],
                            y=df_recent['y'],
                            mode='lines',
                            name="Historique",
                            line=dict(color='royalblue', width=2)
                        )
                    )
            
            # Ajouter le prix actuel comme point de référence
            if hasattr(self.prophet_model, 'prophet_data') and self.prophet_model.prophet_data:
                current_price = self.prophet_model.prophet_data['current_price']
                
                fig.add_trace(
                    go.Scatter(
                        x=[current_date],
                        y=[current_price],
                        mode='markers',
                        name="Prix actuel",
                        marker=dict(color='red', size=10, symbol='circle')
                    )
                )
            
            # Filtrer les prévisions futures
            future_forecast = self.forecast[self.forecast['Date'] >= current_date]
            
            # Ajouter les prévisions
            fig.add_trace(
                go.Scatter(
                    x=future_forecast['Date'],
                    y=future_forecast['Prix prédit'],
                    mode='lines',
                    name="Prévision",
                    line=dict(color='green', width=2)
                )
            )
            
            # Ajouter la zone d'intervalle de confiance
            fig.add_trace(
                go.Scatter(
                    x=future_forecast['Date'].tolist() + future_forecast['Date'].tolist()[::-1],
                    y=future_forecast['Intervalle haut'].tolist() + future_forecast['Intervalle bas'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(0,100,80,0)'),
                    name="Intervalle de confiance"
                )
            )
            
            # Améliorer la mise en page
            asset_name = self.prophet_model.asset_name if hasattr(self.prophet_model, 'asset_name') else "Actif"
            
            fig.update_layout(
                title=f"Prévision de prix pour {asset_name}",
                xaxis_title="Date",
                yaxis_title="Prix (USD)",
                hovermode="x unified",
                legend_title="Légende",
                template="plotly_dark"
            )
            
            # Au lieu d'utiliser add_vline qui cause l'erreur, ajouter une trace shape pour la ligne verticale
            fig.add_shape(
                type="line",
                x0=current_date_str,
                y0=0,
                x1=current_date_str,
                y1=1,
                yref="paper",
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                ),
            )
            
            # Ajouter l'annotation pour "Aujourd'hui" manuellement
            fig.add_annotation(
                x=current_date_str,
                y=1,
                yref="paper",
                text="Aujourd'hui",
                showarrow=False,
                font=dict(color="red"),
                xanchor="right",
                yanchor="bottom"
            )
            
            # Sauvegarder le graphique si un chemin est spécifié
            if save_path:
                fig.write_image(save_path)
                print(f"Graphique sauvegardé sous {save_path}")
            
            # Afficher le graphique
            fig.show()
        except Exception as e:
            print(f"Erreur lors de l'affichage du graphique: {e}")
            import traceback
            traceback.print_exc()
    
    def display_recommendation_card(self, results, mode='new_investment'):
        """Affiche une carte de recommandation dans le terminal."""
        asset = results['asset']
        recommendation = results['recommendation']
        
        # Définir les couleurs ANSI pour le terminal
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        BOLD = '\033[1m'
        END = '\033[0m'
        
        # Définir les couleurs selon le type de recommandation
        reco_colors = {
            'ACHETER': GREEN,
            'VENDRE': RED,
            'ATTENDRE': YELLOW,
            'CONSERVER': BLUE
        }
        
        reco_color = reco_colors.get(recommendation['type'], BOLD)
        
        # Créer une box ASCII pour la recommandation
        print("\n" + "=" * 80)
        print(f"{BOLD}Trading Simulator - Recommandation{END}")
        print("=" * 80)
        
        # Afficher les informations sur l'actif
        print(f"\n{BOLD}{asset['name']} ({asset['symbol']}){END}")
        
        if asset['change_24h'] >= 0:
            change_str = f"{GREEN}+{asset['change_24h']}%{END}"
        else:
            change_str = f"{RED}{asset['change_24h']}%{END}"
            
        print(f"Prix actuel: {BOLD}${asset['current_price']:.2f}{END} ({change_str})")
        
        # Afficher les détails spécifiques au mode
        if mode == 'new_investment':
            investment = results['investment']
            forecast = results['forecast']
            
            print(f"\n{BOLD}Résultats de simulation:{END}")
            print(f"Montant investi: ${investment['amount']:.2f}")
            print(f"Quantité acquise: {investment['quantity']}")
            print(f"Valeur finale estimée: ${investment['future_value']:.2f}")
            
            if investment['expected_return'] >= 0:
                return_str = f"{GREEN}+{investment['expected_return']:.1f}%{END}"
            else:
                return_str = f"{RED}{investment['expected_return']:.1f}%{END}"
                
            print(f"Rendement attendu: {return_str}")
            print(f"Fiabilité prédiction: {investment['reliability']}%")
            
            if investment['optimal_entry']['days_to_wait'] > 0:
                print(f"Date optimale d'achat: {YELLOW}{investment['optimal_entry']['date']}{END}")
            else:
                print(f"Date optimale d'achat: {GREEN}Maintenant{END}")
                
        else:  # mode portfolio
            position = results['position']
            forecast = results['forecast']
            
            print(f"\n{BOLD}Position actuelle:{END}")
            print(f"Quantité détenue: {position['quantity']}")
            print(f"Prix d'achat moyen: ${position['avg_purchase_price']:.2f}")
            print(f"Valeur investie: ${position['initial_value']:.2f}")
            print(f"Valeur actuelle: ${position['current_value']:.2f}")
            
            if position['current_return'] >= 0:
                current_return_str = f"{GREEN}+{position['current_return']:.1f}%{END}"
            else:
                current_return_str = f"{RED}{position['current_return']:.1f}%{END}"
                
            print(f"Performance: {current_return_str}")
            
            print(f"\n{BOLD}Prévision à {forecast['horizon']} jours:{END}")
            print(f"Valeur future estimée: ${position['future_value']:.2f}")
            
            if position['future_return'] >= 0:
                future_return_str = f"{GREEN}+{position['future_return']:.1f}%{END}"
            else:
                future_return_str = f"{RED}{position['future_return']:.1f}%{END}"
                
            print(f"Performance projetée: {future_return_str}")
            print(f"Fiabilité prédiction: {forecast['reliability']}%")
        
        # Afficher la recommandation
        print("\n" + "-" * 80)
        print(f"{BOLD}Recommandation: {reco_color}{recommendation['type']}{END}")
        print(f"Confiance: {recommendation['confidence']}%")
        print(f"Explication: {recommendation['explanation']}")
        if 'source' in recommendation:
            print(f"Source: {recommendation['source']}")
        print("-" * 80 + "\n")


def interactive_asset_selection():
    """Fonction interactive pour sélectionner un actif"""
    # Demander le symbole de l'actif
    asset_symbol = input("\nEntrez le symbole/identifiant de l'actif (ex: BTC, ETH, AAPL): ")
    
    # Demander explicitement le type d'actif
    print("\nType d'actif :")
    print("1. Crypto-monnaie")
    print("2. Action")
    asset_type_choice = input("Votre choix (1/2) [1]: ")
    
    # Déterminer le type d'actif
    if asset_type_choice == "2":
        asset_type = "stock"
    else:
        asset_type = "crypto"
        
    return asset_symbol, asset_type


