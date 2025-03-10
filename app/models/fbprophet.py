import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

import pandas as pd
import numpy as np
from prophet import Prophet
from data.preprocessing import DataPreprocessor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import traceback
import time

# Gestion de cmdstanpy améliorée
def setup_cmdstan():
    try:
        import cmdstanpy
        cmdstan_path = os.path.expanduser('~/cmdstan_custom/cmdstan-2.36.0')
        if not os.path.exists(cmdstan_path):
            print(f"Installation de CmdStan dans {cmdstan_path}...")
            os.makedirs(os.path.dirname(cmdstan_path), exist_ok=True)
            cmdstanpy.install_cmdstan(dir=cmdstan_path, overwrite=True)
        cmdstanpy.set_cmdstan_path(cmdstan_path)
        return True
    except Exception as e:
        print(f"Avertissement: Problème avec CmdStan. Utilisation de L-BFGS par défaut. Erreur: {e}")
        return False

warnings.filterwarnings('ignore')

class ProphetModel:
    """Classe pour la prédiction de séries temporelles avec Prophet"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.forecast = None
        self.metrics = None
        self.prophet_data = None
        self.params = None
        self.cmdstan_available = setup_cmdstan()
        self.asset_name = None

    def load_data(self, symbol: str, asset_type: str = 'crypto') -> Dict:
        """Charge et prépare les données pour le modèle Prophet, ajuste les paramètres dynamiquement"""
        print(f"\n=== Chargement des données pour {symbol} ===")
        
        try:
            if asset_type == 'crypto':
                current_data = self.preprocessor.collector.get_crypto_current(symbol)
                historical_data = self.preprocessor.collector.get_crypto_historical(symbol)
                print(f"\nDonnées actuelles pour {current_data['name']} ({current_data['id']}):")
                print(f"Prix: ${current_data['price']} ({current_data['change_24h']:+.2f}%)")
                print(f"Volume 24h: ${current_data['volume']:,.2f}")
                print(f"Market Cap: ${current_data['market_cap']:,.2f}")
                self.asset_name = current_data['name']
            else:
                current_data = self.preprocessor.collector.get_stock_current(symbol)
                historical_data = self.preprocessor.collector.get_stock_historical(symbol)
                print(f"\nDonnées actuelles pour {current_data['name']} ({current_data['symbol']}):")
                print(f"Prix: ${current_data['price']} ({current_data['change']:+.2f}%)")
                print(f"Volume: {current_data['volume']:,}")
                print(f"P/E Ratio: {current_data['pe_ratio']}")
                self.asset_name = current_data['name']
            
            self.prophet_data = self.preprocessor.prepare_for_prophet(historical_data, current_data, asset_type)
            
            # Ajustement dynamique des paramètres basé sur la volatilité
            volatility = self.prophet_data['volatility']
            print(f"Volatilité annualisée détectée: {volatility:.4f}")
            
            # Utiliser le mode additif pour tous les actifs, mais ajuster les autres paramètres
            if volatility < 0.3:  # Actifs stables
                self.params = {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10.0,
                    'seasonality_mode': 'additive',
                    'weekly_seasonality': True
                }
                print("Actif stable détecté. Paramètres appliqués: additif, saisonnalité forte.")
            else:  # Actifs volatils
                self.params = {
                    'changepoint_prior_scale': 0.01,  # Réduit pour moins de flexibilité
                    'seasonality_prior_scale': 5.0,
                    'seasonality_mode': 'additive',  # Changé de multiplicatif à additif
                    'weekly_seasonality': False      # Désactivé pour éviter les problèmes de week-end
                }
                print("Actif volatil détecté. Paramètres appliqués: additif, saisonnalité réduite, sans saisonnalité hebdomadaire.")
            
            return self.prophet_data
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                print("Limite de taux atteinte. Attente de 60 secondes...")
                time.sleep(60)
                return self.load_data(symbol, asset_type)
            print(f"Erreur lors du chargement des données: {e}")
            raise

    def build_model(self) -> Prophet:
        """Construction et entraînement du modèle Prophet avec paramètres dynamiques"""
        if self.prophet_data is None or self.params is None:
            raise ValueError("Les données ou paramètres n'ont pas été chargés. Appelez load_data() d'abord.")
        
        print("\n=== Construction du modèle Prophet ===")
        
        df = self.prophet_data['df']
        has_volume = 'volume' in df.columns
        has_market_cap = 'market_cap' in df.columns
        
        print("\nConfiguration des paramètres:")
        print(f"- Saisonnalité annuelle: True")
        print(f"- Saisonnalité hebdomadaire: {self.params['weekly_seasonality']}")
        print(f"- Saisonnalité journalière: False")
        print(f"- Changepoint prior scale: {self.params['changepoint_prior_scale']}")
        print(f"- Seasonality prior scale: {self.params['seasonality_prior_scale']}")
        print(f"- Seasonality mode: {self.params['seasonality_mode']}")
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=self.params['weekly_seasonality'],
            daily_seasonality=False,
            mcmc_samples=0,  # Désactivé car peut causer des problèmes avec CmdStan
            changepoint_prior_scale=self.params['changepoint_prior_scale'],
            seasonality_prior_scale=self.params['seasonality_prior_scale'],
            seasonality_mode=self.params['seasonality_mode'],
            interval_width=0.95
        )
        
        # Utiliser des régresseurs seulement si disponibles et significatifs
        if has_volume:
            # Vérifier si le volume a une corrélation significative avec le prix
            correlation = df[['y', 'volume']].corr().iloc[0, 1]
            if abs(correlation) > 0.2:  # Seuil arbitraire de corrélation
                print(f"\nAjout du volume comme régresseur (corrélation: {correlation:.2f})...")
                self.model.add_regressor('volume', standardize=True, mode='additive')
            else:
                print(f"\nVolume non ajouté comme régresseur (corrélation faible: {correlation:.2f})")
        
        if has_market_cap:
            correlation = df[['y', 'market_cap']].corr().iloc[0, 1]
            if abs(correlation) > 0.2:
                print(f"Ajout de la market cap comme régresseur (corrélation: {correlation:.2f})...")
                self.model.add_regressor('market_cap', standardize=True, mode='additive')
            else:
                print(f"Market cap non ajoutée comme régresseur (corrélation faible: {correlation:.2f})")
        
        print("\nEntraînement du modèle...")
        self.model.fit(df)
        print("✓ Modèle entraîné avec succès!")
        
        return self.model

    def make_predictions(self, periods: int = 365) -> pd.DataFrame:
        """Génère les prédictions pour un nombre de jours spécifié"""
        if self.model is None:
            raise ValueError("Le modèle n'a pas été construit. Appelez build_model() d'abord.")
            
        print(f"\n=== Génération des prédictions ({periods} jours) ===")
        
        df = self.prophet_data['df']
        
        # Création d'un dataframe futur uniquement avec les jours de bourse (du lundi au vendredi)
        current_date = pd.Timestamp.now().normalize()
        future_dates = [current_date + timedelta(days=i) for i in range(periods + 1)]
        # Exclure les week-ends si actif volatil
        if not self.params.get('weekly_seasonality', True):
            future_dates = [date for date in future_dates if date.weekday() < 5]
        
        future = pd.DataFrame({'ds': future_dates})
        
        has_volume = 'volume' in df.columns
        has_market_cap = 'market_cap' in df.columns
        
        if has_volume:
            future['volume'] = np.nan
        if has_market_cap:
            future['market_cap'] = np.nan
        
        # Remplir les données historiques pour les dates existantes
        future_dates_set = set(future['ds'])
        historical_dates = set(df['ds'])
        
        for date in historical_dates:
            if date in future_dates_set:
                mask_future = future['ds'] == date
                mask_df = df['ds'] == date
                if has_volume and any(mask_df):
                    future.loc[mask_future, 'volume'] = df.loc[mask_df, 'volume'].values[0]
                if has_market_cap and any(mask_df):
                    future.loc[mask_future, 'market_cap'] = df.loc[mask_df, 'market_cap'].values[0]
        
        future_only_mask = ~future['ds'].isin(historical_dates)
        
        # Amélioration de l'extrapolation pour les régresseurs
        if has_volume:
            # Utiliser une régression linéaire simple pour prédire le volume futur
            df_recent = df.tail(60).copy()  # Utiliser les 60 derniers jours pour la tendance
            df_recent['time_idx'] = np.arange(len(df_recent))
            if len(df_recent) > 30:  # Assez de données pour la régression
                from sklearn.linear_model import LinearRegression
                X = df_recent[['time_idx']]
                y = df_recent['volume']
                lr_model = LinearRegression().fit(X, y)
                
                # Calculer les prochains indices de temps
                next_indices = np.arange(len(df_recent), len(df_recent) + sum(future_only_mask))
                next_volumes = lr_model.predict(next_indices.reshape(-1, 1))
                
                # S'assurer que les volumes ne sont pas négatifs
                next_volumes = np.maximum(next_volumes, df['volume'].min())
                
                future.loc[future_only_mask, 'volume'] = next_volumes
            else:
                # Fallback sur la moyenne récente
                latest_volume = df['volume'].tail(30).ewm(span=7).mean().iloc[-1]
                future.loc[future_only_mask, 'volume'] = latest_volume
            
            # Vérifier les valeurs manquantes
            if future['volume'].isna().any():
                latest_volume = df['volume'].tail(30).ewm(span=7).mean().iloc[-1]
                future['volume'].fillna(latest_volume, inplace=True)
        
        if has_market_cap:
            # Approche similaire pour market_cap
            df_recent = df.tail(60).copy()
            df_recent['time_idx'] = np.arange(len(df_recent))
            if len(df_recent) > 30:
                from sklearn.linear_model import LinearRegression
                X = df_recent[['time_idx']]
                y = df_recent['market_cap']
                lr_model = LinearRegression().fit(X, y)
                
                next_indices = np.arange(len(df_recent), len(df_recent) + sum(future_only_mask))
                next_market_caps = lr_model.predict(next_indices.reshape(-1, 1))
                
                next_market_caps = np.maximum(next_market_caps, df['market_cap'].min())
                
                future.loc[future_only_mask, 'market_cap'] = next_market_caps
            else:
                latest_market_cap = df['market_cap'].tail(30).ewm(span=7).mean().iloc[-1]
                future.loc[future_only_mask, 'market_cap'] = latest_market_cap
            
            if future['market_cap'].isna().any():
                latest_market_cap = df['market_cap'].tail(30).ewm(span=7).mean().iloc[-1]
                future['market_cap'].fillna(latest_market_cap, inplace=True)
        
        # Vérifications finales
        if has_volume and future['volume'].isna().any():
            raise ValueError("NaN persistants dans 'volume'")
        if has_market_cap and future['market_cap'].isna().any():
            raise ValueError("NaN persistants dans 'market_cap'")
        
        try:
            self.forecast = self.model.predict(future)
            
            # S'assurer que les prédictions ne sont pas négatives
            self.forecast['yhat'] = self.forecast['yhat'].clip(lower=0)
            self.forecast['yhat_lower'] = self.forecast['yhat_lower'].clip(lower=0)
            self.forecast['yhat_upper'] = self.forecast['yhat_upper'].clip(lower=0)
            
            self._align_first_prediction()
            
            # Afficher l'aperçu des prédictions
            current_date = pd.Timestamp.now().normalize()
            future_forecast = self.forecast[self.forecast['ds'] >= current_date]
            
            print("\nAperçu des prédictions futures:")
            forecast_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            if not future_forecast.empty:
                print("\nPremières lignes:")
                print(future_forecast[forecast_cols].head())
                print("\nDernières lignes:")
                print(future_forecast[forecast_cols].tail())
            else:
                print("\nAucune prédiction future disponible.")
            
            return self.forecast
            
        except Exception as e:
            print(f"Erreur lors de la génération des prédictions: {e}")
            traceback.print_exc()
            raise
    
    def _align_first_prediction(self) -> None:
        """Ajuste la première prédiction pour correspondre au prix actuel"""
        if self.forecast is None or self.prophet_data is None:
            return
        
        try:
            current_price = self.prophet_data['current_price']
            current_date = pd.Timestamp.now().normalize()
            
            # Trouver l'indice de la première prédiction future
            future_forecast = self.forecast[self.forecast['ds'] >= current_date]
            if len(future_forecast) == 0:
                return
                
            first_future_idx = future_forecast.index[0]
            predicted_price = self.forecast.loc[first_future_idx, 'yhat']
            
            # Si la différence entre la prédiction et le prix actuel est trop grande, 
            # ajuster toutes les prédictions futures
            if abs(current_price - predicted_price) / current_price > 0.02:  # 2% d'écart
                adjust_factor = current_price / predicted_price
                future_indices = future_forecast.index
                for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                    if col in self.forecast.columns:
                        self.forecast.loc[future_indices, col] *= adjust_factor
                
                print(f"Prédictions ajustées pour correspondre au prix actuel de {current_price}$")
        except Exception as e:
            print(f"Impossible d'ajuster les prédictions: {e}")
    
    def evaluate_model(self) -> Dict[str, float]:
        """Évalue les performances du modèle sur les données d'entraînement"""
        if self.model is None:
            raise ValueError("Le modèle doit être construit avant l'évaluation.")
            
        print("\n=== Évaluation du modèle ===")
        
        # Créer des prédictions spécifiquement pour les données historiques
        df = self.prophet_data['df']
        historical_future = df[['ds']].copy()
        
        if 'volume' in df.columns and hasattr(self.model, 'extra_regressors') and 'volume' in self.model.extra_regressors:
            historical_future['volume'] = df['volume']
        if 'market_cap' in df.columns and hasattr(self.model, 'extra_regressors') and 'market_cap' in self.model.extra_regressors:
            historical_future['market_cap'] = df['market_cap']
        
        # Générer des prédictions sans ajustement
        historical_forecast = self.model.predict(historical_future)
        
        # Comparer avec les valeurs réelles
        y_true = df['y'].values
        y_pred = historical_forecast['yhat'].values
        
        if len(y_true) > 0:
            self.metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                'R2': r2_score(y_true, y_pred)
            }
            
            print("\nMétriques de performance:")
            for metric, value in self.metrics.items():
                print(f"{metric}: {value:.2f}")
            
            return self.metrics
        else:
            print("Pas assez de données pour calculer les métriques")
            return {}
    
    def backtest_model(self, test_days: int = 30) -> pd.DataFrame:
        """Évalue les performances du modèle sur les derniers jours connus"""
        if self.prophet_data is None:
            raise ValueError("Les données n'ont pas été chargées. Appelez load_data() d'abord.")
            
        print(f"\n=== Analyse de performance sur les {test_days} derniers jours ===")
        
        df = self.prophet_data['df']
        if len(df) <= test_days:
            print("Pas assez de données pour l'analyse rétrospective")
            return pd.DataFrame()
        
        train_df = df.iloc[:-test_days].copy()
        test_df = df.iloc[-test_days:].copy()
        
        has_volume = 'volume' in df.columns
        has_market_cap = 'market_cap' in df.columns
        
        # Créer un modèle identique au modèle principal mais entraîné uniquement sur les données historiques
        test_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=self.params.get('weekly_seasonality', True),
            daily_seasonality=False,
            changepoint_prior_scale=self.params['changepoint_prior_scale'],
            seasonality_prior_scale=self.params['seasonality_prior_scale'],
            seasonality_mode=self.params['seasonality_mode']
        )
        
        # Ajouter seulement les régresseurs significatifs
        if has_volume and hasattr(self.model, 'extra_regressors') and 'volume' in self.model.extra_regressors:
            test_model.add_regressor('volume', standardize=True, mode='additive')
        if has_market_cap and hasattr(self.model, 'extra_regressors') and 'market_cap' in self.model.extra_regressors:
            test_model.add_regressor('market_cap', standardize=True, mode='additive')
        
        test_model.fit(train_df)
        
        # Créer un dataframe pour la prédiction incluant les données historiques et futures
        future = pd.DataFrame({'ds': df['ds']})
        if has_volume:
            future['volume'] = df['volume'].values
        if has_market_cap:
            future['market_cap'] = df['market_cap'].values
        
        try:
            forecast = test_model.predict(future)
            
            # S'assurer que les prédictions ne sont pas négatives
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            
            # Préparer la comparaison entre prédictions et réalité
            comparison = pd.DataFrame()
            comparison['Date'] = test_df['ds']
            comparison['Prix réel'] = test_df['y'].round(2)
            
            test_dates = set(test_df['ds'])
            forecast_test = forecast[forecast['ds'].isin(test_dates)]
            
            date_to_pred = dict(zip(forecast_test['ds'], forecast_test['yhat']))
            comparison['Prix prédit'] = comparison['Date'].map(date_to_pred).round(2)
            
            # Calculer les métriques d'erreur
            comparison['Écart'] = (comparison['Prix prédit'] - comparison['Prix réel']).round(2)
            comparison['Erreur (%)'] = (abs(comparison['Prix réel'] - comparison['Prix prédit']) / comparison['Prix réel'] * 100).round(2)
            
            print("\nComparaison prédictions vs réalité:")
            print(comparison)
            
            # Calculer les moyennes d'erreur
            mape = comparison['Erreur (%)'].mean()
            mae = abs(comparison['Écart']).mean()
            
            print(f"\nMAPE moyen: {mape:.2f}%")
            print(f"MAE moyen: {mae:.2f}")
            
            # Calculer les métriques sur les dernières données
            last_week = comparison.tail(7)
            if len(last_week) > 0:
                last_mape = last_week['Erreur (%)'].mean()
                last_mae = abs(last_week['Écart']).mean()
                print(f"\nMAPE dernière semaine: {last_mape:.2f}%")
                print(f"MAE dernière semaine: {last_mae:.2f}")
            
            return comparison
            
        except Exception as e:
            print(f"Erreur lors du backtesting: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def predict_for_date(self, target_date: str) -> Dict[str, Any]:
        """Prédit le prix pour une date spécifique (format 'YYYY-MM-DD')"""
        if self.model is None:
            raise ValueError("Le modèle n'a pas été construit. Appelez build_model() d'abord.")
            
        if self.forecast is None:
            df = self.prophet_data['df']
            last_date = df['ds'].max()
            target_date_dt = pd.to_datetime(target_date)
            days_diff = (target_date_dt - last_date).days + 1
            
            if days_diff <= 0:
                historical_data = df[df['ds'] == target_date_dt]
                if len(historical_data) > 0:
                    return {
                        'date': target_date,
                        'price': historical_data['y'].values[0],
                        'is_historical': True
                    }
                else:
                    return {
                        'date': target_date,
                        'error': 'Date dans le passé sans données disponibles'
                    }
            
            periods = max(365, days_diff + 30)
            self.make_predictions(periods=periods)
        
        target_date_dt = pd.to_datetime(target_date)
        
        # Vérifier si c'est un week-end et si l'actif est sensible aux week-ends
        is_weekend = target_date_dt.weekday() >= 5
        if is_weekend and not self.params.get('weekly_seasonality', True):
            return {
                'date': target_date,
                'error': 'Date correspondant à un week-end (marché fermé)'
            }
        
        target_forecast = self.forecast[self.forecast['ds'] == target_date_dt]
        
        if len(target_forecast) > 0:
            prediction = target_forecast.iloc[0]
            return {
                'date': target_date,
                'price': round(prediction['yhat'], 2),
                'price_lower': round(prediction['yhat_lower'], 2),
                'price_upper': round(prediction['yhat_upper'], 2),
                'is_historical': False
            }
        else:
            # Si la date n'est pas trouvée, chercher la date la plus proche
            all_dates = self.forecast['ds'].values
            nearest_idx = np.argmin(np.abs(all_dates - target_date_dt))
            nearest_date = all_dates[nearest_idx]
            nearest_forecast = self.forecast[self.forecast['ds'] == nearest_date]
            
            if len(nearest_forecast) > 0:
                prediction = nearest_forecast.iloc[0]
                return {
                    'date': target_date,
                    'nearest_date': nearest_date.strftime('%Y-%m-%d'),
                    'price': round(prediction['yhat'], 2),
                    'price_lower': round(prediction['yhat_lower'], 2),
                    'price_upper': round(prediction['yhat_upper'], 2),
                    'is_historical': False,
                    'note': 'Date exacte non trouvée, utilisation de la date la plus proche'
                }
            else:
                return {
                    'date': target_date,
                    'error': 'Date hors de la plage de prédiction'
                }
    
    def forecast_specific_period(self, days: int = 30) -> pd.DataFrame:
        """Génère des prédictions pour un nombre spécifique de jours dans le futur uniquement"""
        if days > 365:
            print("Attention: Les prédictions au-delà de 365 jours peuvent être moins fiables.")
            days = 365
        
        current_date = pd.Timestamp.now().normalize()
        
        if self.forecast is None or self.forecast['ds'].max() < current_date + timedelta(days=days):
            self.make_predictions(periods=days)
        
        future_forecast = self.forecast[self.forecast['ds'] >= current_date].copy()
        
        # Si on ne prédit pas pour les week-ends, filtrer les résultats
        if not self.params.get('weekly_seasonality', True):
            weekday_mask = future_forecast['ds'].dt.weekday < 5
            future_forecast = future_forecast[weekday_mask]
        
        # Limiter au nombre de jours demandés
        future_forecast = future_forecast.head(days + 1)
        
        if len(future_forecast) == 0:
            print("Erreur: Pas de prédictions futures disponibles.")
            return pd.DataFrame()
        
        result = pd.DataFrame({
            'Date': future_forecast['ds'],
            'Prix prédit': future_forecast['yhat'].round(2),
            'Intervalle bas': future_forecast['yhat_lower'].round(2),
            'Intervalle haut': future_forecast['yhat_upper'].round(2)
        })
        
        # Assurer que la première ligne représente le prix actuel
        if self.prophet_data and 'current_price' in self.prophet_data and len(result) > 0:
            current_price = self.prophet_data['current_price']
            today_idx = result[result['Date'].dt.date == current_date.date()].index
            if len(today_idx) > 0:
                result.loc[today_idx[0], 'Prix prédit'] = current_price
                result.loc[today_idx[0], 'Intervalle bas'] = current_price * 0.95
                result.loc[today_idx[0], 'Intervalle haut'] = current_price * 1.05
        
        print(f"\n=== Prédictions pour les {len(result)} prochains jours ===")
        print(result)
        
        # Calculer la tendance globale
        if len(result) > 1:
            first_price = result['Prix prédit'].iloc[0]
            last_price = result['Prix prédit'].iloc[-1]
            price_change = ((last_price / first_price) - 1) * 100
            days_predicted = (result['Date'].iloc[-1] - result['Date'].iloc[0]).days
            
            print(f"\nTendance sur {days_predicted} jours: {price_change:+.2f}%")
            print(f"Prix actuel: {first_price:.2f}$")
            print(f"Prix prédit pour le {result['Date'].iloc[-1].strftime('%Y-%m-%d')}: {last_price:.2f}$")
        
        return result

def run_analysis():
    """Interface interactive pour l'analyse"""
    prophet_model = ProphetModel()
    
    print("=== Analyse de Séries Temporelles avec Prophet ===")
    
    asset_type = input("\nType d'actif (stock/crypto): ").lower()
    if asset_type not in ['stock', 'crypto']:
        print("Type d'actif invalide. Utilisation par défaut: crypto")
        asset_type = 'crypto'
    
    symbol = input(f"\nEntrez le symbole ou le nom de l'{'action' if asset_type == 'stock' else 'crypto'}: ")
    
    try:
        print("\nChargement des données...")
        prophet_data = prophet_model.load_data(symbol, asset_type)
        
        print("\nConstruction du modèle...")
        model = prophet_model.build_model()
        
        while True:
            print("\nChoisissez une option:")
            print("1. Prédiction pour un nombre de jours spécifique")
            print("2. Prédiction pour une date spécifique")
            print("3. Analyse de performance sur données historiques")
            print("4. Quitter")
            
            choice = input("\nOption (1-4): ")
            
            if choice == "1":
                days_to_predict = int(input("\nNombre de jours à prédire (max 365): ") or "30")
                print("\nGénération des prédictions futures...")
                forecast = prophet_model.forecast_specific_period(days=days_to_predict)
                prophet_model.evaluate_model()
                
            elif choice == "2":
                target_date = input("\nDate cible (format YYYY-MM-DD): ")
                print(f"\nPrédiction pour le {target_date}...")
                result = prophet_model.predict_for_date(target_date)
                print("\nRésultat de la prédiction:")
                for key, value in result.items():
                    print(f"{key}: {value}")
                prophet_model.evaluate_model()
                
            elif choice == "3":
                days_backtest = int(input("\nNombre de jours pour l'analyse (30 par défaut): ") or "30")
                print("\nAnalyse des performances...")
                backtest_results = prophet_model.backtest_model(test_days=days_backtest)
                
            elif choice == "4":
                print("\nAnalyse terminée.")
                break
                
            else:
                print("\nOption invalide. Veuillez réessayer.")
        
    except Exception as e:
        print(f"\nErreur lors de l'analyse: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_analysis()