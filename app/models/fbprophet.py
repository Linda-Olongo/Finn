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
from typing import Dict, Tuple, List, Any, Optional
from datetime import datetime, timedelta

# Suppression des avertissements
warnings.filterwarnings('ignore')

class ProphetModel:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.forecast = None
        self.metrics = None
        self.df = None
        self.symbol = None
        self.asset_type = None
        
    def load_and_prepare_data(self, symbol: str, asset_type: str = 'crypto') -> pd.DataFrame:
        """Charge et prépare les données sans visualisation"""
        print(f"\n=== Chargement des données pour {symbol} ===")
        self.symbol = symbol
        self.asset_type = asset_type
        
        # Récupération des données
        if asset_type == 'crypto':
            current_data = self.preprocessor.collector.get_crypto_current(symbol)
            historical_data = self.preprocessor.collector.get_crypto_historical(symbol)
            
            print(f"\nDonnées actuelles pour {current_data['name']} ({current_data['id']}):")
            print(f"Prix: ${current_data['price']} ({current_data['change_24h']:+.2f}%)")
            print(f"Volume 24h: ${current_data['volume']:,.2f}")
            print(f"Market Cap: ${current_data['market_cap']:,.2f}")
            
        else:
            current_data = self.preprocessor.collector.get_stock_current(symbol)
            historical_data = self.preprocessor.collector.get_stock_historical(symbol)
            
            print(f"\nDonnées actuelles pour {current_data['name']} ({current_data['symbol']}):")
            print(f"Prix: ${current_data['price']} ({current_data['change']:+.2f}%)")
            print(f"Volume: {current_data['volume']:,}")
            print(f"P/E Ratio: {current_data['pe_ratio']}")
        
        # Informations sur les données historiques sans visualisation
        print("\nAperçu des données historiques:")
        print("\nPremières lignes:")
        print(historical_data.head())
        print("\nDernières lignes:")
        print(historical_data.tail())
        
        # Préparation pour Prophet
        df = historical_data.reset_index()
        
        # Suppression du timezone
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Création du DataFrame Prophet
        self.df = pd.DataFrame()
        self.df['ds'] = df['Date']
        self.df['y'] = df['Close']
        if 'Volume' in df.columns:
            self.df['volume'] = df['Volume']
        if 'Market Cap' in df.columns:
            self.df['market_cap'] = df['Market Cap']
        
        # Traitement des valeurs aberrantes
        self._handle_outliers()
        
        print("\nDonnées préparées pour Prophet:")
        print(self.df.head())
        
        return self.df
    
    def _handle_outliers(self) -> None:
        """Détecte et traite les valeurs aberrantes dans les données de prix"""
        # Utilisation de la méthode IQR (Interquartile Range)
        Q1 = self.df['y'].quantile(0.25)
        Q3 = self.df['y'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Définition des limites (plus souples pour les marchés financiers)
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
        
        # Nombre de valeurs aberrantes détectées
        outliers = self.df[(self.df['y'] < lower_bound) | (self.df['y'] > upper_bound)]
        outlier_count = len(outliers)
        
        # Traitement des valeurs aberrantes - remplacer par la médiane mobile
        if outlier_count > 0:
            print(f"Valeurs aberrantes détectées: {outlier_count}")
            
            # Pour chaque valeur aberrante, remplacer par la médiane des 5 jours précédents
            for idx in outliers.index:
                if idx > 5:
                    window = self.df.loc[idx-5:idx-1, 'y']
                    self.df.loc[idx, 'y'] = window.median()
            
            print(f"Valeurs aberrantes remplacées par la médiane mobile")
        else:
            print("Aucune valeur aberrante détectée")

    def build_model(self) -> Prophet:
        """Construction et entraînement du modèle Prophet avec paramètres fixes optimisés"""
        print("\n=== Construction du modèle Prophet ===")
        
        # Paramètres pré-optimisés pour les actifs financiers
        changepoint_prior_scale = 0.5
        seasonality_prior_scale = 10.0
        seasonality_mode = 'additive'
        
        # Configuration du modèle
        print("\nConfiguration des paramètres:")
        print(f"- Saisonnalité annuelle: True")
        print(f"- Saisonnalité hebdomadaire: True")
        print(f"- Saisonnalité journalière: False")
        print(f"- Changepoint prior scale: {changepoint_prior_scale}")
        print(f"- Seasonality prior scale: {seasonality_prior_scale}")
        print(f"- Seasonality mode: {seasonality_mode}")
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode
        )
        
        # Ajout des régresseurs
        if 'volume' in self.df.columns:
            print("\nAjout du volume comme régresseur...")
            self.model.add_regressor('volume')
        
        if 'market_cap' in self.df.columns:
            print("Ajout de la market cap comme régresseur...")
            self.model.add_regressor('market_cap')
        
        # Entraînement
        print("\nEntraînement du modèle...")
        self.model.fit(self.df)
        print("✓ Modèle entraîné avec succès!")
        
        return self.model

    def make_predictions(self, periods: int = 365) -> pd.DataFrame:
        """Génère les prédictions pour un nombre de jours spécifié"""
        print(f"\n=== Génération des prédictions ({periods} jours) ===")
        
        # Création du dataframe futur
        future = self.model.make_future_dataframe(periods=periods)
        
        # Prévision des régresseurs pour les périodes futures
        if 'volume' in self.df.columns:
            # Utilisation de la tendance récente (moyenne des 30 derniers jours) au lieu de la moyenne globale
            future['volume'] = self.df['volume'].tail(30).mean()
        
        if 'market_cap' in self.df.columns:
            # Utilisation de la tendance récente (moyenne des 30 derniers jours) au lieu de la moyenne globale
            future['market_cap'] = self.df['market_cap'].tail(30).mean()
        
        # Prédictions
        self.forecast = self.model.predict(future)
        
        # Ajout des intervalles de confiance personnalisés pour les marchés financiers
        self._add_custom_intervals()
        
        # Date de la dernière donnée historique pour le filtrage
        last_date = self.df['ds'].max()
        
        # Filtrer uniquement les prédictions futures pour l'aperçu
        future_forecast = self.forecast[self.forecast['ds'] > last_date]
        
        # Affichage des prédictions futures uniquement
        print("\nAperçu des prédictions futures:")
        forecast_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'yhat_lower_90', 'yhat_upper_90', 'yhat_lower_50', 'yhat_upper_50']
        print("\nPremières lignes:")
        print(future_forecast[forecast_cols].head())
        print("\nDernières lignes:")
        print(future_forecast[forecast_cols].tail())
        
        return self.forecast
    
    def _add_custom_intervals(self) -> None:
        """Ajoute des intervalles de confiance personnalisés adaptés aux marchés financiers"""
        # Les marchés financiers sont souvent plus volatils que ce que Prophet prédit
        # On ajoute des intervalles de confiance plus larges pour le secteur financier
        
        # Calcul de la volatilité historique (écart-type des rendements journaliers)
        historical_returns = np.log(self.df['y'] / self.df['y'].shift(1)).dropna()
        historical_volatility = historical_returns.std()
        
        # Intervalles de confiance personnalisés (plus larges pour le secteur financier)
        # 90% - très conservateur pour le risque élevé
        self.forecast['yhat_lower_90'] = self.forecast['yhat'] * np.exp(-1.645 * historical_volatility * np.sqrt(1 + np.arange(len(self.forecast)) / 252))
        self.forecast['yhat_upper_90'] = self.forecast['yhat'] * np.exp(1.645 * historical_volatility * np.sqrt(1 + np.arange(len(self.forecast)) / 252))
        
        # 50% - plus réaliste pour les prévisions à court terme
        self.forecast['yhat_lower_50'] = self.forecast['yhat'] * np.exp(-0.674 * historical_volatility * np.sqrt(1 + np.arange(len(self.forecast)) / 252))
        self.forecast['yhat_upper_50'] = self.forecast['yhat'] * np.exp(0.674 * historical_volatility * np.sqrt(1 + np.arange(len(self.forecast)) / 252))

    def evaluate_model(self) -> Dict[str, float]:
        """Évalue les performances du modèle sans visualisation"""
        print("\n=== Évaluation du modèle ===")
        
        # Calcul des métriques sur les données d'entraînement
        train_indices = self.df.index
        y_true = self.df['y'].values
        y_pred = self.forecast['yhat'][train_indices].values
        
        self.metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R2': r2_score(y_true, y_pred)
        }
        
        # Affichage des métriques
        print("\nMétriques de performance:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.2f}")
        
        return self.metrics
    
    def predict_for_date(self, target_date: str) -> Dict[str, Any]:
        """Prédit le prix pour une date spécifique (format 'YYYY-MM-DD')"""
        # Vérifier si le modèle a été construit
        if self.model is None:
            raise ValueError("Le modèle n'a pas été construit. Appelez build_model() d'abord.")
            
        # Vérifier si les prédictions ont été générées
        if self.forecast is None:
            # Calculer le nombre de jours entre la dernière date connue et la date cible
            last_date = self.df['ds'].max()
            target_date_dt = pd.to_datetime(target_date)
            days_diff = (target_date_dt - last_date).days + 1
            
            # Si la date cible est dans le passé, retourner les données historiques
            if days_diff <= 0:
                historical_data = self.df[self.df['ds'] == target_date_dt]
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
            
            # Générer des prédictions sur un horizon suffisant
            periods = max(365, days_diff + 30)  # Au moins assez de jours pour atteindre la date cible
            self.make_predictions(periods=periods)
        
        # Rechercher la date cible dans les prédictions
        target_date_dt = pd.to_datetime(target_date)
        target_forecast = self.forecast[self.forecast['ds'] == target_date_dt]
        
        # Si la date est trouvée dans les prédictions
        if len(target_forecast) > 0:
            prediction = target_forecast.iloc[0]
            return {
                'date': target_date,
                'price': round(prediction['yhat'], 2),
                'price_lower_90': round(prediction['yhat_lower_90'], 2),
                'price_upper_90': round(prediction['yhat_upper_90'], 2),
                'price_lower_50': round(prediction['yhat_lower_50'], 2),
                'price_upper_50': round(prediction['yhat_upper_50'], 2),
                'is_historical': False
            }
        else:
            # La date est trop lointaine ou invalide
            return {
                'date': target_date,
                'error': 'Date hors de la plage de prédiction'
            }
    
    def forecast_specific_period(self, days: int = 30) -> pd.DataFrame:
        """Génère des prédictions pour un nombre spécifique de jours dans le futur uniquement"""
        if days > 365:
            print("Attention: Les prédictions au-delà de 365 jours peuvent être moins fiables.")
            days = 365
        
        # Date de la dernière donnée historique
        last_date = self.df['ds'].max()
        
        # Si nous avons déjà fait des prédictions
        if self.forecast is None:
            # Générer des prédictions
            self.make_predictions(periods=days)
        
        # Filtrer uniquement les dates futures
        future_forecast = self.forecast[self.forecast['ds'] > last_date].head(days)
        
        # Vérifier qu'il y a des données futures
        if len(future_forecast) == 0:
            print("Erreur: Pas de prédictions futures disponibles.")
            return pd.DataFrame()
        
        # Création d'un DataFrame avec les informations essentielles
        result = pd.DataFrame({
            'Date': future_forecast['ds'],
            'Prix prédit': future_forecast['yhat'].round(2),
            'Intervalle bas (90%)': future_forecast['yhat_lower_90'].round(2),
            'Intervalle haut (90%)': future_forecast['yhat_upper_90'].round(2),
            'Intervalle bas (50%)': future_forecast['yhat_lower_50'].round(2),
            'Intervalle haut (50%)': future_forecast['yhat_upper_50'].round(2)
        })
        
        print(f"\n=== Prédictions pour les {len(result)} prochains jours ===")
        print(result)
        
        return result

def run_analysis():
    """Interface interactive pour l'analyse avec optimisations de performance"""
    prophet_model = ProphetModel()
    
    print("=== Analyse de Séries Temporelles avec Prophet (Version Optimisée) ===")
    
    # Sélection du type d'actif
    asset_type = input("\nType d'actif (stock/crypto): ").lower()
    if asset_type not in ['stock', 'crypto']:
        print("Type d'actif invalide. Utilisation par défaut: crypto")
        asset_type = 'crypto'
    
    # Sélection du symbole
    symbol = input(f"\nEntrez le symbole ou le nom de l'{'action' if asset_type == 'stock' else 'crypto'}: ")
    
    try:
        # Chargement et préparation des données
        print("\nChargement des données...")
        df = prophet_model.load_and_prepare_data(symbol, asset_type)
        
        # Construction du modèle
        print("\nConstruction du modèle...")
        model = prophet_model.build_model()
        
        # Mode d'utilisation
        print("\nChoisissez le mode d'utilisation:")
        print("1. Prédiction pour un nombre de jours spécifique")
        print("2. Prédiction pour une date spécifique")
        mode = input("Mode (1 ou 2): ")
        
        if mode == "1":
            # Demande du nombre de jours pour la prédiction
            days_to_predict = int(input("\nNombre de jours à prédire (max 365): "))
            
            # Génération des prédictions spécifiques
            print("\nGénération des prédictions futures...")
            forecast = prophet_model.forecast_specific_period(days=days_to_predict)
        else:
            # Demande de la date cible
            target_date = input("\nDate cible (format YYYY-MM-DD): ")
            
            # Prédiction pour la date spécifique
            print(f"\nPrédiction pour le {target_date}...")
            result = prophet_model.predict_for_date(target_date)
            print(result)
        
        # Évaluation du modèle
        print("\nÉvaluation du modèle...")
        metrics = prophet_model.evaluate_model()
            
        print("\nAnalyse terminée.")
        
    except Exception as e:
        print(f"\nErreur lors de l'analyse: {e}")

# Point d'entrée pour exécution standalone
if __name__ == "__main__":
    run_analysis()