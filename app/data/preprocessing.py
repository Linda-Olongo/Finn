import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from api_fetcher import DataCollector
import time
import traceback

class DataPreprocessor:
    """Classe pour le prétraitement des données financières"""
    
    def __init__(self):
        self.collector = DataCollector()
        self.max_retries = 3
        self.retry_delay = 2  # secondes

    def perform_eda(self, df: pd.DataFrame, asset_type: str) -> None:
        """Réalise l'analyse exploratoire des données"""
        print("\n=== Analyse Exploratoire des Données ===")
        
        # 1. Analyse statistique
        print("\nStatistiques descriptives:")
        print(df.describe().round(2))
        
        # 2. Analyse des valeurs manquantes
        print("\nAnalyse des valeurs manquantes:")
        print(df.isnull().sum())
        
        # 3. Types de données
        print("\nTypes de données:")
        print(df.dtypes)
        
        # 4. Analyse temporelle
        if 'Close' in df.columns:
            # Calcul des rendements
            df['Returns'] = df['Close'].pct_change()
            
            print("\nStatistiques des rendements:")
            returns_stats = df['Returns'].describe()
            print(f"Moyenne: {returns_stats['mean']:.4f}")
            print(f"Écart-type: {returns_stats['std']:.4f}")
            print(f"Skewness: {df['Returns'].skew():.4f}")
            print(f"Kurtosis: {df['Returns'].kurtosis():.4f}")
            
            # Volatilité sur 30 jours
            df['Volatility'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
            print(f"\nVolatilité moyenne sur 30 jours: {df['Volatility'].mean():.4f}")
        
        # 5. Analyse des volumes
        if 'Volume' in df.columns:
            print("\nAnalyse des volumes:")
            volume_stats = df['Volume'].describe()
            print(f"Volume moyen: {volume_stats['mean']:.0f}")
            print(f"Volume médian: {volume_stats['50%']:.0f}")
            print(f"Volume max: {volume_stats['max']:.0f}")
        
        # 6. Décomposition saisonnière
        if 'Close' in df.columns and len(df) >= 252:  # Au moins un an de données
            try:
                decomposition = seasonal_decompose(df['Close'], period=252)  # ~252 jours de trading par an
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
                
                decomposition.observed.plot(ax=ax1)
                ax1.set_title('Observed')
                decomposition.trend.plot(ax=ax2)
                ax2.set_title('Trend')
                decomposition.seasonal.plot(ax=ax3)
                ax3.set_title('Seasonal')
                decomposition.resid.plot(ax=ax4)
                ax4.set_title('Residual')
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Impossible de réaliser la décomposition saisonnière: {e}")
        elif 'Close' in df.columns:
            print("\nPas assez de données pour la décomposition saisonnière (moins d'un an).")

    def prepare_for_prophet(self, historical_data: pd.DataFrame, current_data: dict, asset_type: str) -> dict:
        """Prépare les données pour Prophet avec traitement des valeurs aberrantes"""
        print("\n=== Préparation des données pour Prophet ===")
        
        # Réinitialisation de l'index pour avoir la date comme colonne
        df = historical_data.reset_index()
        
        # Suppression du timezone si présent
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Création du DataFrame Prophet
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df['Date']
        prophet_df['y'] = df['Close'] if 'Close' in df.columns else df['price']
        
        # Ajout des variables supplémentaires
        if 'Volume' in df.columns:
            prophet_df['volume'] = df['Volume']
        
        if asset_type == 'crypto' and 'Market Cap' in df.columns:
            prophet_df['market_cap'] = df['Market Cap']
        
        # Traitement des valeurs aberrantes
        prophet_df = self._handle_outliers(prophet_df)
        
        # Extraction des métadonnées importantes
        if asset_type == 'stock':
            asset_name = current_data.get('name')
            asset_symbol = current_data.get('symbol')
        else:  # crypto
            asset_name = current_data.get('name')
            asset_symbol = current_data.get('id')
            
        current_price = current_data.get('price')
        
        # Création d'un dictionnaire complet avec les données et métadonnées
        prophet_data = {
            'df': prophet_df,
            'asset_type': asset_type,
            'asset_name': asset_name,
            'asset_symbol': asset_symbol,
            'current_price': current_price,
            'volatility': self.calculate_volatility(prophet_df)
        }
        
        print(f"\nDonnées préparées pour {asset_name} ({asset_symbol})")
        print(f"Nombre de lignes: {len(prophet_df)}")
        print(f"Période couverte: de {prophet_df['ds'].min().date()} à {prophet_df['ds'].max().date()}")
        print(f"Volatilité annualisée: {prophet_data['volatility']:.4f}")
        
        return prophet_data
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Détecte et traite les valeurs aberrantes dans les données de prix"""
        print("Recherche de valeurs aberrantes...")
        
        # Utilisation de la méthode IQR (Interquartile Range)
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Définition des limites (plus souples pour les marchés financiers)
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
        
        # Nombre de valeurs aberrantes détectées
        outliers = df[(df['y'] < lower_bound) | (df['y'] > upper_bound)]
        outlier_count = len(outliers)
        
        # Traitement des valeurs aberrantes avec lissage exponentiel (EMA)
        if outlier_count > 0:
            print(f"Valeurs aberrantes détectées: {outlier_count} ({outlier_count/len(df)*100:.2f}% des données)")
            
            # Calculer l'EMA sur 5 jours
            ema = df['y'].ewm(span=5).mean()
            
            # Pour chaque valeur aberrante, remplacer par l'EMA
            for idx in outliers.index:
                df.loc[idx, 'y'] = ema.loc[idx]
            
            print(f"Valeurs aberrantes remplacées par l'EMA")
        else:
            print("Aucune valeur aberrante détectée")
        
        return df
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calcule la volatilité historique pour les intervalles de confiance personnalisés"""
        # Calcul des rendements logarithmiques quotidiens
        if 'y' in df.columns and len(df) > 1:
            returns = np.log(df['y'] / df['y'].shift(1)).dropna()
            volatility = returns.std()
            annualized_volatility = volatility * np.sqrt(252)  # Annualisation (252 jours de trading)
            return annualized_volatility
        return 0.15  # Valeur par défaut si pas assez de données (15% est une volatilité moyenne)

    def get_data_with_retry(self, asset_type: str, query: str) -> tuple:
        """Récupère les données avec système de retry en cas d'erreur de connexion"""
        for attempt in range(self.max_retries):
            try:
                if asset_type == "stock":
                    current_data = self.collector.get_stock_current(query)
                    hist_data = self.collector.get_stock_historical(query)
                else:  # crypto
                    current_data = self.collector.get_crypto_current(query)
                    hist_data = self.collector.get_crypto_historical(query)
                
                return current_data, hist_data
                
            except Exception as e:
                if "Connection" in str(e) and attempt < self.max_retries - 1:
                    print(f"Erreur de connexion ({attempt+1}/{self.max_retries}). Nouvelle tentative dans {self.retry_delay} secondes...")
                    time.sleep(self.retry_delay)
                    # Augmenter progressivement le délai entre les tentatives
                    self.retry_delay *= 2
                else:
                    raise  # Remonter l'erreur si ce n'est pas une erreur de connexion ou si on a épuisé les tentatives

def main():
    preprocessor = DataPreprocessor()
    
    while True:
        print("\n=== Analyse et Prétraitement des Données Financières ===")
        print("1. Données d'une action")
        print("2. Données d'une crypto")
        print("3. Quitter")
        
        choice = input("\nVotre choix (1-3) : ")
        
        if choice == "3":
            print("Au revoir !")
            break
            
        if choice not in ["1", "2"]:
            print("Choix invalide. Veuillez réessayer.")
            continue
        
        asset_type = "stock" if choice == "1" else "crypto"
        query = input("\nEntrez le nom ou le symbole : ")
        
        try:
            # Récupération des données avec système de retry
            current_data, hist_data = preprocessor.get_data_with_retry(asset_type, query)
            
            # Affichage des informations actuelles
            if asset_type == "stock":
                print(f"\n=== {current_data['name']} ({current_data['symbol']}) ===")
                print(f"Prix actuel: {current_data['price']}$ ({current_data['change']:+.2f}%)")
                print(f"Volume: {current_data['volume']:,}")
                print(f"P/E Ratio: {current_data['pe_ratio']}")
                print(f"Market Cap: {current_data['market_cap']:,}$")
            else:
                print(f"\n=== {current_data['name']} ({current_data['id']}) ===")
                print(f"Prix actuel: {current_data['price']}$ ({current_data['change_24h']:+.2f}%)")
                print(f"Volume 24h: {current_data['volume']:,}$")
                print(f"Market Cap: {current_data['market_cap']:,}$")
            
            # Analyse exploratoire
            preprocessor.perform_eda(hist_data, asset_type)
            
            # Préparation pour Prophet
            prophet_data = preprocessor.prepare_for_prophet(hist_data, current_data, asset_type)
            print("\nDonnées préparées pour Prophet:")
            print("\nPremières lignes:")
            print(prophet_data['df'].head())
            print("\nDernières lignes:")
            print(prophet_data['df'].tail())
            
        except Exception as e:
            print(f"\nErreur : {e}")
            if "RemoteDisconnected" in str(e) or "Connection" in str(e):
                print("\nConseil: Vérifiez votre connexion internet ou réessayez plus tard.")
                print("Pour les actions, essayez d'utiliser le symbole boursier (ex: 'AAPL' pour Apple) plutôt que le nom complet.")
            
            # Afficher la trace complète en mode débogage
            traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramme interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\nErreur inattendue : {e}")
        traceback.print_exc()