import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from api_fetcher import DataCollector

class DataPreprocessor:
    """Classe pour le prétraitement des données financières"""
    
    def __init__(self):
        self.collector = DataCollector()

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
        if 'Close' in df.columns:
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

    def prepare_for_prophet(self, df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
        """Prépare les données pour Prophet"""
        prophet_df = pd.DataFrame()
        
        # Réinitialisation de l'index pour avoir la date comme colonne
        df = df.reset_index()
        
        # Renommage des colonnes pour Prophet
        prophet_df['ds'] = df['Date']
        prophet_df['y'] = df['Close'] if 'Close' in df.columns else df['price']
        
        # Ajout des variables supplémentaires
        if 'Volume' in df.columns:
            prophet_df['volume'] = df['Volume']
        
        if asset_type == 'crypto' and 'Market Cap' in df.columns:
            prophet_df['market_cap'] = df['Market Cap']
        
        # Conversion des dates au format attendu par Prophet
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        return prophet_df

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
            # Récupération des données via DataCollector
            if asset_type == "stock":
                current_data = preprocessor.collector.get_stock_current(query)
                hist_data = preprocessor.collector.get_stock_historical(query)
                print(f"\n=== {current_data['name']} ({current_data['symbol']}) ===")
                print(f"Prix actuel: {current_data['price']}$ ({current_data['change']:+.2f}%)")
                print(f"Volume: {current_data['volume']:,}")
                print(f"P/E Ratio: {current_data['pe_ratio']}")
                print(f"Market Cap: {current_data['market_cap']:,}$")
            else:
                current_data = preprocessor.collector.get_crypto_current(query)
                hist_data = preprocessor.collector.get_crypto_historical(query)
                print(f"\n=== {current_data['name']} ({current_data['id']}) ===")
                print(f"Prix actuel: {current_data['price']}$ ({current_data['change_24h']:+.2f}%)")
                print(f"Volume 24h: {current_data['volume']:,}$")
                print(f"Market Cap: {current_data['market_cap']:,}$")
            
            # Analyse exploratoire
            preprocessor.perform_eda(hist_data, asset_type)
            
            # Préparation pour Prophet
            prophet_data = preprocessor.prepare_for_prophet(hist_data, asset_type)
            print("\nDonnées préparées pour Prophet:")
            print("\nPremières lignes:")
            print(prophet_data.head())
            print("\nDernières lignes:")
            print(prophet_data.tail())
            
        except Exception as e:
            print(f"\nErreur : {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramme interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\nErreur inattendue : {e}")