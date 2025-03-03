import yfinance as yf
import pandas as pd
import requests
from typing import Optional, Tuple, Dict
from datetime import datetime
from pycoingecko import CoinGeckoAPI
import time 

class SymbolMapper:
    """Classe pour mapper les noms/symboles aux identifiants corrects"""
    
    def get_stock_info(self, query: str) -> Optional[Tuple[str, str]]:
        """Recherche un symbole boursier via l'API Yahoo Finance"""
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if 'quotes' in data and data['quotes']:
                for quote in data['quotes']:
                    if quote.get('quoteType') == 'EQUITY':
                        return (quote['symbol'], quote.get('longname', quote['symbol']))
            return None
        except Exception as e:
            print(f"Erreur lors de la recherche Yahoo Finance: {e}")
            return None

    def get_crypto_info(self, query: str) -> Optional[Tuple[str, str]]:
        """Recherche une crypto via l'API CoinGecko"""
        try:
            # D'abord, essayer de trouver par ID exact
            search_url = f"https://api.coingecko.com/api/v3/coins/{query.lower()}"
            response = requests.get(search_url)
            if response.status_code == 200:
                data = response.json()
                return (data['id'], data['name'])

            # Sinon, faire une recherche
            search_url = f"https://api.coingecko.com/api/v3/search?query={query}"
            response = requests.get(search_url)
            response.raise_for_status()
            data = response.json()
            
            if data.get('coins'):
                coin = data['coins'][0]
                return (coin['id'], coin['name'])
            return None
        except Exception as e:
            print(f"Erreur lors de la recherche CoinGecko: {e}")
            return None

class DataCollector:
    """Classe pour la collecte des données financières"""
    
    def __init__(self):
        self.mapper = SymbolMapper()
        self.cg = CoinGeckoAPI()
        self.retry_delay = 60  # 60 secondes entre les tentatives
        self.max_retries = 3   # Maximum 3 tentatives

    def get_stock_historical(self, query: str) -> pd.DataFrame:
        """Récupère les données historiques d'une action"""
        result = self.mapper.get_stock_info(query)
        if not result:
            raise ValueError(f"Action non trouvée: {query}")

        symbol, name = result
        try:
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0'})
            stock = yf.Ticker(symbol, session=session)
            
            # Récupération des données avec auto_adjust=True pour avoir les prix ajustés
            df = stock.history(period="5y")  # Sans auto_adjust
            if df.empty:
                raise ValueError(f"Aucune donnée historique pour {symbol}")
            
            # Gestion de l'Adj Close si non présent
            if 'Adj Close' not in df.columns:
                # Si pas d'Adj Close, on utilise Close comme approximation
                df['Adj Close'] = df['Close']
                
            # Sélection des colonnes selon les captures
            df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            
            # Ajout des métadonnées
            df.attrs['symbol'] = symbol
            df.attrs['name'] = name
            df.attrs['type'] = 'stock'
            
            return df.round(2)

        except Exception as e:
            raise RuntimeError(f"Erreur lors de la collecte des données historiques: {e}")

    def get_stock_current(self, query: str) -> Dict:
        """Récupère les données en temps réel d'une action"""
        result = self.mapper.get_stock_info(query)
        if not result:
            raise ValueError(f"Action non trouvée: {query}")

        symbol, name = result
        try:
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0'})
            stock = yf.Ticker(symbol, session=session)
            
            info = stock.info
            df = stock.history(period='5d')
            
            current_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else df['Open'].iloc[-1]
            change = ((current_price - prev_close) / prev_close) * 100

            return {
                'name': name,
                'symbol': symbol,
                'price': round(current_price, 2),
                'change': round(change, 2),
                'change_value': round(current_price - prev_close, 2),
                'volume': int(df['Volume'].iloc[-1]),
                'open': round(df['Open'].iloc[-1], 2),
                'high': round(df['High'].iloc[-1], 2),
                'low': round(df['Low'].iloc[-1], 2),
                'pe_ratio': info.get('forwardPE'),
                'market_cap': info.get('marketCap'),
                'dividend_yield': info.get('dividendYield')
            }

        except Exception as e:
            raise RuntimeError(f"Erreur lors de la collecte des données actuelles: {e}")

    def get_crypto_historical(self, query: str) -> pd.DataFrame:
        """Récupère les données historiques d'une crypto"""
        result = self.mapper.get_crypto_info(query)
        if not result:
            raise ValueError(f"Crypto non trouvée: {query}")

        crypto_id, name = result
        try:
            # Récupération des données avec le maximum d'historique
            # Limité à 365 jours pour l'API gratuite
            data = self._make_coingecko_request(
                self.cg.get_coin_market_chart_by_id,
                id=crypto_id,
                vs_currency='usd',
                days='365',
                interval='daily'
            )
            
            # Création du DataFrame avec toutes les données
            prices_df = pd.DataFrame(data['prices'], columns=['Date', 'Close'])
            market_caps_df = pd.DataFrame(data['market_caps'], columns=['Date', 'Market Cap'])
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['Date', 'Volume'])
            
            # Fusion des données
            df = pd.merge(prices_df, market_caps_df, on='Date')
            df = pd.merge(df, volumes_df, on='Date')
            
            # Conversion de la date et traitement pour ne garder que les données quotidiennes
            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
            df['Date'] = df['Date'].dt.normalize()  # Normalise à minuit
            
            # Grouper par jour et prendre la dernière valeur de chaque jour
            df = df.groupby('Date').last().reset_index()
            
            # Ne garder que les jours complets (pas aujourd'hui)
            df = df[df['Date'].dt.date < datetime.now().date()]
            
            df.set_index('Date', inplace=True)
            
            # Ajout de Open (prix de clôture du jour précédent)
            df['Open'] = df['Close'].shift(1)
            
            # Réorganisation en dataframe
            df = df[['Market Cap', 'Volume', 'Open', 'Close']]
            
            # Arrondir les valeurs
            df = df.round(2)
            
            # Ajout des métadonnées
            df.attrs['id'] = crypto_id
            df.attrs['name'] = name
            df.attrs['type'] = 'crypto'
            
            return df.round(2)

        except Exception as e:
            raise RuntimeError(f"Erreur lors de la collecte des données historiques: {e}")

    def _make_coingecko_request(self, func, *args, **kwargs):
        """Fonction utilitaire pour gérer les requêtes CoinGecko avec retry"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) and attempt < self.max_retries - 1:
                    print(f"Limite de taux atteinte. Attente de {self.retry_delay} secondes...")
                    time.sleep(self.retry_delay)
                    continue
                raise

    def get_crypto_current(self, query: str) -> Dict:
        """Récupère les données en temps réel d'une crypto"""
        result = self.mapper.get_crypto_info(query)
        if not result:
            raise ValueError(f"Crypto non trouvée: {query}")

        crypto_id, name = result
        try:
            # Récupération des données via l'API coins/markets
            market_data = self._make_coingecko_request(
                self.cg.get_coins_markets,
                vs_currency='usd',
                ids=[crypto_id],
                order='market_cap_desc',
                per_page=1,
                page=1,
                sparkline=False
            )[0]

            price_data = {
                crypto_id: {
                    'usd': market_data['current_price'],
                    'usd_market_cap': market_data['market_cap'],
                    'usd_24h_vol': market_data['total_volume'],
                    'usd_24h_change': market_data['price_change_percentage_24h'] or 0
                }
            }
            
            # Récupération des données de marché supplémentaires
            market_data = self.cg.get_coins_markets(
                vs_currency='usd',
                ids=[crypto_id],
                per_page=1,
                page=1
            )[0]

            return {
                'name': name,
                'id': crypto_id,
                'price': round(price_data[crypto_id]['usd'], 2),
                'market_cap': round(price_data[crypto_id]['usd_market_cap'], 2),
                'volume': round(price_data[crypto_id]['usd_24h_vol'], 2),
                'change_24h': round(price_data[crypto_id]['usd_24h_change'], 2),
                'diluted_valuation': market_data.get('fully_diluted_valuation'),
                'circulating_supply': market_data.get('circulating_supply'),
                'total_supply': market_data.get('total_supply'),
                'max_supply': market_data.get('max_supply')
            }

        except Exception as e:
            raise RuntimeError(f"Erreur lors de la collecte des données actuelles: {e}")

if __name__ == "__main__":
    collector = DataCollector()
    
    while True:
        print("\n=== Collecteur de Données Financières ===")
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
        
        query = input("\nEntrez le nom ou le symbole : ")
        
        try:
            if choice == "1":
                print(f"\nRécupération des données pour : {query}")
                hist_data = collector.get_stock_historical(query)
                current_data = collector.get_stock_current(query)
                
                print(f"\n=== {current_data['name']} ({current_data['symbol']}) ===")
                print(f"Prix actuel: {current_data['price']}$ ({current_data['change']:+.2f}%)")
                print(f"Volume: {current_data['volume']:,}")
                print(f"Ouverture: {current_data['open']}$")
                print(f"+Haut: {current_data['high']}$")
                print(f"+Bas: {current_data['low']}$")
                
            else:
                print(f"\nRécupération des données pour : {query}")
                hist_data = collector.get_crypto_historical(query)
                current_data = collector.get_crypto_current(query)
                
                print(f"\n=== {current_data['name']} ({current_data['id']}) ===")
                print(f"Prix actuel: {current_data['price']}$ ({current_data['change_24h']:+.2f}%)")
                print(f"Volume 24h: {current_data['volume']:,}$")
                print(f"Capitalisation: {current_data['market_cap']:,}$")
                
                # Gestion sécurisée de l'évaluation après dilution
                if current_data['diluted_valuation']:
                    print(f"Évaluation après dilution: {current_data['diluted_valuation']:,}$")
                else:
                    print("Évaluation après dilution: Non disponible")
                
                # Gestion sécurisée de l'offre en circulation
                if current_data['circulating_supply']:
                    print(f"Offre en circulation: {current_data['circulating_supply']:,.2f}")
                else:
                    print("Offre en circulation: Non disponible")
                
                # Gestion sécurisée de l'offre totale
                if current_data['total_supply']:
                    print(f"Offre totale: {current_data['total_supply']:,.2f}")
                else:
                    print("Offre totale: Non disponible")
                
                # Gestion sécurisée de l'offre maximale
                if current_data['max_supply']:
                    print(f"Offre maximale: {current_data['max_supply']:,.2f}")
                else:
                    print("Offre maximale: Non définie")

            
            print("\nDonnées historiques :")
            print("\nPremières lignes:")
            print(hist_data.head())
            print("\nDernières lignes:")
            print(hist_data.tail())
            
        except Exception as e:
            print(f"\nErreur : {e}")