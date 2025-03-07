import os
import sys
from pathlib import Path
import time
import backoff
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime, timedelta
import logging
import numpy as np
from dataclasses import dataclass
import yfinance as yf
import pandas as pd
import requests
import ta
import json
from functools import lru_cache

__all__ = ['MarketDataManager', 'DataCache', 'AssetInfo', 'main']

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from config.settings import Config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(root_dir) / "logs" / "market_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AssetInfo:
    """Classe pour stocker les informations d'un actif"""
    symbol: str
    name: str
    asset_type: str  # 'stock' ou 'crypto'
    current_price: float
    market_cap: float
    volume_24h: float
    change_24h: float
    last_updated: datetime = None
    
    def to_dict(self) -> dict:
        """Convertit l'objet en dictionnaire"""
        return {k: str(v) if isinstance(v, datetime) else v 
                for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AssetInfo':
        """Crée une instance depuis un dictionnaire"""
        if 'last_updated' in data and data['last_updated']:
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

class DataCache:
    """Gestionnaire de cache pour les données financières"""
    def __init__(self, cache_dir: Path, expiration: int = 3600):
        self.cache_dir = cache_dir
        self.expiration = expiration
        self.memory_cache = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Récupère les données du cache"""
        try:
            # Vérification du cache mémoire
            if key in self.memory_cache:
                data, timestamp = self.memory_cache[key]
                if (datetime.now() - timestamp).seconds < self.expiration:
                    return data.copy()

            # Vérification du cache disque
            cache_file = self.cache_dir / f"{key}.parquet"
            if cache_file.exists():
                metadata_file = self.cache_dir / f"{key}.meta.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    if (datetime.now() - datetime.fromisoformat(metadata['timestamp'])).seconds < self.expiration:
                        return pd.read_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Erreur lecture cache {key}: {e}")
        return None
    
    def set(self, key: str, df: pd.DataFrame) -> None:
        """Sauvegarde les données dans le cache"""
        try:
            # Cache mémoire
            self.memory_cache[key] = (df.copy(), datetime.now())

            # Cache disque
            cache_file = self.cache_dir / f"{key}.parquet"
            metadata_file = self.cache_dir / f"{key}.meta.json"
            
            # Sauvegarde des données
            df.to_parquet(cache_file)
            
            # Sauvegarde des métadonnées
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'rows': len(df),
                'columns': list(df.columns)
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            logger.warning(f"Erreur sauvegarde cache {key}: {e}")
    
    def clean(self, max_age: int = 86400) -> None:
        """Nettoie les fichiers de cache obsolètes"""
        try:
            now = datetime.now()
            for file in self.cache_dir.glob("*.parquet"):
                if file.stat().st_mtime < (now - timedelta(seconds=max_age)).timestamp():
                    file.unlink()
                    meta_file = file.with_suffix('.meta.json')
                    if meta_file.exists():
                        meta_file.unlink()
        except Exception as e:
            logger.warning(f"Erreur nettoyage cache: {e}")

class MarketDataManager:
    """Gestionnaire principal des données de marché"""
    def __init__(self):
        self.config = Config()
        self.session = self._initialize_session()
        self.cache = DataCache(
            self.config.MARKET_DATA_DIR,
            self.config.CACHE_CONFIG['EXPIRATION']
        )
        
    def _initialize_session(self) -> requests.Session:
        """Initialise la session HTTP avec les bons headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0'
        })
        return session

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def get_asset_info(self, symbol: str) -> AssetInfo:
        """Récupère les informations d'un actif"""
        try:
            # Tentative comme action
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            
            if hasattr(info, 'last_price') and info.last_price is not None:
                return AssetInfo(
                    symbol=symbol,
                    name=getattr(info, 'longname', symbol),
                    asset_type='stock',
                    current_price=info.last_price,
                    market_cap=getattr(info, 'market_cap', 0),
                    volume_24h=getattr(info, 'volume', 0),
                    change_24h=getattr(info, 'day_change', 0) * 100,
                    last_updated=datetime.now()
                )
            
            # Tentative comme crypto
            url = f"{self.config.API_CONFIG['COINGECKO']['BASE_URL']}/simple/price"
            params = {
                'ids': symbol.lower(),
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data and symbol.lower() in data:
                    crypto_data = data[symbol.lower()]
                    return AssetInfo(
                        symbol=symbol,
                        name=symbol,
                        asset_type='crypto',
                        current_price=crypto_data['usd'],
                        market_cap=crypto_data.get('usd_market_cap', 0),
                        volume_24h=crypto_data.get('usd_24h_vol', 0),
                        change_24h=crypto_data.get('usd_24h_change', 0),
                        last_updated=datetime.now()
                    )

            raise ValueError(f"Données non disponibles pour {symbol}")
            
        except Exception as e:
            logger.error(f"Erreur récupération info {symbol}: {str(e)}")
            raise

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def get_historical_data(self, symbol: str, timeframe: str = "1y") -> pd.DataFrame:
        """Récupère les données historiques d'un actif"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Vérifier le cache
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            # Tentative comme action
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=timeframe)
            
            if not df.empty:
                df = df.reset_index()
                df = self._prepare_stock_data(df)
                self.cache.set(cache_key, df)
                return df
            
            # Tentative comme crypto
            url = f"{self.config.API_CONFIG['COINGECKO']['BASE_URL']}/coins/{symbol.lower()}/market_chart"
            days = self._timeframe_to_days(timeframe)
            params = {
                'vs_currency': 'usd',
                'days': str(days),
                'interval': 'daily'
            }
            
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                df = self._prepare_crypto_data(data)
                self.cache.set(cache_key, df)
                return df
            
            raise ValueError(f"Données non disponibles pour {symbol}")
            
        except Exception as e:
            logger.error(f"Erreur récupération historique {symbol}: {str(e)}")
            raise

    def _prepare_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare les données d'actions"""
        # Vérification des colonnes requises
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Données incomplètes")
            
        # Ajout des indicateurs techniques
        df = self._add_technical_indicators(df)
        
        # Calcul des rendements
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df

    def _prepare_crypto_data(self, data: dict) -> pd.DataFrame:
        """Prépare les données crypto"""
        df = pd.DataFrame(data['prices'], columns=['Date', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        
        # Ajout des volumes si disponibles
        if 'total_volumes' in data:
            df['Volume'] = [x[1] for x in data['total_volumes']]
        
        # Calcul OHLC approximatif si non disponible
        if 'Open' not in df.columns:
            df['Open'] = df['Close'].shift(1)
            df['High'] = df['Close'].rolling(window=2).max()
            df['Low'] = df['Close'].rolling(window=2).min()
        
        # Ajout des indicateurs techniques
        df = self._add_technical_indicators(df)
        
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs techniques"""
        try:
            price_col = 'Close'
            
            # RSI
            df['RSI'] = ta.momentum.RSIIndicator(df[price_col]).rsi()
            
            # Moyennes mobiles
            for period in self.config.TECHNICAL_INDICATORS['MA']['periods']:
                df[f'MA{period}'] = ta.trend.sma_indicator(df[price_col], window=period)
            
            # MACD
            macd = ta.trend.MACD(
                df[price_col],
                window_fast=self.config.TECHNICAL_INDICATORS['MACD']['fast'],
                window_slow=self.config.TECHNICAL_INDICATORS['MACD']['slow'],
                window_sign=self.config.TECHNICAL_INDICATORS['MACD']['signal']
            )
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            # Bandes de Bollinger
            bollinger = ta.volatility.BollingerBands(
                df[price_col],
                window=self.config.TECHNICAL_INDICATORS['BB']['period'],
                window_dev=self.config.TECHNICAL_INDICATORS['BB']['std_dev']
            )
            df['BB_High'] = bollinger.bollinger_hband()
            df['BB_Mid'] = bollinger.bollinger_mavg()
            df['BB_Low'] = bollinger.bollinger_lband()
            
            return df
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs: {str(e)}")
            return df

    def _timeframe_to_days(self, timeframe: str) -> int:
        """Convertit un timeframe en nombre de jours"""
        mapping = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "max": 3650
        }
        return mapping.get(timeframe, 365)

    def get_market_summary(self, symbol: str) -> Dict[str, Any]:
        """Génère un résumé des métriques de marché"""
        try:
            info = self.get_asset_info(symbol)
            data = self.get_historical_data(symbol)
            
            return {
                'asset_info': info.to_dict(),
                'technical': {
                    'rsi': data['RSI'].iloc[-1],
                    'macd': data['MACD'].iloc[-1],
                    'macd_signal': data['MACD_Signal'].iloc[-1],
                    'volatility': data['Volatility'].iloc[-1] if 'Volatility' in data else None
                },
                'trend': self._analyze_trend(data),
                'support_resistance': self._find_support_resistance(data)
            }
        except Exception as e:
            logger.error(f"Erreur génération résumé {symbol}: {str(e)}")
            raise

    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse la tendance d'un actif"""
        try:
            price = df['Close'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            ma50 = df['MA50'].iloc[-1]
            ma200 = df['MA200'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            
            # Calcul de la force de la tendance
            trend_strength = sum([
                price > ma20,
                price > ma50,
                price > ma200
            ]) / 3.0
            
            # Détermination de la tendance
            if trend_strength > 0.7:
                trend = "haussière forte"
            elif trend_strength > 0.3:
                trend = "haussière"
            elif trend_strength < 0.3:
                trend = "baissière"
            else:
                trend = "neutre"
            
            return {
                'direction': trend,
                'strength': trend_strength,
                'momentum': {
                    'rsi': rsi,
                    'rsi_signal': 'suracheté' if rsi > 70 else 'survendu' if rsi < 30 else 'neutre',
                    'price_to_ma20': (price / ma20 - 1) * 100 if ma20 else 0,
                    'price_to_ma50': (price / ma50 - 1) * 100 if ma50 else 0
                }
            }
        except Exception as e:
            logger.error(f"Erreur analyse tendance: {str(e)}")
            return {
                'direction': 'indéterminée',
                'strength': 0,
                'momentum': {
                        'rsi': rsi,
                        'rsi_signal': 'suracheté' if rsi > 70 else 'survendu' if rsi < 30 else 'neutre',
                        'price_to_ma20': (price / ma20 - 1) * 100 if ma20 else 0,
                        'price_to_ma50': (price / ma50 - 1) * 100 if ma50 else 0
                    }
                }
        except Exception as e:
                logger.error(f"Erreur analyse tendance: {str(e)}")
                return {
                    'direction': 'indéterminée',
                    'strength': 0,
                    'momentum': {'rsi': 0, 'rsi_signal': 'inconnu', 'price_to_ma20': 0, 'price_to_ma50': 0}
                }

    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Identifie les niveaux de support et résistance"""
        try:
            window = 20
            price = df['Close'].iloc[-1]
            
            # Identification des pivots
            highs = df['High'].rolling(window=window, center=True).max()
            lows = df['Low'].rolling(window=window, center=True).min()
            
            # Filtrage des niveaux significatifs
            resistance_levels = sorted([
                level for level in set(highs.dropna())
                if level > price
            ])[:3]  # Top 3 niveaux de résistance
            
            support_levels = sorted([
                level for level in set(lows.dropna())
                if level < price
            ], reverse=True)[:3]  # Top 3 niveaux de support
            
            return {
                'support': support_levels,
                'resistance': resistance_levels,
                'current_price': price
            }
        except Exception as e:
            logger.error(f"Erreur calcul support/résistance: {str(e)}")
            return {'support': [], 'resistance': [], 'current_price': 0}

    def get_risk_metrics(self, symbol: str, timeframe: str = "1y") -> Dict[str, float]:
        """Calcule les métriques de risque pour un actif"""
        try:
            df = self.get_historical_data(symbol, timeframe)
            returns = df['Close'].pct_change().dropna()
            
            # Calcul des métriques de risque
            volatility = returns.std() * np.sqrt(252)
            sharpe = (returns.mean() * 252) / volatility if volatility != 0 else 0
            
            # Calcul du drawdown maximum
            rolling_max = df['Close'].expanding().max()
            drawdowns = (df['Close'] - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            return {
                'volatility_annualized': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            }
        except Exception as e:
            logger.error(f"Erreur calcul métriques risque {symbol}: {str(e)}")
            raise

    def compare_assets(self, symbols: List[str], timeframe: str = "1y") -> Dict[str, Any]:
        """Compare plusieurs actifs"""
        try:
            results = {}
            base_date = None
            
            for symbol in symbols:
                try:
                    # Récupération des données
                    df = self.get_historical_data(symbol, timeframe)
                    info = self.get_asset_info(symbol)
                    risk = self.get_risk_metrics(symbol, timeframe)
                    
                    # Normalisation des rendements
                    returns = df['Close'].pct_change()
                    cumul_returns = (1 + returns).cumprod()
                    
                    if base_date is None:
                        base_date = df['Date'].iloc[0]
                    
                    results[symbol] = {
                        'info': info.to_dict(),
                        'performance': {
                            'total_return': (cumul_returns.iloc[-1] - 1) * 100,
                            'volatility': risk['volatility_annualized'],
                            'sharpe': risk['sharpe_ratio'],
                            'max_drawdown': risk['max_drawdown']
                        },
                        'current': {
                            'price': info.current_price,
                            'change_24h': info.change_24h
                        }
                    }
                except Exception as e:
                    logger.warning(f"Erreur comparaison {symbol}: {str(e)}")
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Erreur comparaison globale: {str(e)}")
            raise

    def main():
        """Fonction principale de test"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        try:
            mdm = MarketDataManager()
            
            while True:
                print("\n=== Assistant Financier - Menu de Test ===")
                print("1. Tester une crypto")
                print("2. Tester une action")
                print("3. Comparer plusieurs actifs")
                print("4. Analyse complète d'un actif")
                print("5. Quitter")
                
                try:
                    choix = input("\nVotre choix (1-5): ").strip()
                    
                    if choix == "1":
                        symbol = input("Entrez le symbole de la crypto (ex: BTC): ").strip().upper()
                        try:
                            info = mdm.get_asset_info(symbol)
                            print(f"\nInfo {symbol}:")
                            print(f"Nom: {info.name}")
                            print(f"Prix: ${info.current_price:,.2f}")
                            print(f"Cap. marché: ${info.market_cap:,.0f}")
                            print(f"Volume 24h: ${info.volume_24h:,.0f}")
                            print(f"Variation 24h: {info.change_24h:.1f}%")
                            
                            data = mdm.get_historical_data(symbol)
                            print(f"\nDonnées historiques récupérées: {len(data)} jours")
                            
                        except Exception as e:
                            print(f"Erreur lors de l'analyse de {symbol}: {str(e)}")
                    
                    elif choix == "2":
                        symbol = input("Entrez le symbole de l'action (ex: AAPL): ").strip().upper()
                        try:
                            info = mdm.get_asset_info(symbol)
                            print(f"\nInfo {symbol}:")
                            print(f"Nom: {info.name}")
                            print(f"Prix: ${info.current_price:.2f}")
                            print(f"Cap. marché: ${info.market_cap:,.0f}")
                            
                            data = mdm.get_historical_data(symbol)
                            print(f"\nDonnées historiques récupérées: {len(data)} jours")
                            print(f"RSI actuel: {data['RSI'].iloc[-1]:.1f}")
                            
                        except Exception as e:
                            print(f"Erreur lors de l'analyse de {symbol}: {str(e)}")
                    
                    elif choix == "3":
                        symbols = input("Entrez les symboles séparés par des espaces: ").strip().upper().split()
                        try:
                            results = mdm.compare_assets(symbols)
                            for symbol, data in results.items():
                                print(f"\n{symbol}:")
                                print(f"Prix: ${data['current']['price']:,.2f}")
                                print(f"Variation 24h: {data['current']['change_24h']:.1f}%")
                                print(f"Performance: {data['performance']['total_return']:.1f}%")
                                
                        except Exception as e:
                            print(f"Erreur lors de la comparaison: {str(e)}")
                    
                    elif choix == "4":
                        symbol = input("Entrez un symbole: ").strip().upper()
                        try:
                            summary = mdm.get_market_summary(symbol)
                            print("\nRésumé du marché:")
                            print(f"Prix actuel: ${summary['asset_info']['current_price']:,.2f}")
                            print(f"RSI: {summary['technical']['rsi']:.1f}")
                            print(f"Tendance: {summary['trend']['direction']}")
                            print(f"Force: {summary['trend']['strength']:.1%}")
                            
                        except Exception as e:
                            print(f"Erreur lors de l'analyse: {str(e)}")
                    
                    elif choix == "5":
                        print("\nAu revoir!")
                        break
                    
                    else:
                        print("Choix invalide. Veuillez entrer un nombre entre 1 et 5.")
                    
                except Exception as e:
                    print(f"Erreur dans le menu: {str(e)}")
                
                input("\nAppuyez sur Entrée pour continuer...")
        
        except Exception as e:
            logger.error(f"Erreur critique: {str(e)}")
            print("Une erreur critique est survenue. Consultez les logs pour plus de détails.")

# Point d'entrée du programme
if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        MarketDataManager().main()
    except Exception as e:
        print(f"Erreur au démarrage: {e}")