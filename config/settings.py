# config/settings.py
from pathlib import Path
import yaml
import requests
import time
from typing import Dict, Any
from datetime import datetime, timedelta
import logging

class Config:
    def __init__(self):
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    # Chemins du projet
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MARKET_DATA_DIR = DATA_DIR / "market_data"
    KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
    MODELS_DIR = DATA_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"

    # Configuration cache avec TTL (Time To Live)
    CACHE_CONFIG = {
        "EXPIRATION": 3600,  # 1 heure
        "MAX_SIZE": 1000,    # Nombre max d'entrées
        "REFRESH_THRESHOLD": 0.8,  # Rafraîchir quand 80% expiré
        "BACKENDS": {
            "MEMORY": True,
            "DISK": True
        }
    }

    # Configuration des APIs avec gestion des quotas
    API_CONFIG = {
        "YAHOO_FINANCE": {
            "BASE_URL": "https://query2.finance.yahoo.com/v8/finance/chart/",
            "RATE_LIMIT": 2000,  # Requêtes/heure
            "TIMEOUT": 10,       # Timeout en secondes
            "RETRY_COUNT": 3,    # Nombre de tentatives
            "RETRY_DELAY": 5     # Délai entre tentatives
        },
        "COINGECKO": {
            "BASE_URL": "https://api.coingecko.com/api/v3",
            "RATE_LIMIT": 50,    # Requêtes/minute
            "TIMEOUT": 10,
            "RETRY_COUNT": 3,
            "RETRY_DELAY": 5
        }
    }

    # Configuration Prophet étendue
    PROPHET_CONFIG = {
        "DEFAULT": {
            "changepoint_prior_scale": 0.05,
            "seasonality_mode": "multiplicative",
            "interval_width": 0.95,
            "daily_seasonality": True
        },
        "CRYPTO": {
            "changepoint_prior_scale": 0.1,
            "daily_seasonality": True,
            "changepoint_range": 0.95,  # Plus sensible aux changements récents
            "seasonality_prior_scale": 10.0
        },
        "STOCKS": {
            "weekly_seasonality": True,
            "yearly_seasonality": True,
            "holidays_prior_scale": 10.0,
            "seasonality_prior_scale": 10.0
        }
    }

    # Configuration des timeframes et intervalles
    TIMEFRAMES = {
        "1d": {"interval": "5m", "limit": 100},
        "5d": {"interval": "15m", "limit": 200},
        "1mo": {"interval": "1h", "limit": 500},
        "3mo": {"interval": "1d", "limit": 1000},
        "6mo": {"interval": "1d", "limit": 1000},
        "1y": {"interval": "1d", "limit": 1000}
    }

    # Configuration des indicateurs techniques
    TECHNICAL_INDICATORS = {
        "RSI": {"period": 14},
        "MA": {"periods": [20, 50, 200]},
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "BB": {"period": 20, "std_dev": 2}
    }

    @classmethod
    def validate_config(cls) -> None:
        """Valide la configuration au démarrage et crée les répertoires nécessaires"""
        required_dirs = [
            cls.DATA_DIR, 
            cls.MARKET_DATA_DIR, 
            cls.KNOWLEDGE_BASE_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def test_api_connection(cls, api_name: str) -> bool:
        """Teste la connexion à une API spécifique avec retry et backoff"""
        api_config = cls.API_CONFIG.get(api_name)
        if not api_config:
            raise ValueError(f"Configuration non trouvée pour l'API : {api_name}")

        retry_delay = api_config["RETRY_DELAY"]
        
        for attempt in range(api_config["RETRY_COUNT"]):
            try:
                if api_name == "YAHOO_FINANCE":
                    url = f"{api_config['BASE_URL']}AAPL"
                    headers = {'User-Agent': 'Mozilla/5.0'}
                else:  # COINGECKO
                    url = f"{api_config['BASE_URL']}/ping"
                    headers = {}

                response = requests.get(
                    url, 
                    headers=headers,
                    timeout=api_config["TIMEOUT"]
                )
                
                if response.status_code == 200:
                    return True
                elif response.status_code == 429 and attempt < api_config["RETRY_COUNT"] - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Backoff exponentiel
                else:
                    return False
                    
            except Exception as e:
                if attempt == api_config["RETRY_COUNT"] - 1:
                    return False
                time.sleep(retry_delay)
                retry_delay *= 2
        
        return False

if __name__ == "__main__":
    config = Config()
    config.validate_config()
    
    # Test des connexions API
    apis_status = {}
    for api_name in config.API_CONFIG.keys():
        apis_status[api_name] = "✓" if config.test_api_connection(api_name) else "⚠️"
        
    print("\n=== Configuration de l'Assistant Financier ===")
    print(f"📁 Répertoire de base: {config.BASE_DIR}")
    print(f"📊 Yahoo Finance API: {apis_status['YAHOO_FINANCE']}")
    print(f"🪙 CoinGecko API: {apis_status['COINGECKO']}")