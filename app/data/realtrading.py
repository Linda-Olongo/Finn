# app/data/realtrading.py
import requests
import random
import datetime
import time
import plotly.graph_objects as go

class RealTimeTrading:
    def __init__(self):
        # Liste d'actions et de cryptos populaires
        self.stocks = [
            {"symbol": "BTC-USD", "name": "Bitcoin", "type": "crypto"},
            {"symbol": "ETH-USD", "name": "Ethereum", "type": "crypto"},
            {"symbol": "SOL-USD", "name": "Solana", "type": "crypto"},
            {"symbol": "DOGE-USD", "name": "Dogecoin", "type": "crypto"},
            {"symbol": "ADA-USD", "name": "Cardano", "type": "crypto"},
            {"symbol": "XRP-USD", "name": "Ripple", "type": "crypto"},
            {"symbol": "AVAX-USD", "name": "Avalanche", "type": "crypto"},
            {"symbol": "DOT-USD", "name": "Polkadot", "type": "crypto"}
        ]
        
        # Définir la seed pour la reproductibilité de la sélection
        random.seed(2)
        
        # Date du jour pour renouveler la sélection chaque jour
        self.today = datetime.datetime.now().strftime("%Y-%m-%d")
        self.selected_assets = None  # Sera initialisé à la demande
    
    def _select_daily_assets(self):
        """Sélectionne 5 actifs aléatoires pour la journée avec des données disponibles"""
        # Utiliser la date comme seed pour changer chaque jour
        day_seed = sum(ord(c) for c in self.today)
        random.seed(2 + day_seed)
        
        # Mélanger la liste et récupérer les données pour trouver 5 actifs avec données
        shuffled_assets = random.sample(self.stocks, len(self.stocks))
        
        valid_assets = []
        for asset in shuffled_assets:
            if len(valid_assets) >= 5:
                break
                
            asset_data = self._fetch_asset_data(asset["symbol"])
            if asset_data and asset_data.get('chart_data'):
                asset.update(asset_data)
                valid_assets.append(asset)
        
        return valid_assets
    
    def _fetch_asset_data(self, symbol):
        """Récupère les données réelles pour un actif depuis Yahoo Finance"""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=5m&range=1d"
            
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            # Extraire les données pertinentes
            result = data.get('chart', {}).get('result', [])
            if not result:
                return None
            
            # Obtenir la série temporelle
            timestamps = result[0].get('timestamp', [])
            
            # Obtenir les données de prix
            quote = result[0].get('indicators', {}).get('quote', [{}])[0]
            close_prices = quote.get('close', [])
            
            # S'assurer que nous avons des données
            if not timestamps or not close_prices:
                return None
            
            # Obtenir le prix actuel
            current_price = next((price for price in reversed(close_prices) if price is not None), None)
            
            if current_price is None:
                return None
            
            # Créer des points de données pour le graphique
            chart_data = []
            for i, timestamp in enumerate(timestamps):
                if i < len(close_prices) and close_prices[i] is not None:
                    dt = datetime.datetime.fromtimestamp(timestamp)
                    chart_data.append({
                        "timestamp": dt,
                        "price": round(close_prices[i], 2)
                    })
            
            # Trier par ordre chronologique
            chart_data = sorted(chart_data, key=lambda x: x['timestamp'])
            
            # Calculer la variation en pourcentage
            if not chart_data:
                return None
                
            first_price = chart_data[0]['price']
            price_change = current_price - first_price
            percent_change = (price_change / first_price) * 100
            
            return {
                "current_price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "percent_change": round(percent_change, 2),
                "chart_data": chart_data
            }
            
        except Exception as e:
            print(f"Erreur lors de la récupération des données pour {symbol}: {e}")
            return None
    
    def get_current_assets(self):
        """Renvoie les actifs sélectionnés pour aujourd'hui"""
        # Vérifier si nous sommes toujours le même jour ou si c'est la première demande
        current_day = datetime.datetime.now().strftime("%Y-%m-%d")
        if self.selected_assets is None or current_day != self.today:
            self.today = current_day
            self.selected_assets = self._select_daily_assets()
        
        return self.selected_assets
    
    def update_asset_data(self, symbol):
        """Met à jour les données d'un actif spécifique"""
        if self.selected_assets is None:
            self.get_current_assets()
            
        asset_data = self._fetch_asset_data(symbol)
        if not asset_data:
            return None
        
        # Trouver l'actif dans la liste et mettre à jour ses données
        for asset in self.selected_assets:
            if asset["symbol"] == symbol:
                asset.update(asset_data)
                return asset
        
        return None
    
    def get_updated_prices(self):
        """Met à jour et renvoie les données actuelles de tous les actifs"""
        if self.selected_assets is None:
            self.get_current_assets()
            
        for asset in self.selected_assets:
            self.update_asset_data(asset["symbol"])
        
        return self.selected_assets
    
    def get_price_chart_data(self, symbol):
        """Retourne les données du graphique dans un format simple pour l'affichage dans Flask"""
        # Récupérer ou mettre à jour l'actif
        asset = None
        for a in self.get_current_assets():
            if a["symbol"] == symbol:
                asset = a
                break
        
        if not asset:
            asset = self.update_asset_data(symbol)
        
        if not asset or "chart_data" not in asset:
            return None
            
        # Préparer les données pour l'intégration dans Flask
        data = {
            "symbol": asset["symbol"],
            "name": asset["name"],
            "current_price": asset["current_price"],
            "price_change": asset["price_change"],
            "percent_change": asset["percent_change"],
            "labels": [point["timestamp"].strftime('%H:%M') for point in asset["chart_data"]],
            "prices": [point["price"] for point in asset["chart_data"]],
            "color": "green" if asset["price_change"] >= 0 else "red"
        }
        
        return data


# Fonction pour visualiser un actif avec plotly (pour tests uniquement)
def visualize_asset(asset_data):
    if not asset_data:
        print("Données d'actif non disponibles.")
        return
    
    # Créer une figure Plotly avec un style épuré
    fig = go.Figure()
    
    # Déterminer la couleur basée sur le changement de prix
    color = "rgb(0, 150, 0)" if asset_data["price_change"] >= 0 else "rgb(230, 0, 0)"
    
    # Ajouter la ligne de prix
    fig.add_trace(go.Scatter(
        x=[point["timestamp"] for point in asset_data["chart_data"]],
        y=[point["price"] for point in asset_data["chart_data"]],
        mode='lines',
        line=dict(color=color, width=2),
        hovertemplate='$%{y:.2f}<extra></extra>'
    ))
    
    # Formatter les axes de manière minimaliste
    fig.update_xaxes(
        tickformat="%H:%M",
        gridcolor='lightgray',
        showgrid=True,
        title=None  # Suppression du titre "Heure"
    )
    
    fig.update_yaxes(
        tickprefix="$",
        gridcolor='lightgray',
        showgrid=True,
        title=None  # Suppression du titre "Prix"
    )
    
    # Mise en forme minimaliste
    fig.update_layout(
        title=None,  # Suppression du titre en haut
        plot_bgcolor='white',
        height=300,
        margin=dict(l=30, r=30, t=10, b=30),  # Marge supérieure réduite
        hovermode="x unified"
    )
    
    # Ajouter une annotation pour le dernier prix
    last_point = asset_data["chart_data"][-1]
    fig.add_annotation(
        x=last_point["timestamp"],
        y=last_point["price"],
        text=f"${last_point['price']}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=color,
        arrowsize=1,
        arrowwidth=2,
        ax=40,
        ay=-40
    )
    
    # Montrer la figure
    fig.show()


# Pour tester directement ce module
if __name__ == "__main__":
    print("Test du module RealTimeTrading")
    trader = RealTimeTrading()
    
    # Récupérer les actifs disponibles
    assets = trader.get_current_assets()
    
    if not assets:
        print("Aucun actif avec données disponibles n'a été trouvé.")
        exit()
    
    print(f"\n{len(assets)} actifs avec données disponibles :")
    
    for i, asset in enumerate(assets, 1):
        print(f"\n{i}. {asset['name']} ({asset['symbol']})")
        print(f"   Prix actuel: ${asset.get('current_price', 'N/A')}")
        
        if 'price_change' in asset and 'percent_change' in asset:
            change_symbol = "+" if asset['price_change'] >= 0 else ""
            print(f"   Variation: {change_symbol}${asset['price_change']} ({change_symbol}{asset['percent_change']}%)")
        
        chart_data = asset.get('chart_data', [])
        if chart_data:
            print(f"   Données disponibles: {len(chart_data)} points")
            print(f"   Première mesure: {chart_data[0]['timestamp'].strftime('%H:%M')} - ${chart_data[0]['price']}")
            print(f"   Dernière mesure: {chart_data[-1]['timestamp'].strftime('%H:%M')} - ${chart_data[-1]['price']}")
    
    # Choisir un actif à visualiser
    selected_index = 0
    if len(assets) > 1:
        try:
            selected_index = int(input(f"\nChoisissez un actif à visualiser (1-{len(assets)}): ")) - 1
            if selected_index < 0 or selected_index >= len(assets):
                selected_index = 0
        except:
            selected_index = 0
    
    selected_asset = assets[selected_index]
    print(f"\nVisualisation de {selected_asset['name']} ({selected_asset['symbol']})")
    
    # Visualiser l'actif sélectionné
    visualize_asset(selected_asset)
    
    # Démonstration de mise à jour automatique
    print("\nDémonstration de mise à jour automatique (Ctrl+C pour arrêter)")
    try:
        for i in range(3):  # Faire 3 mises à jour
            time.sleep(10)  # Attendre 10 secondes
            updated_asset = trader.update_asset_data(selected_asset['symbol'])
            print(f"Mise à jour #{i+1}: {updated_asset['name']} - ${updated_asset['current_price']}")
            visualize_asset(updated_asset)
    except KeyboardInterrupt:
        print("\nMise à jour automatique arrêtée.")