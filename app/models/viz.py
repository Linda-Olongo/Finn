import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fbprophet import ProphetModel
from data.preprocessing import DataPreprocessor

class FinanceVisualizer:
    def __init__(self):
        self.prophet_model = ProphetModel()
        self.preprocessor = DataPreprocessor()

    def display_header(self, symbol: str, asset_type: str = 'crypto'):
        """Affiche les informations d'en-tête dans le terminal"""
        try:
            if asset_type == 'crypto':
                current_data = self.preprocessor.collector.get_crypto_current(symbol)
                print("\n" + "="*50)
                print(f"📊 {current_data['name']} ({current_data['id'].upper()})")
                print("="*50)
                print(f"Prix: ${current_data['price']:.2f} ({current_data['change_24h']:+.2f}%)")
                print(f"Volume 24h: ${current_data['volume']:,.2f}")
                print(f"Market Cap: ${current_data['market_cap']:,.2f}")
            else:
                current_data = self.preprocessor.collector.get_stock_current(symbol)
                print("\n" + "="*50)
                print(f"📈 {current_data['name']} ({current_data['symbol']})")
                print("="*50)
                print(f"Prix: ${current_data['price']:.2f} ({current_data['change']:+.2f}%)")
                print(f"Volume: {current_data['volume']:,}")
                print(f"P/E Ratio: {current_data['pe_ratio']}")
            print("="*50 + "\n")
            return current_data
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            return None

    def create_interactive_plot(self, symbol: str, asset_type: str = 'crypto', periods: int = None):
        """Crée un graphique interactif avec Plotly"""
        try:
            # Chargement et préparation des données
            df = self.prophet_model.load_and_prepare_data(symbol, asset_type)
            
            # Limitation à 3 mois d'historique pour la lisibilité
            three_months_ago = datetime.now() - timedelta(days=90)
            df = df[df['ds'] >= three_months_ago]
            
            # Construction et prédictions
            self.prophet_model.build_model()
            forecast = self.prophet_model.make_predictions(periods=periods)
            
            # Récupération des données actuelles
            current_data = self.display_header(symbol, asset_type)
            if not current_data:
                return

            # Création du graphique Plotly
            if asset_type == 'stock':
                title = f"{current_data['name']} ({symbol.upper()})<br><sup>Prix: ${df['y'].iloc[-1]:.2f} ({current_data['change']:+.2f}%) | Volume: {current_data['volume']:,} | Market Cap: ${current_data['market_cap']:,.2f}B</sup>"
            else:
                title = f"{current_data['name']} ({symbol.upper()})<br><sup>Prix: ${df['y'].iloc[-1]:.2f} ({current_data['change_24h']:+.2f}%) | Volume 24h: ${current_data['volume']:,.2f} | Market Cap: ${current_data['market_cap']:,.2f}</sup>"

            fig = make_subplots(rows=2, cols=1, 
                              row_heights=[0.7, 0.3],
                              subplot_titles=('', 'Volume'),
                              vertical_spacing=0.12)

            # Date future pour séparer historique et prédictions
            future_date = datetime.now()
            
            # Graphique principal - Historique
            fig.add_trace(
                go.Scatter(
                    x=df['ds'], 
                    y=df['y'], 
                    name='Historique',
                    line=dict(color='blue', width=2),
                    hovertemplate='%{x}<br>Prix: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
            
            # Prédictions et intervalles de confiance (seulement pour le futur)
            mask_future = forecast['ds'] > future_date
            fig.add_trace(
                go.Scatter(
                    x=forecast.loc[mask_future, 'ds'],
                    y=forecast.loc[mask_future, 'yhat'],
                    name='Prédiction',
                    line=dict(color='orange', width=2, dash='dot'),
                    hovertemplate='%{x}<br>Prédiction: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
            
            fig.add_trace(
                go.Scatter(
                    x=forecast.loc[mask_future, 'ds'],
                    y=forecast.loc[mask_future, 'yhat_upper'],
                    name='Intervalle de confiance',
                    line=dict(width=0),
                    showlegend=False,
                    hovertemplate='%{x}<br>Max: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
            
            fig.add_trace(
                go.Scatter(
                    x=forecast.loc[mask_future, 'ds'],
                    y=forecast.loc[mask_future, 'yhat_lower'],
                    fill='tonexty',
                    fillcolor='rgba(255,165,0,0.2)',
                    line=dict(width=0),
                    showlegend=False,
                    hovertemplate='%{x}<br>Min: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)

            # Graphique du volume si disponible
            if 'volume' in df.columns:
                fig.add_trace(
                    go.Bar(x=df['ds'], y=df['volume'],
                          name='Volume', marker_color='lightblue'),
                    row=2, col=1)

            # Mise en page
            fig.update_layout(
                title=title,
                title_x=0.5,
                height=800,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            # Mise à jour des axes
            fig.update_xaxes(
                title="Date",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
                dtick="M1",
                tickformat="%b %Y"
            )
            
            fig.update_yaxes(
                title="Prix ($)",
                row=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            )
            
            fig.update_yaxes(
                title="Volume",
                row=2,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            )

            # Afficher le graphique
            fig.show()

            # Afficher les métriques
            metrics = self.prophet_model.evaluate_model()
            print("\nMétriques de performance:")
            print(f"RMSE: {metrics['RMSE']:.2f}")
            print(f"MAE: {metrics['MAE']:.2f}")
            print(f"MAPE: {metrics['MAPE']:.2f}%")
            print(f"R²: {metrics['R2']:.2f}")

        except Exception as e:
            print(f"\n❌ Erreur lors de la création du graphique: {e}")

    def compare_assets(self, symbols: list, asset_type: str = 'crypto'):
        """Compare plusieurs actifs sur le même graphique"""
        try:
            # Création du graphique de comparaison avec sous-titre
            fig = go.Figure()
            title_parts = []
            
            # Ajout de chaque actif
            for symbol in symbols:
                # Récupération des données actuelles
                if asset_type == 'crypto':
                    current_data = self.preprocessor.collector.get_crypto_current(symbol)
                    title_part = f"{current_data['name']} ({symbol.upper()}): ${current_data['price']:.2f} ({current_data['change_24h']:+.2f}%)"
                else:
                    current_data = self.preprocessor.collector.get_stock_current(symbol)
                    title_part = f"{current_data['name']} ({symbol.upper()}): ${current_data['price']:.2f} ({current_data['change']:+.2f}%)"
                title_parts.append(title_part)
                
                # Préparation des données
                df = self.prophet_model.load_and_prepare_data(symbol, asset_type)
                
                # Limitation à 3 mois
                three_months_ago = datetime.now() - timedelta(days=90)
                df = df[df['ds'] >= three_months_ago]
                
                # Ajout au graphique avec informations détaillées
                hover_template = (
                    "%{x}<br>" +
                    "Prix: $%{y:.2f}<br>" +
                    f"Volume: {current_data['volume']:,}<br>" +
                    f"Market Cap: ${current_data['market_cap']:,.2f}"
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['ds'],
                        y=df['y'],
                        name=symbol.upper(),
                        mode='lines',
                        hovertemplate=hover_template + '<extra></extra>'
                    )
                )

            # Mise en page
            fig.update_layout(
                title="Comparaison des actifs<br><sup>" + " | ".join(title_parts) + "</sup>",
                height=800,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Mise à jour des axes
            fig.update_xaxes(
                title="Date",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
                dtick="M1",
                tickformat="%b %Y"
            )
            
            fig.update_yaxes(
                title="Prix ($)",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            )

            # Afficher le graphique
            fig.show()

        except Exception as e:
            print(f"\n❌ Erreur lors de la comparaison: {e}")

def main():
    visualizer = FinanceVisualizer()
    
    while True:
        print("\n=== Visualisation des Données Financières ===")
        print("1. Analyser une action")
        print("2. Analyser une crypto")
        print("3. Comparer plusieurs actifs")
        print("4. Quitter")
        
        choice = input("\nVotre choix (1-4): ")
        
        if choice == "4":
            print("\nAu revoir!")
            break
            
        if choice == "3":
            print("\n=== Mode Comparaison ===")
            asset_type = input("Type d'actif (stock/crypto): ").lower()
            if asset_type not in ['stock', 'crypto']:
                print("Type d'actif invalide!")
                continue
                
            symbols = input("Entrez les symboles (2-5, séparés par des virgules): ").split(',')
            symbols = [s.strip() for s in symbols]
            
            if 2 <= len(symbols) <= 5:
                visualizer.compare_assets(symbols, asset_type)
            else:
                print("Veuillez entrer entre 2 et 5 symboles!")
                
        elif choice in ["1", "2"]:
            asset_type = "stock" if choice == "1" else "crypto"
            symbol = input(f"\nEntrez le symbole {'de l`action' if asset_type == 'stock' else 'de la crypto'}: ")
            
            # Demande de la période de prédiction
            while True:
                try:
                    periods = int(input("\nNombre de jours de prédiction (max 365): "))
                    if 1 <= periods <= 365:
                        break
                    else:
                        print("La période doit être comprise entre 1 et 365 jours!")
                except ValueError:
                    print("Veuillez entrer un nombre valide!")
            
            visualizer.display_header(symbol, asset_type)
            visualizer.create_interactive_plot(symbol, asset_type, periods)
        
        else:
            print("Choix invalide!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramme interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\nErreur inattendue: {e}")