import sys
import os
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fbprophet import ProphetModel
from data.preprocessing import DataPreprocessor


class PredictionVisualizer:
    def __init__(self):
        self.prophet_model = ProphetModel()
        self.preprocessor = DataPreprocessor()
        
    def _clean_price(self, price):
        """Méthode utilitaire pour nettoyer la valeur du prix"""
        if isinstance(price, (int, float)):
            return float(price)
            
        if isinstance(price, str):
            # Extraire uniquement la valeur numérique
            numeric_match = re.search(r'\d+\.?\d*', price)
            if numeric_match:
                return float(numeric_match.group(0))
            else:
                # Fallback: supprimer les caractères non numériques
                cleaned = re.sub(r'[^\d.]', '', price)
                try:
                    return float(cleaned) if cleaned else 0.0
                except ValueError:
                    return 0.0
        return 0.0

    def display_header(self, symbol: str, asset_type: str = 'crypto') -> dict:
        """Affiche les informations d'en-tête dans le terminal et retourne les données actuelles"""
        try:
            if asset_type == 'crypto':
                current_data = self.preprocessor.collector.get_crypto_current(symbol)
                print("\n" + "="*50)
                print(f"📊 {current_data['name']} ({current_data['id'].upper()})")
                print("="*50)
                
                # Nettoyer le prix
                current_data['price'] = self._clean_price(current_data['price'])
                
                print(f"Prix: ${current_data['price']:.2f} ({current_data['change_24h']:+.2f}%)")
                print(f"Volume 24h: ${current_data['volume']:,.2f}")
                print(f"Market Cap: ${current_data['market_cap']:,.2f}")
                # Ajout des métriques spécifiques aux cryptos
                if 'diluted_valuation' in current_data and current_data['diluted_valuation']:
                    print(f"Fully Diluted Valuation: ${current_data['diluted_valuation']:,.2f}")
                if 'circulating_supply' in current_data and current_data['circulating_supply']:
                    print(f"Circulating Supply: {current_data['circulating_supply']:,}")
            else:
                current_data = self.preprocessor.collector.get_stock_current(symbol)
                print("\n" + "="*50)
                print(f"📈 {current_data['name']} ({current_data['symbol']})")
                print("="*50)
                
                # Nettoyer le prix
                current_data['price'] = self._clean_price(current_data['price'])
                
                print(f"Prix: ${current_data['price']:.2f} ({current_data['change']:+.2f}%)")
                print(f"Volume: {current_data['volume']:,}")
                # Métriques supplémentaires si disponibles
                if 'high' in current_data and current_data['high'] is not None:
                    print(f"Haut du jour: ${current_data['high']:.2f}")
                if 'low' in current_data and current_data['low'] is not None:
                    print(f"Bas du jour: ${current_data['low']:.2f}")
                if 'pe_ratio' in current_data and current_data['pe_ratio'] is not None:
                    print(f"P/E Ratio: {current_data['pe_ratio']:.2f}")
            print("="*50 + "\n")
            return current_data
        except Exception as e:
            print(f"\n❌ Erreur lors de la récupération des données: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_single_asset_plot(self, symbol: str, asset_type: str = 'crypto', periods: int = 30):
        """Crée un graphique interactif pour un seul actif avec historique et prédictions"""
        try:
            # Chargement des données
            self.prophet_model.load_data(symbol, asset_type)
            self.prophet_model.build_model()
            forecast = self.prophet_model.make_predictions(periods=periods)

            # Récupérer les données historiques préparées
            df = self.prophet_model.prophet_data['df']
            
            # Récupérer les données actuelles pour l'en-tête
            current_data = self.display_header(symbol, asset_type)
            if not current_data:
                return

            # Limiter l'historique à 3 mois pour la lisibilité
            three_months_ago = datetime.now() - timedelta(days=90)
            df = df[df['ds'] >= three_months_ago]

            # Créer un graphique simple
            fig = go.Figure()

            # Date actuelle pour séparer historique et prédictions
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            # Historique (en bleu)
            fig.add_trace(
                go.Scatter(
                    x=df['ds'],
                    y=df['y'],
                    name='Historique',
                    line=dict(color='blue', width=2),
                    mode='lines',
                    hovertemplate='%{x|%d %b %Y}<br>Prix: $%{y:.2f}<extra></extra>'
                )
            )

            # Prédictions avec style zigzag
            mask_future = forecast['ds'] >= current_date
            future_forecast = forecast[mask_future].copy()
            
            # Calculer la date de la dernière donnée historique
            if not df.empty:
                last_hist_date = df['ds'].max()
                # Ajouter une ligne verticale pour marquer la séparation entre historique et prédiction
                fig.add_vline(x=last_hist_date, line_width=1, line_dash="dash", line_color="black")
                
                # Récupérer le dernier prix historique connu
                last_price = df.loc[df['ds'] == last_hist_date, 'y'].iloc[0] if not df[df['ds'] == last_hist_date].empty else None
                
                # Récupérer la première prédiction
                first_pred = future_forecast['yhat'].iloc[0] if not future_forecast.empty else None
                
                # Ajouter un pont entre les deux si les données existent
                if last_price is not None and first_pred is not None:
                    # Trouver la date de la première prédiction
                    first_pred_date = future_forecast['ds'].iloc[0]
                    
                    # Créer une connexion entre historique et prédiction
                    fig.add_trace(
                        go.Scatter(
                            x=[last_hist_date, first_pred_date],
                            y=[last_price, first_pred],
                            mode='lines',
                            line=dict(color='grey', width=1, dash='dot'),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )

            # Ajouter les prédictions futures avec style zigzag (variations simulées)
            if not future_forecast.empty:
                # Créer des variations aléatoires pour simuler un zigzag plus naturel
                import numpy as np
                np.random.seed(42)  # Pour la reproductibilité
                
                # Déterminer une amplitude raisonnable pour les variations basée sur la volatilité historique
                if len(df) > 5:
                    volatility = df['y'].pct_change().std() * 100  # Volatilité en pourcentage
                    # Limiter la volatilité à une plage raisonnable
                    volatility = max(min(volatility, 2.0), 0.2)
                else:
                    volatility = 0.5  # Valeur par défaut si peu de données
                
                print(f"Volatilité estimée: {volatility:.2f}%")
                
                # Appliquer de petites variations aux prédictions pour créer un effet zigzag
                future_forecast['yhat_zigzag'] = future_forecast['yhat'] * (1 + np.random.uniform(-volatility/100, volatility/100, size=len(future_forecast)))
                
                # Lissage pour éviter des changements trop brusques
                future_forecast['yhat_smooth'] = future_forecast['yhat_zigzag'].ewm(span=3).mean()
                
                # Assurer que la première prédiction reste alignée
                if len(future_forecast) > 0:
                    future_forecast.loc[future_forecast.index[0], 'yhat_smooth'] = future_forecast.loc[future_forecast.index[0], 'yhat']
                
                # Tracer la ligne de prédiction avec le style en pointillé
                fig.add_trace(
                    go.Scatter(
                        x=future_forecast['ds'],
                        y=future_forecast['yhat_smooth'],
                        name='Prédiction',
                        line=dict(color='grey', width=2, dash='dash'),  # Ligne en pointillé
                        mode='lines',
                        hovertemplate='%{x|%d %b %Y}<br>Prédiction: $%{y:.2f}<extra></extra>'
                    )
                )

                # Ajouter la ligne de tendance lissée comme référence (optionnel, peu visible)
                fig.add_trace(
                    go.Scatter(
                        x=future_forecast['ds'],
                        y=future_forecast['yhat'],
                        name='Tendance',
                        line=dict(color='grey', width=1, dash='dot'),
                        mode='lines',
                        opacity=0.4,
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
                
                # Intervalles de confiance
                fig.add_trace(
                    go.Scatter(
                        x=future_forecast['ds'],
                        y=future_forecast['yhat_upper'],
                        name='Intervalle de confiance',
                        line=dict(width=0),
                        showlegend=False,
                        hovertemplate='%{x|%d %b %Y}<br>Max: $%{y:.2f}<extra></extra>'
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=future_forecast['ds'],
                        y=future_forecast['yhat_lower'],
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.2)',
                        line=dict(width=0),
                        showlegend=False,
                        hovertemplate='%{x|%d %b %Y}<br>Min: $%{y:.2f}<extra></extra>'
                    )
                )

            # Définir les couleurs pour les points de prédiction min et max
            point_min_color = 'rgb(77, 200, 255)'  # Bleu clair
            point_max_color = 'rgb(255, 153, 51)'  # Orange

            # Trouver le min et max pour chaque mois future
            if not future_forecast.empty:
                # Regrouper par mois
                future_forecast['month'] = future_forecast['ds'].dt.to_period('M')
                monthly_data = {}
                
                for month, group in future_forecast.groupby('month'):
                    min_idx = group['yhat_lower'].idxmin()
                    max_idx = group['yhat_upper'].idxmax()
                    
                    min_date = group.loc[min_idx, 'ds']
                    max_date = group.loc[max_idx, 'ds']
                    min_value = group.loc[min_idx, 'yhat_lower']
                    max_value = group.loc[max_idx, 'yhat_upper']
                    
                    # Pour le dernier mois de prédiction, ajoutons les points min et max
                    if month == future_forecast['month'].iloc[-1]:
                        # Point minimum
                        fig.add_trace(
                            go.Scatter(
                                x=[min_date],
                                y=[min_value],
                                mode='markers',
                                marker=dict(color=point_min_color, size=10),
                                name='Minimum',
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )
                        
                        # Point maximum
                        fig.add_trace(
                            go.Scatter(
                                x=[max_date],
                                y=[max_value],
                                mode='markers',
                                marker=dict(color=point_max_color, size=10),
                                name='Maximum',
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )
                        
                        # Stocker pour l'affichage dans la légende
                        monthly_data[month.strftime('%b %Y')] = {
                            'min_date': min_date,
                            'max_date': max_date,
                            'min_value': min_value,
                            'max_value': max_value,
                            'avg_value': group['yhat'].mean()
                        }

            # Créer une légende en haut à gauche comme dans l'image de référence
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    line=dict(color='grey', width=2, dash='dash'),
                    name='Prédiction',
                    showlegend=True
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Historique',
                    showlegend=True
                )
            )

            # Style de l'en-tête comme dans l'image de référence (aligné à gauche)
            # Utiliser l'affichage simplifié pour éviter les problèmes avec les balises HTML
            if asset_type == 'stock':
                change = current_data.get('change', 0)
                change_color = 'green' if change >= 0 else 'red'
                change_text = f"({change:+.2f}%)"
                
                # Titre avec nom et symbole
                fig.add_annotation(
                    x=0, y=1.12,
                    xref='paper', yref='paper',
                    text=f"<b style='font-size:24px; color:blue'>{symbol.upper()}</b> {current_data['name']} - " + 
                         f"<b>${current_data['price']:.2f}</b> <span style='color:{change_color}'>{change_text}</span>",
                    showarrow=False,
                    font=dict(size=18),
                    align='left',
                    xanchor='left'
                )
                
                # Information de volume
                fig.add_annotation(
                    x=0, y=1.06,
                    xref='paper', yref='paper',
                    text=f"Volume: {current_data['volume']:,}",
                    showarrow=False,
                    font=dict(size=14),
                    align='left',
                    xanchor='left'
                )
                
            else:
                change = current_data.get('change_24h', 0)
                change_color = 'green' if change >= 0 else 'red'
                change_text = f"({change:+.2f}%)"
                
                # Titre avec nom et symbole
                fig.add_annotation(
                    x=0, y=1.12,
                    xref='paper', yref='paper',
                    text=f"<b style='font-size:24px; color:blue'>{symbol.upper()}</b> {current_data['name']} - " + 
                         f"<b>${current_data['price']:.2f}</b> <span style='color:{change_color}'>{change_text}</span>",
                    showarrow=False,
                    font=dict(size=18),
                    align='left',
                    xanchor='left'
                )
                
                # Information de volume et market cap
                fig.add_annotation(
                    x=0, y=1.06,
                    xref='paper', yref='paper',
                    text=f"Volume 24h: ${current_data['volume']:,.2f}<br>MarketCap :{current_data['market_cap']:,.2f}",
                    showarrow=False,
                    font=dict(size=14),
                    align='left',
                    xanchor='left'
                )

            # Mise en page du graphique avec plus d'espace pour les annotations
            fig.update_layout(
                height=800,
                margin=dict(t=100, b=50, l=50, r=50),
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top", 
                    y=0.99, 
                    xanchor="left", 
                    x=0.01, 
                    bgcolor='rgba(255,255,255,0.8)',
                    orientation="h"
                ),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            # Mise à jour des axes - afficher seulement mois/année sur l'axe X
            fig.update_xaxes(
                title="Date",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
                dtick="M1",  # Une date par mois
                tickformat="%b %Y"  # Format mois année sans jour
            )
            
            fig.update_yaxes(
                title="Prix ($)",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            )

            # Ajouter des informations de prédiction pour le dernier mois avec jours précis
            if monthly_data:
                last_month = list(monthly_data.keys())[-1]
                pred_data = monthly_data[last_month]
                
                # Récupérer la date exacte pour l'affichage (jour mois année)
                min_date_str = pred_data['min_date'].strftime('%d %b %Y')
                max_date_str = pred_data['max_date'].strftime('%d %b %Y')
                current_date_str = datetime.now().strftime('%d %b %Y')
                
                # Créer la boîte avec les détails de prédiction
                prediction_box = f"<b>{min_date_str.split(' ')[0]} {last_month}</b><br><br>"
                
                # Point MIN (bleu)
                prediction_box += f"<span style='color:{point_min_color}'>●</span> "
                prediction_box += f"{min_date_str}<br>"
                prediction_box += f"Min: ${pred_data['min_value']:.2f}<br><br>"
                
                # Point MAX (orange)
                prediction_box += f"<span style='color:{point_max_color}'>●</span> "
                prediction_box += f"{max_date_str}<br>"
                prediction_box += f"Max: ${pred_data['max_value']:.2f}<br><br>"
                
                # Ligne de prédiction moyenne (gris)
                prediction_box += f"<span style='color:grey'>---</span> "
                prediction_box += f"{min_date_str.split(' ')[0]} {last_month}<br>"
                prediction_box += f"Prédiction: ${pred_data['avg_value']:.2f}<br><br>"
                
                # Prix actuel (bleu)
                prediction_box += f"<span style='color:blue'>―</span> "
                prediction_box += f"{current_date_str}<br>"
                prediction_box += f"Prix: ${current_data['price']:.2f}"
                
                # Position de l'annotation près de la dernière valeur de prédiction
                fig.add_annotation(
                    x=1,
                    y=0.5,
                    xref='paper',
                    yref='paper',
                    text=prediction_box,
                    showarrow=False,
                    font=dict(size=12),
                    align='left',
                    xanchor='left',
                    yanchor='middle',
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="lightgrey",
                    borderwidth=1,
                    borderpad=4
                )
            
            # Afficher le graphique
            fig.show()

            # Afficher les métriques de performance
            metrics = self.prophet_model.evaluate_model()
            if metrics:
                print("\nMétriques de performance:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.2f}")

        except Exception as e:
            print(f"\n❌ Erreur lors de la création du graphique: {e}")
            import traceback
            traceback.print_exc()

    def compare_assets(self, symbols: list, asset_type: str = 'crypto', periods: int = 30):
        """Compare plusieurs actifs (max 5) sur le même graphique avec prédictions optionnelles"""
        if len(symbols) > 5:
            print("❌ Erreur: Maximum 5 actifs pour la comparaison.")
            return

        try:
            fig = go.Figure()
            
            # Couleurs différentes pour chaque actif
            colors = ['blue', 'orange', 'green', 'red', 'purple']
            
            # Variables pour stocker les informations d'en-tête
            header_info = []

            for idx, symbol in enumerate(symbols):
                # Récupérer les données actuelles
                try:
                    if asset_type == 'crypto':
                        current_data = self.preprocessor.collector.get_crypto_current(symbol)
                        change = current_data.get('change_24h', 0)
                        change_color = 'green' if change >= 0 else 'red'
                        
                        # Nettoyer le prix
                        current_price = self._clean_price(current_data['price'])
                        
                        symbol_info = {
                            'symbol': symbol.upper(),
                            'name': current_data['name'],
                            'price': current_price,
                            'change': change,
                            'change_color': change_color,
                            'color': colors[idx % len(colors)]
                        }
                    else:
                        current_data = self.preprocessor.collector.get_stock_current(symbol)
                        change = current_data.get('change', 0)
                        change_color = 'green' if change >= 0 else 'red'
                        
                        # Nettoyer le prix
                        current_price = self._clean_price(current_data['price'])
                        
                        symbol_info = {
                            'symbol': symbol.upper(),
                            'name': current_data['name'],
                            'price': current_price,
                            'change': change,
                            'change_color': change_color,
                            'color': colors[idx % len(colors)]
                        }
                    header_info.append(symbol_info)

                    # Charger les données historiques
                    self.prophet_model.load_data(symbol, asset_type)
                    df = self.prophet_model.prophet_data['df']

                    # Limiter à 3 mois
                    three_months_ago = datetime.now() - timedelta(days=90)
                    df = df[df['ds'] >= three_months_ago]

                    # Ajouter la trace pour chaque actif (historique)
                    fig.add_trace(
                        go.Scatter(
                            x=df['ds'],
                            y=df['y'],
                            name=f"{symbol.upper()} (Historique)",
                            line=dict(color=colors[idx % len(colors)], width=2),
                            mode='lines',
                            hovertemplate='%{x|%d %b %Y}<br>' + symbol.upper() + ': $%{y:.2f}<extra></extra>'
                        )
                    )
                    
                    # Ajouter les prédictions si demandé
                    if periods > 0:
                        # Générer les prédictions
                        self.prophet_model.build_model()
                        forecast = self.prophet_model.make_predictions(periods=periods)
                        
                        # Filtrer pour n'avoir que les prédictions futures
                        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                        future_forecast = forecast[forecast['ds'] >= current_date]
                        
                        if not future_forecast.empty:
                            # Ajouter des zigzags aux prédictions
                            import numpy as np
                            np.random.seed(42 + idx)  # Différentes graines pour différents actifs
                            
                            # Estimer la volatilité
                            if len(df) > 5:
                                volatility = df['y'].pct_change().std() * 100  # Volatilité en pourcentage
                                volatility = max(min(volatility, 2.0), 0.2)
                            else:
                                volatility = 0.5
                            
                            # Simuler des zigzags
                            future_forecast['yhat_zigzag'] = future_forecast['yhat'] * (1 + np.random.uniform(-volatility/100, volatility/100, size=len(future_forecast)))
                            future_forecast['yhat_smooth'] = future_forecast['yhat_zigzag'].ewm(span=3).mean()
                            
                            # S'assurer que la première prédiction correspond à la dernière donnée historique
                            if len(future_forecast) > 0 and not df.empty:
                                last_hist_date = df['ds'].max()
                                last_price = df.loc[df['ds'] == last_hist_date, 'y'].iloc[0] if not df[df['ds'] == last_hist_date].empty else None
                                first_pred_date = future_forecast['ds'].iloc[0]
                                
                                # Pont entre historique et prédiction
                                if last_price is not None:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=[last_hist_date, first_pred_date],
                                            y=[last_price, future_forecast['yhat_smooth'].iloc[0]],
                                            mode='lines',
                                            line=dict(color=colors[idx % len(colors)], width=1, dash='dot'),
                                            showlegend=False,
                                            hoverinfo='skip'
                                        )
                                    )
                            
                            # Ajouter la ligne de prédiction
                            fig.add_trace(
                                go.Scatter(
                                    x=future_forecast['ds'],
                                    y=future_forecast['yhat_smooth'],
                                    name=f"{symbol.upper()} (Prédiction)",
                                    line=dict(color=colors[idx % len(colors)], width=2, dash='dash'),
                                    mode='lines',
                                    hovertemplate='%{x|%d %b %Y}<br>' + symbol.upper() + ' (Prédiction): $%{y:.2f}<extra></extra>'
                                )
                            )
                except Exception as e:
                    print(f"❌ Erreur lors du traitement de {symbol}: {e}")
                    continue

            if not header_info:
                print("❌ Aucune donnée récupérée pour les symboles spécifiés.")
                return
                
            # Ajouter les annotations d'en-tête pour chaque actif
            for i, info in enumerate(header_info):
                # Position verticale décalée pour chaque actif
                y_pos = 1.15 - (i * 0.06)
                
                fig.add_annotation(
                    x=0, y=y_pos,
                    xref='paper', yref='paper',
                    text=f"<b style='color:{info['color']}'>{info['symbol']}</b> {info['name']} - " +
                         f"<b>${info['price']:.2f}</b> <span style='color:{info['change_color']}'>({info['change']:+.2f}%)</span>",
                    showarrow=False,
                    font=dict(size=14),
                    align='left',
                    xanchor='left'
                )

            # Mise en page du graphique
            fig.update_layout(
                height=800,
                margin=dict(t=100 + (len(header_info) * 25), b=50, l=50, r=50),  # Ajuster l'espace en fonction du nombre d'actifs
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left", 
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.8)',
                    orientation="h"
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                title=dict(
                    text=f"Comparaison de {'crypto-monnaies' if asset_type == 'crypto' else 'actions'}" +
                         (f" avec prédictions sur {periods} jours" if periods > 0 else ""),
                    y=0.98,
                    x=0.5,
                    xanchor='center',
                    yanchor='top'
                )
            )

            # Mise à jour des axes - mois/année sans jour sur l'axe X
            fig.update_xaxes(
                title="Date",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
                dtick="M1",  # Une date par mois
                tickformat="%b %Y"  # Format mois année sans jour
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
            import traceback
            traceback.print_exc()


def main():
    visualizer = PredictionVisualizer()

    while True:
        print("\n=== Visualisation des Prédictions Financières ===")
        print("1. Visualiser une action")
        print("2. Visualiser une crypto")
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

            # Ajout de l'option pour les prédictions dans le graphique de comparaison
            prediction_option = input("Voulez-vous inclure des prédictions? (o/n): ").lower()
            if prediction_option in ['o', 'oui', 'y', 'yes']:
                try:
                    periods = int(input("Nombre de jours de prédiction (max 365, défaut 30): ") or "30")
                    if periods < 0 or periods > 365:
                        print("La période doit être comprise entre 0 et 365 jours. Utilisation de la valeur par défaut (30).")
                        periods = 30
                except ValueError:
                    print("Valeur invalide. Utilisation de la valeur par défaut (30).")
                    periods = 30
            else:
                periods = 0  # Pas de prédictions

            if 2 <= len(symbols) <= 5:
                visualizer.compare_assets(symbols, asset_type, periods)
            else:
                print("Veuillez entrer entre 2 et 5 symboles!")

        elif choice in ["1", "2"]:
            asset_type = "stock" if choice == "1" else "crypto"
            symbol = input("\nEntrez le symbole {}: ".format('de l\'action' if asset_type == 'stock' else 'de la crypto'))

            # Demande de la période de prédiction
            while True:
                try:
                    periods = int(input("\nNombre de jours de prédiction (max 365, défaut 30): ") or "30")
                    if 1 <= periods <= 365:
                        break
                    else:
                        print("La période doit être comprise entre 1 et 365 jours!")
                except ValueError:
                    print("Veuillez entrer un nombre valide!")

            visualizer.create_single_asset_plot(symbol, asset_type, periods)

        else:
            print("Choix invalide!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramme interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\nErreur inattendue: {e}")
        import traceback
        traceback.print_exc()