import sys
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import re

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import google.generativeai as genai
from typing import Dict, Any, List, Optional, Tuple, Union
from dotenv import load_dotenv
import langdetect

# Importer le collecteur de données financières
from data.api_fetcher import DataCollector

# Charger les variables d'environnement
load_dotenv()

class MarketDataProvider:
    """Fournit des données de marché pour actions et cryptos."""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.cached_data = {}  # Cache pour éviter de trop solliciter les APIs
        self.cache_expiry = {}  # Expiration du cache
        self.cache_duration = 900  # 15 minutes en secondes
    
    def get_current_price(self, asset_type: str, query: str) -> Dict:
        """Récupère le prix actuel d'un actif."""
        cache_key = f"{asset_type}_{query}_current"
        current_time = datetime.now()
        
        # Vérifier si les données sont en cache et valides
        if (cache_key in self.cached_data and 
            cache_key in self.cache_expiry and 
            current_time < self.cache_expiry[cache_key]):
            return self.cached_data[cache_key]
        
        try:
            if asset_type.lower() == "stock":
                data = self.data_collector.get_stock_current(query)
            elif asset_type.lower() == "crypto":
                data = self.data_collector.get_crypto_current(query)
            else:
                raise ValueError(f"Type d'actif non supporté: {asset_type}")
            
            # Mettre en cache
            self.cached_data[cache_key] = data
            self.cache_expiry[cache_key] = current_time + timedelta(seconds=self.cache_duration)
            
            return data
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la récupération du prix actuel: {str(e)}")
    
    def get_historical_data(self, asset_type: str, query: str) -> pd.DataFrame:
        """Récupère les données historiques d'un actif."""
        cache_key = f"{asset_type}_{query}_historical"
        current_time = datetime.now()
        
        # Vérifier si les données sont en cache et valides
        if (cache_key in self.cached_data and 
            cache_key in self.cache_expiry and 
            current_time < self.cache_expiry[cache_key]):
            return self.cached_data[cache_key]
        
        try:
            if asset_type.lower() == "stock":
                data = self.data_collector.get_stock_historical(query)
            elif asset_type.lower() == "crypto":
                data = self.data_collector.get_crypto_historical(query)
            else:
                raise ValueError(f"Type d'actif non supporté: {asset_type}")
            
            # Mettre en cache
            self.cached_data[cache_key] = data
            self.cache_expiry[cache_key] = current_time + timedelta(seconds=self.cache_duration)
            
            return data
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la récupération des données historiques: {str(e)}")
    
    def get_price_at_date(self, asset_type: str, query: str, target_date: datetime) -> Optional[Dict]:
        """Récupère le prix d'un actif à une date spécifique."""
        try:
            # Récupérer les données historiques
            hist_data = self.get_historical_data(asset_type, query)
            
            # Vérifier si la date est dans les données disponibles
            if target_date.date() < hist_data.index.min().date():
                return {
                    "status": "error",
                    "message": f"Date trop ancienne ({target_date.strftime('%d/%m/%Y')}). Nos données les plus anciennes commencent le {hist_data.index.min().strftime('%d/%m/%Y')}."
                }
            
            # Si c'est une date future par rapport à aujourd'hui
            current_date = datetime.now().date()
            if target_date.date() > current_date:
                # Si c'est une date future, retourner une indication spéciale
                # mais ne pas bloquer la demande pour les prédictions futures
                is_future = True
            else:
                is_future = False
            
            # Convertir la date cible au format de l'index
            target_date_normalized = pd.Timestamp(target_date.date())
            
            # Si la date est dans le futur par rapport aux données disponibles
            if target_date.date() > hist_data.index.max().date():
                # Pour les dates futures par rapport aux données disponibles,
                # utiliser la dernière date disponible pour les données historiques
                if not is_future:
                    target_date_normalized = hist_data.index.max()
            
            # Trouver la date la plus proche si la date exacte n'existe pas
            if target_date_normalized not in hist_data.index:
                closest_dates = hist_data.index[hist_data.index <= target_date_normalized]
                if len(closest_dates) == 0:
                    closest_dates = hist_data.index[hist_data.index >= target_date_normalized]
                
                if len(closest_dates) == 0:
                    return {
                        "status": "error",
                        "message": "Aucune donnée disponible à proximité de la date demandée."
                    }
                
                target_date_normalized = closest_dates[-1]
            
            # Récupérer les données à la date spécifiée
            price_data = hist_data.loc[target_date_normalized]
            
            # Préparer le résultat
            result = {
                "status": "success",
                "date": target_date_normalized.strftime('%d/%m/%Y'),
                "price": float(price_data["Close"]),  # Convertir en float pour garantir la formatabilité
                "open": float(price_data.get("Open", 0)) if pd.notna(price_data.get("Open")) else None,
                "volume": float(price_data.get("Volume", 0)) if pd.notna(price_data.get("Volume")) else None,
                "name": hist_data.attrs.get("name", ""),
                "symbol": hist_data.attrs.get("symbol", hist_data.attrs.get("id", "")),
                "type": asset_type,
                "is_exact_date": target_date.date() == target_date_normalized.date(),
                "is_future": is_future
            }
            
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erreur lors de la récupération du prix historique: {str(e)}"
            }
    
    def get_specific_data(self, asset_type: str, query: str, data_field: str, target_date: Optional[datetime] = None) -> Dict:
        """Récupère une information spécifique sur un actif (prix, volume, etc.)."""
        try:
            # Si une date est spécifiée, obtenir les données historiques
            if target_date:
                historical_data = self.get_price_at_date(asset_type, query, target_date)
                
                if historical_data["status"] == "error":
                    return historical_data
                
                # Si la date est future par rapport à la date actuelle et qu'on a pas encore activé les prédictions
                if historical_data.get("is_future", False):
                    # Pour l'instant, utiliser simplement les données actuelles (sera remplacé par prédiction plus tard)
                    current_data = self.get_current_price(asset_type, query)
                    return {
                        "status": "success",
                        "field": data_field.lower(),
                        "value": current_data.get("price" if data_field.lower() in ["price", "prix"] else data_field.lower(), "N/A"),
                        "date": "actuel",
                        "asset_name": current_data.get("name", ""),
                        "asset_symbol": current_data.get("symbol", current_data.get("id", "")),
                        "message": f"Données pour {target_date.strftime('%d/%m/%Y')}"
                    }
                
                # Si le champ demandé existe dans les données
                if data_field.lower() in ["price", "prix"]:
                    return {
                        "status": "success",
                        "field": "prix",
                        "value": historical_data["price"],
                        "date": historical_data["date"],
                        "asset_name": historical_data["name"],
                        "asset_symbol": historical_data["symbol"]
                    }
                elif data_field.lower() in ["volume"]:
                    return {
                        "status": "success",
                        "field": "volume",
                        "value": historical_data["volume"],
                        "date": historical_data["date"],
                        "asset_name": historical_data["name"],
                        "asset_symbol": historical_data["symbol"]
                    }
                elif data_field.lower() in ["open", "ouverture"]:
                    return {
                        "status": "success",
                        "field": "ouverture",
                        "value": historical_data["open"],
                        "date": historical_data["date"],
                        "asset_name": historical_data["name"],
                        "asset_symbol": historical_data["symbol"]
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Le champ {data_field} n'est pas disponible pour les données historiques."
                    }
            
            # Sinon, obtenir les données actuelles
            current_data = self.get_current_price(asset_type, query)
            
            # Mapper les champs demandés aux champs disponibles dans les données
            field_mappings = {
                "price": "price", "prix": "price",
                "volume": "volume",
                "open": "open", "ouverture": "open",
                "symbol": "symbol", "symbole": "symbol",
                "name": "name", "nom": "name",
                "change": "change", "variation": "change",
                "high": "high", "haut": "high",
                "low": "low", "bas": "low"
            }
            
            # Ajouter des mappings spécifiques pour les cryptos
            if asset_type.lower() == "crypto":
                field_mappings.update({
                    "market_cap": "market_cap", "capitalisation": "market_cap",
                    "supply": "circulating_supply", "offre": "circulating_supply",
                    "total_supply": "total_supply", "offre_totale": "total_supply",
                    "max_supply": "max_supply", "offre_maximale": "max_supply"
                })
            
            # Vérifier si le champ demandé existe
            normalized_field = field_mappings.get(data_field.lower())
            if not normalized_field:
                return {
                    "status": "error",
                    "message": f"Le champ {data_field} n'est pas disponible."
                }
            
            # Récupérer la valeur
            if normalized_field in current_data:
                return {
                    "status": "success",
                    "field": data_field.lower(),
                    "value": current_data[normalized_field],
                    "date": "actuel",
                    "asset_name": current_data.get("name", ""),
                    "asset_symbol": current_data.get("symbol", current_data.get("id", ""))
                }
            else:
                return {
                    "status": "error",
                    "message": f"Le champ {data_field} n'est pas disponible pour cet actif."
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erreur lors de la récupération des données spécifiques: {str(e)}"
            }
            
    def compare_assets(self, assets_list: List[Dict], fields_list: List[str]) -> Dict:
        """Compare plusieurs actifs sur différents critères."""
        try:
            result = {
                "status": "success",
                "comparison": [],
                "fields": fields_list
            }
            
            for asset in assets_list:
                asset_type = asset.get("type", "unknown")
                asset_name = asset.get("name", "")
                
                if asset_type not in ["stock", "crypto"]:
                    continue
                
                try:
                    # Récupérer les données actuelles de l'actif
                    current_data = self.get_current_price(asset_type, asset_name)
                    
                    # Préparer les données pour cet actif
                    asset_data = {
                        "name": current_data.get("name", asset_name),
                        "symbol": current_data.get("symbol", current_data.get("id", "")),
                        "type": asset_type,
                        "fields": {}
                    }
                    
                    # Récupérer les valeurs des champs demandés
                    for field in fields_list:
                        field_data = self.get_specific_data(asset_type, asset_name, field)
                        if field_data["status"] == "success":
                            asset_data["fields"][field] = field_data["value"]
                        else:
                            asset_data["fields"][field] = "N/A"
                    
                    result["comparison"].append(asset_data)
                    
                except Exception as e:
                    print(f"Erreur lors de la récupération des données pour {asset_name}: {str(e)}")
                    # Continuer avec l'actif suivant
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erreur lors de la comparaison des actifs: {str(e)}"
            }

class ConversationMemory:
    """Gestion de l'historique des conversations."""
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
        self.repeated_queries = {}
        
    def add_interaction(self, query: str, response: str):
        """Ajoute une interaction à l'historique."""
        self.history.append({"user": query, "assistant": response})
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        # Détecter les requêtes répétées
        query_normalized = query.lower().strip()
        if query_normalized in self.repeated_queries:
            self.repeated_queries[query_normalized] += 1
        else:
            self.repeated_queries[query_normalized] = 1
            
    def is_repeated_query(self, query: str) -> bool:
        """Vérifie si la requête a été posée précédemment."""
        query_normalized = query.lower().strip()
        return query_normalized in self.repeated_queries
    
    def get_repetition_count(self, query: str) -> int:
        """Retourne le nombre de fois qu'une requête a été posée."""
        query_normalized = query.lower().strip()
        return self.repeated_queries.get(query_normalized, 0)
    
    def get_recent_interactions(self, count=3) -> List[Dict[str, str]]:
        """Retourne les interactions récentes."""
        return self.history[-count:] if len(self.history) >= count else self.history

class ConversationMetadata:
    """Gestion des métadonnées conversationnelles."""
    def __init__(self):
        self.current_language = 'fr'  # Par défaut en français
        self.user_level = 'débutant'
        self.session_stats = {
            'msg_count': 0,
            'avg_msg_length': 0,
        }
        self.last_update = datetime.now()
        self.is_first_message = True
        self.variation_counter = 0

    def update_stats(self, message: str):
        """Mise à jour des statistiques de session."""
        self.session_stats['msg_count'] += 1
        new_length = len(message)
        old_avg = self.session_stats['avg_msg_length']
        self.session_stats['avg_msg_length'] = (
            (old_avg * (self.session_stats['msg_count'] - 1) + new_length) / self.session_stats['msg_count']
        )
        self.is_first_message = False
        self.variation_counter += 1

    def detect_language(self, text: str) -> str:
        """Détection automatique de la langue."""
        if not text or len(text.strip()) < 3:
            return self.current_language  # Conserver la langue précédente si message trop court
            
        try:
            detected = langdetect.detect(text)
            if detected in ['fr', 'en']:
                self.current_language = detected
            return self.current_language
        except:
            return self.current_language  # Conserve la langue précédente si erreur

class DateParser:
    """Analyse et convertit différents formats de date."""
    
    @staticmethod
    def parse_date(date_str: str) -> Optional[datetime]:
        """Parse une date dans différents formats possibles."""
        # Liste des formats à essayer
        date_formats = [
            # Formats numériques
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",  # européen (jour/mois/année)
            "%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d",  # ISO (année/mois/jour)
            "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",  # américain (mois/jour/année)
            
            # Formats courts
            "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",  # européen court
            "%y/%m/%d", "%y-%m-%d", "%y.%m.%d",  # ISO court
            "%m/%d/%y", "%m-%d-%y", "%m.%d.%y",  # américain court
        ]
        
        # Essayer tous les formats
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Si aucun format numérique ne fonctionne, essayer de parser des dates textuelles
        try:
            # Pour les dates comme "12 mars 2024", "12 March 2024", etc.
            # Utiliser une approche plus souple via une regex complexe
            
            # Exemple simplifié pour les mois en français
            months_fr = {
                "janvier": 1, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
                "juillet": 7, "août": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12
            }
            
            # Exemple simplifié pour les mois en anglais
            months_en = {
                "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
            }
            
            # Regex pour capturer "jour mois année"
            pattern_fr = r"(\d{1,2})[ \t]*([a-zéûù]+)[ \t]*(\d{2,4})"
            pattern_en = r"([a-z]+)[ \t]*(\d{1,2})(?:st|nd|rd|th)?,?[ \t]*(\d{2,4})"
            
            # Essayer le pattern français
            match_fr = re.search(pattern_fr, date_str.lower())
            if match_fr:
                day, month_name, year = match_fr.groups()
                for month_fr, month_num in months_fr.items():
                    if month_name.startswith(month_fr[:3]) or month_fr.startswith(month_name[:3]):
                        month = month_num
                        # Ajuster l'année si nécessaire
                        if len(year) == 2:
                            year = f"20{year}" if int(year) < 50 else f"19{year}"
                        return datetime(int(year), month, int(day))
            
            # Essayer le pattern anglais
            match_en = re.search(pattern_en, date_str.lower())
            if match_en:
                month_name, day, year = match_en.groups()
                for month_en, month_num in months_en.items():
                    if month_name.startswith(month_en[:3]) or month_en.startswith(month_name[:3]):
                        month = month_num
                        # Ajuster l'année si nécessaire
                        if len(year) == 2:
                            year = f"20{year}" if int(year) < 50 else f"19{year}"
                        return datetime(int(year), month, int(day))
                        
            # Essayer de détecter des mots-clés pour les dates relatives
            if "hier" in date_str.lower() or "yesterday" in date_str.lower():
                return datetime.now() - timedelta(days=1)
                
            if "aujourd'hui" in date_str.lower() or "today" in date_str.lower():
                return datetime.now()
                
            return None
        except:
            return None

class GeminiHandler:
    def __init__(self):
        """Initialisation du gestionnaire Gemini."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Clé API GEMINI introuvable.")
        
        genai.configure(api_key=api_key)
        # Utiliser le bon modèle - Gemini 1.5 Pro ou Gemini 1.0 Pro (vérifier la disponibilité)
        try:
            self.model = genai.GenerativeModel('gemini-1.5-pro')
        except Exception:
            try:
                self.model = genai.GenerativeModel('gemini-1.0-pro')
            except Exception:
                self.model = genai.GenerativeModel('gemini-pro')
        
        self.metadata = ConversationMetadata()
        self.memory = ConversationMemory()
        self.market_data = MarketDataProvider()
        self.chat = self.model.start_chat(history=[])

    def _generate_system_prompt(self, query: str) -> str:
        """Construit un prompt basé sur le contexte et la langue détectée."""
        detected_language = self.metadata.detect_language(query)
        self.metadata.current_language = detected_language
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Obtenir les informations sur les répétitions
        is_repeated = self.memory.is_repeated_query(query)
        repetition_count = self.memory.get_repetition_count(query)
        
        # Récupérer l'historique récent
        recent_interactions = self.memory.get_recent_interactions(2)
        recent_history = ""
        
        if recent_interactions:
            recent_history = "Historique récent :\n"
            for i, interaction in enumerate(recent_interactions):
                recent_history += f"Q{i+1}: {interaction['user']}\n"
                recent_history += f"R{i+1}: {interaction['assistant']}\n"
        
        # Construction du prompt avec instructions de concision
        system_prompt = f"""
        Tu es Finn, un conseiller financier virtuel expert, capable de communiquer en anglais et en français.
        Date et heure actuelles : {current_time}

        1- Contexte actuel :
        - Langue détectée : {detected_language}
        - Message répété : {is_repeated} (répétition #{repetition_count})
        - Premier message : {self.metadata.is_first_message}
        
        2- Historique :
        {recent_history}

        3- RÈGLES CRUCIALES À SUIVRE :
        a- Réponds avec un style ÉQUILIBRÉ: ni trop court ni trop long.
        b- Pour une salutation simple, réponds en une phrase amicale dans la langue détectée.
        c- Ne jamais commencer par "Je suis Finn" ou "En tant que conseiller".
        d- Si la question concerne la finance, fournir 3-4 phrases informatives avec une structure claire.
        e- Si la question est répétée, varie ta réponse.
        f- Si la requête concerne une donnée financière spécifique (prix, volume):
           - Pour un prix: réponds avec le montant suivi de "$" (ex: "145.75 $")
           - Pour un volume: donne le chiffre formaté avec des séparateurs de milliers
        g- Si une requête concerne un prix passé, réponds simplement avec le prix à cette date sans mentionner que c'est "une prédiction" ou "une date passée".
        h- Pour les tableaux comparatifs, utilise 4-5 critères pertinents même si l'utilisateur n'a pas précisé.
        i- Adapte ta réponse selon la complexité de la question: 
           - Question simple: 2-3 phrases concises
           - Question complexe: 3-4 phrases structurées
           - Explication technique: 4-5 points clairs
        j- Ta réponse doit toujours respecter la langue détectée (français ou anglais).
        
        4- FORMAT DES RÉPONSES SPÉCIFIQUES:
        - Prix actuel: "143.75 $" (toujours ajouter $ après le prix)
        - Prix historique: "143.75 $ (le 15/03/2024)" 
        - Conseil financier: structure claire de 3-4 points essentiels
        - Question d'explication générale: 3-4 phrases informatives
        - Ne jamais dire "voici le prix de X" ou "le prix de X est", donner DIRECTEMENT la valeur

        Message de l'utilisateur : {query}
        
        Réponds directement en texte simple, sans mise en forme JSON.
        """
        return system_prompt

    def _extract_financial_data_request(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extrait des informations sur la demande de données financières.
        Retourne les détails de la requête: type d'actif, nom, champ demandé et date éventuelle.
        """
        try:
            # Créer un prompt pour l'extraction d'informations
            extraction_prompt = f"""
            Analyse cette requête concernant des données financières.
            Format de réponse strict: JSON avec les champs suivants:
            - asset_type: "stock", "crypto", ou "unknown"
            - asset_name: le nom ou symbole de l'actif (ou liste d'actifs séparés par des virgules)
            - data_field: le type de donnée demandée ("price", "volume", "open", "symbol", etc.) ou liste de champs
            - date_str: la date mentionnée sous forme de texte, ou null si aucune date n'est mentionnée
            - is_request_for_specific_data: true si la demande concerne une donnée spécifique, false sinon
            - is_comparison_request: true si la demande concerne une comparaison entre plusieurs actifs
            
            Requête: "{query}"
            
            N'inclus AUCUN texte supplémentaire, seulement le JSON.
            """
            
            response = self.model.generate_content(extraction_prompt)
            if response and response.text:
                # Nettoyer la réponse pour s'assurer qu'elle contient uniquement du JSON
                cleaned_response = response.text.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                # Analyser le JSON
                info = json.loads(cleaned_response)
                
                # Extraire les informations
                asset_type = info.get("asset_type", "unknown")
                asset_name = info.get("asset_name")
                data_field = info.get("data_field")
                date_str = info.get("date_str")
                is_specific = info.get("is_request_for_specific_data", False)
                is_comparison = info.get("is_comparison_request", False)
                
                # Traiter les cas où asset_name pourrait être une liste
                assets_list = []
                if asset_name and isinstance(asset_name, str) and "," in asset_name:
                    # Si l'asset_name contient des virgules, c'est probablement une liste
                    asset_names = [name.strip() for name in asset_name.split(",")]
                    for name in asset_names:
                        assets_list.append({
                            "name": name,
                            "type": asset_type
                        })
                elif asset_name:
                    assets_list.append({
                        "name": asset_name,
                        "type": asset_type
                    })
                
                # Traiter les cas où data_field pourrait être une liste
                fields_list = []
                if data_field and isinstance(data_field, str) and "," in data_field:
                    # Si data_field contient des virgules, c'est probablement une liste
                    fields_list = [field.strip() for field in data_field.split(",")]
                elif data_field:
                    fields_list = [data_field]
                
                # Parser la date si présente
                target_date = None
                if date_str:
                    target_date = DateParser.parse_date(date_str)
                
                return {
                    "asset_type": asset_type,
                    "asset_name": asset_name,
                    "data_field": data_field,
                    "target_date": target_date,
                    "date_str": date_str,
                    "is_specific_request": is_specific,
                    "is_comparison_request": is_comparison,
                    "assets_list": assets_list,
                    "fields_list": fields_list
                }
            
            return None
        except Exception as e:
            print(f"Erreur lors de l'extraction d'informations: {str(e)}")
            return None

    def _get_specific_market_data(self, request_info: Dict[str, Any]) -> Optional[Dict]:
        """Obtient une donnée financière spécifique basée sur la requête."""
        try:
            asset_type = request_info["asset_type"]
            asset_name = request_info["asset_name"]
            data_field = request_info["data_field"]
            target_date = request_info.get("target_date")
            
            # Récupérer la donnée spécifique
            return self.market_data.get_specific_data(asset_type, asset_name, data_field, target_date)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erreur lors de la récupération des données: {str(e)}"
            }

    def process_query(self, query: str) -> Dict[str, Any]:
        """Traite la requête et génère une réponse en fonction du contexte et de la langue détectée."""
        # Mettre à jour les statistiques
        self.metadata.update_stats(query)
        
        # Vérifier si la requête concerne une donnée financière spécifique
        data_request = self._extract_financial_data_request(query)
        market_data_response = None
        
        if data_request and data_request["is_specific_request"]:
            try:
                market_data_response = self._get_specific_market_data(data_request)
            except Exception as e:
                print(f"Erreur lors de la récupération des données spécifiques: {str(e)}")
        
        # Générer le prompt système
        system_prompt = self._generate_system_prompt(query)
        
        # Ajouter les données de marché au prompt si disponibles
        if market_data_response:
            system_prompt += f"\n\nDonnées financières spécifiques disponibles:\n{json.dumps(market_data_response, indent=2)}\n"
            
            # Instructions spécifiques pour l'utilisation des données financières spécifiques
            system_prompt += """
            INSTRUCTIONS POUR LES DONNÉES SPÉCIFIQUES:
            1. Pour une demande de donnée spécifique comme un prix, un volume, ou un symbole:
               - Si le statut est "success", fournis UNIQUEMENT la valeur demandée sans explication.
               - Format: "[Valeur] [Unité si applicable]" (ex: "152.75 $" pour un prix)
            
            2. Si le statut est "error" ou "future_date":
               - Fournis une réponse courte expliquant le problème.
               - Pour une date future, indique qu'il s'agit d'une prédiction.
            
            3. Ne mentionne JAMAIS que tu as consulté une API ou une base de données.
            """

        try:
            # Utiliser la conversation pour maintenir le contexte
            response = self.chat.send_message(system_prompt)
            
            if response and response.text:
                response_text = response.text.strip()
                # Stocker l'interaction dans la mémoire
                self.memory.add_interaction(query, response_text)
                return {"message": response_text}
            else:
                error_msg = "Je n'ai pas pu traiter votre demande. Pourriez-vous la reformuler?"
                self.memory.add_interaction(query, error_msg)
                return {"message": error_msg}
        except Exception as e:
            print(f"Erreur détaillée: {str(e)}")  # Debug: afficher l'erreur spécifique
            error_msg = "Une erreur est survenue. Veuillez réessayer."
            self.memory.add_interaction(query, error_msg)
            return {"message": error_msg}

if __name__ == "__main__":
    try:
        # Afficher les modèles disponibles
        print("Modèles Gemini disponibles:")
        for model in genai.list_models():
            if "gemini" in model.name.lower():
                print(f"- {model.name}")
    
        handler = GeminiHandler()
        print("\n═════════════════════════════════")
        print("   Finn - Conseiller Financier   ")
        print("═════════════════════════════════\n")
        print("Posez vos questions financières (tapez 'exit' pour quitter)\n")
        
        while True:
            try:
                query = input("\nVous : ").strip()
                
                # Quitter l'application
                if query.lower() in ['quit', 'exit', 'bye', 'au revoir', 'goodbye']:
                    print("\nFinn : À bientôt!")
                    break

                # Traiter la requête utilisateur
                response = handler.process_query(query)
                print(f"\nFinn : {response['message']}")
                    
            except Exception as e:
                print(f"Erreur: {str(e)}")
    except Exception as e:
        print(f"Erreur d'initialisation: {str(e)}")