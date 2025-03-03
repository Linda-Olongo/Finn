import sys
import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from datetime import datetime
import google.generativeai as genai
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
import langdetect

# Charger les variables d'environnement
load_dotenv()

class ConversationMetadata:
    """Gestion avancée des métadonnées conversationnelles."""
    def __init__(self):
        self.current_language = 'fr'  # Par défaut en français
        self.user_level = 'débutant'  # Peut être : "débutant", "intermédiaire", "expert"
        self.session_stats = {
            'msg_count': 0,
            'avg_msg_length': 0,
            'confusion_level': 0,  # Niveau de confusion détecté
        }
        self.last_update = datetime.now()
        self.is_first_message = True

    def update_stats(self, message: str):
        """Mise à jour des statistiques de session."""
        self.session_stats['msg_count'] += 1
        new_length = len(message)
        old_avg = self.session_stats['avg_msg_length']
        self.session_stats['avg_msg_length'] = (
            (old_avg * (self.session_stats['msg_count'] - 1) + new_length) / self.session_stats['msg_count']
        )
        self.is_first_message = False

    def detect_language(self, text: str) -> str:
        """Détection automatique de la langue."""
        try:
            detected = langdetect.detect(text)
            if detected in ['fr', 'en']:
                self.current_language = detected
            return self.current_language
        except:
            return self.current_language  # Conserve la langue précédente si erreur

class GeminiHandler:
    def __init__(self):
        """Initialisation du gestionnaire Gemini."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Clé API GEMINI introuvable.")
        
        genai.configure(api_key=api_key)
        # Utiliser le bon modèle - Gemini 1.5 Pro ou Gemini 1.0 Pro (vérifier la disponibilité)
        try:
            # Essayer d'abord avec gemini-1.5-pro
            self.model = genai.GenerativeModel('gemini-1.5-pro')
        except Exception:
            try:
                # Si gemini-1.5-pro échoue, essayer avec gemini-1.0-pro
                self.model = genai.GenerativeModel('gemini-1.0-pro')
            except Exception:
                # Dernier recours
                self.model = genai.GenerativeModel('gemini-pro')
        
        self.metadata = ConversationMetadata()
        self.chat = self.model.start_chat(history=[])

    def _generate_system_prompt(self, query: str) -> str:
        """Construit un prompt basé sur le contexte et la langue détectée."""
        detected_language = self.metadata.detect_language(query)
        self.metadata.current_language = detected_language
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        system_prompt = f"""
        Tu es un assistant financier virtuel nommé Finn capable de s'exprimer en anglais et en français.  
        Tu dois fournir des réponses précises, adaptées et compréhensibles selon la langue de l'utilisateur.  
        Date et heure actuelles : {current_time}  

        1- Informations sur l'utilisateur :
        - Langue actuelle : {detected_language}
        - Niveau utilisateur : {self.metadata.user_level}
        - Premier message : {self.metadata.is_first_message}
        - Nombre total de messages échangés : {self.metadata.session_stats['msg_count']}

        2- Directives générales :
        a- Toujours fournir des réponses claires et précises, adaptées au contexte.  
        b- Ne jamais répondre avec des phrases toutes faites comme "Bien sûr" ou "Je suis Finn, votre assistant".  
        c- Ne jamais donner d'informations erronées ou spéculatives. Si tu ne sais pas, admets-le humblement.  
        d- Si l'utilisateur semble confus ou frustré, sois pédagogue et patient.  
        e- Si l'utilisateur critique une réponse, accepte la critique avec humilité et reformule.  
        f- Toujours adapter le niveau technique en fonction de l'expertise détectée.  

        Message de l'utilisateur : {query}
        
        Réponds directement en texte simple, sans mise en forme JSON.
        """
        return system_prompt

    def process_query(self, query: str) -> Dict[str, Any]:
        """Traite la requête et génère une réponse en fonction du contexte et de la langue détectée."""
        # Mettre à jour les statistiques
        self.metadata.update_stats(query)
        
        # Générer le prompt système
        system_prompt = self._generate_system_prompt(query)

        try:
            # Utiliser la conversation pour maintenir le contexte
            response = self.chat.send_message(system_prompt)
            
            if response and response.text:
                return {"message": response.text.strip()}
            else:
                raise ValueError("Réponse vide de l'API")
        except Exception as e:
            print(f"Erreur détaillée: {str(e)}")  # Debug: afficher l'erreur spécifique
            return {"message": "Une erreur est survenue, veuillez réessayer."}

if __name__ == "__main__":
    try:
        # Afficher les modèles disponibles
        print("Modèles Gemini disponibles:")
        for model in genai.list_models():
            if "gemini" in model.name.lower():
                print(f"- {model.name}")
    
        handler = GeminiHandler()
        print("Initialisation de Finn terminée. Prêt à discuter.")
        
        while True:
            try:
                query = input("\nVous : ").strip()
                
                # Quitter l'application
                if query.lower() in ['quit', 'exit', 'bye', 'au revoir', 'goodbye']:
                    print("\nFinn : À bientôt ! 👋")
                    break

                # Traiter la requête utilisateur
                response = handler.process_query(query)
                print(f"\nFinn : {response['message']}")
                    
            except Exception as e:
                print(f"Erreur: {str(e)}")
    except Exception as e:
        print(f"Erreur d'initialisation: {str(e)}")