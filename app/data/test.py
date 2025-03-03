import os
from dotenv import load_dotenv
import google.generativeai as genai

# Charger les variables d'environnement
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Vérifier si la clé est disponible
if not GEMINI_API_KEY:
    print("❌ La clé GEMINI_API_KEY est introuvable dans le fichier .env")
    exit(1)

# Configurer Gemini
print(f"Configuration de Gemini avec la clé: {GEMINI_API_KEY[:4]}...{GEMINI_API_KEY[-4:]}")
genai.configure(api_key=GEMINI_API_KEY)

# Test simple
try:
    print("Tentative d'appel à l'API Gemini...")
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content("Dis bonjour en français.")
    
    print("✅ Réponse de l'API reçue avec succès:")
    print(response.text)
    print("\nVotre clé API fonctionne correctement!")
except Exception as e:
    print(f"❌ Erreur lors de l'appel à l'API Gemini: {e}")
    print("\nVérifiez votre clé API ou vos quotas dans la console Google Cloud.")