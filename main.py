# main.py
from app import create_app
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

app = create_app()

if __name__ == "__main__":
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(debug=True, host="127.0.0.1", port=int(os.environ.get('PORT', 5001)))
    else:
        port = int(os.environ.get('PORT', 8000))
        app.run(host='0.0.0.0', port=port)