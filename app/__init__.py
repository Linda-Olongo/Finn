# app/__init__.py
from flask import Flask
from app.config import config
from app.routes import main

def create_app(config_name='default'):
    app = Flask(__name__)
    
    # Charger la configuration
    app.config.from_object(config[config_name])
    
    # Enregistrer les blueprints
    app.register_blueprint(main)
    
    return app