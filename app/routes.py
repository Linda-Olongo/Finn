import os
import re
import uuid
import html
import jwt
import secrets
import logging
from functools import wraps
from datetime import datetime, timedelta
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from werkzeug.security import generate_password_hash

from flask import (
    render_template, Blueprint, request, jsonify,
    session, redirect, url_for, make_response
)

# -- MongoDB --
from pymongo import MongoClient

# -- Google OAuth libs --
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# -- Password hashing --
from werkzeug.security import generate_password_hash, check_password_hash

# -- Imports des classes de traitement du chat
from app.data.news_handler import NewsHandler
from app.nlp.ClaudeHandler import ClaudeHandler  
from app.data.simulator import TradingSimulator

# Clé secrète pour JWT
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_EXPIRATION = 30  # en jours

# Chargé SendGrid pour l'envoie des liens de reset mot de passe
SENDGRID_API_KEYS = os.getenv("SENDGRID_API_KEYS")
SENDER_EMAILS = os.getenv("SENDER_EMAILS")

# Configuration de MongoDB
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["finn"]
users_col = db["users"]  # Nouvelle collection pour les utilisateurs
conversations_col = db["conversations"]  # Nouvelle collection pour les conversations

# Définition de SITE_URL pour tous les environnements
if os.getenv("RENDER") == "true" or os.getenv("IS_PRODUCTION") == "true":
    SITE_URL = os.getenv("RENDER_EXTERNAL_URL", "https://votre-app.onrender.com")
else:
    SITE_URL = os.getenv("LOCAL_SITE_URL", "http://localhost:5000")

# Configuration de Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
BASE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# Définition de GOOGLE_REDIRECT_URI
if os.getenv("RENDER") == "true" or os.getenv("IS_PRODUCTION") == "true":
    GOOGLE_REDIRECT_URI = f"{SITE_URL}/google_callback"
else:
    GOOGLE_REDIRECT_URI = BASE_REDIRECT_URI

main = Blueprint('main', __name__)
news_handler = NewsHandler()
claude_handler = ClaudeHandler()  
trading_simulator = TradingSimulator()

# Configuration du logging pour éviter les logs visibles (rediriger vers un fichier)
if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(
    level=logging.WARNING,  # Seulement les avertissements et erreurs
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/app.log'  # Écriture dans un fichier
)
logger = logging.getLogger(__name__)

# PAGE D'ACCUEIL
@main.route('/')
def index():
    return render_template('loading.html')

# 1. Créer une route spécifique pour la redirection depuis les emails
@main.route('/welcome/<token>')
def welcome_redirect(token):
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = data.get('user_id')
        user = users_col.find_one({"_id": user_id})
        if not user:
            return redirect(url_for('main.show_connexion'))
        response = make_response(redirect(url_for('main.chat')))
        response.set_cookie('finnToken', token, httponly=True, max_age=30*24*60*60)
        return response
    except jwt.ExpiredSignatureError:
        return redirect(url_for('main.show_connexion'))
    except jwt.InvalidTokenError:
        return redirect(url_for('main.show_connexion'))
    
# Fonction pour afficher les conversations recentes
def get_user_conversations(user_id):
    """Récupère les conversations récentes d'un utilisateur."""
    recent_conversations = {}
    user_conversations = conversations_col.find(
        {"user_id": user_id, "messages": {"$exists": True, "$not": {"$size": 0}}}, 
        {"title": 1, "created_at": 1}
    ).sort("created_at", -1).limit(10)
    
    for conv in user_conversations:
        created_at = conv.get('created_at')
        if isinstance(created_at, datetime):
            created_at_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
            created_at_timestamp = created_at.timestamp()
        else:
            created_at_str = str(created_at)
            created_at_timestamp = 0
            
        recent_conversations[conv["_id"]] = {
            "title": conv.get("title", "New Conversation"), 
            "created_at": created_at_str,
            "timestamp": created_at_timestamp
        }
    
    return recent_conversations

# Fonction pour envoyer des mails de bienvenue à la plateforme
def send_welcome_email(to_email, first_name, user_id=None, token=None):
    try:
        message = Mail(
            from_email=SENDER_EMAILS,
            to_emails=to_email,
            subject="Welcome to Finn 2.1 Prime! Your Financial Journey Begins",
            html_content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #000000; max-width: 600px; margin: 0 auto; }}
                    .header {{ text-align: center; padding: 20px 0; border-bottom: 1px solid #e0e0e0; }}
                    .logo {{ width: 60px; height: 60px; margin-bottom: 10px; }}
                    h1, h2, h3, p {{ color: #000000; }}
                    .content {{ padding: 20px; }}
                    .features {{ display: flex; justify-content: space-between; margin: 30px 0; text-align: center; }}
                    .feature {{ flex: 1; padding: 15px; border-radius: 5px; background-color: #f5f5f5; margin: 0 5px; }}
                    .feature p {{ color: #000000; }}
                    .highlight {{ font-weight: bold; color: #000000; font-size: 1.1em; }}
                    .footer {{ text-align: center; font-size: 12px; color: #000000; padding: 20px 0; border-top: 1px solid #e0e0e0; }}
                    .footer p {{ color: #000000; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <svg class="logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="black" fill="none" stroke-width="2"/>
                    </svg>
                    <h1>Welcome to Finn 2.1 Prime!</h1>
                </div>
                <div class="content">
                    <h2>Hello {first_name},</h2>
                    <p>Thank you for joining Finn 2.1 Prime! We're thrilled to have you on board and can't wait for you to experience the future of financial intelligence.</p>
                    <p>With Finn, you'll unlock powerful insights that will transform the way you approach your financial decisions.</p>
                    <p class="highlight">Your account is now active. Connect to the platform to start exploring all our features.</p>
                    <div class="features">
                        <div class="feature">
                            <h3>AI Chat</h3>
                            <p>Get instant financial insights and advice through our advanced AI assistant.</p>
                        </div>
                        <div class="feature">
                            <h3>Market News</h3>
                            <p>Stay updated with the latest market trends and financial news.</p>
                        </div>
                        <div class="feature">
                            <h3>Market Simulator</h3>
                            <p>Test strategies and gain trading experience in our risk-free environment.</p>
                        </div>
                    </div>
                    <p>If you have any questions or need assistance, don't hesitate to reach out to our support team at {SENDER_EMAILS}.</p>
                    <p>Best regards,<br>The Finn Team</p>
                </div>
                <div class="footer">
                    <p>© 2025 Finn. All rights reserved.</p>
                    <p>This email was sent to {to_email}. If you didn't sign up for Finn, please ignore this email.</p>
                </div>
            </body>
            </html>
            """
        )
        sg = SendGridAPIClient(SENDGRID_API_KEYS)
        response = sg.send(message)
        return True
    except Exception as e:
        return False

# INSCRIPTION (MANUELLE) - GET => Formulaire, POST => Création
@main.route('/inscription', methods=['GET'])
def show_inscription():
    return render_template('inscription.html')

@main.route('/check-email', methods=['POST'])
def check_email():
    try:
        data = request.get_json()
        if not data or "email" not in data:
            return jsonify({"message": "Email is required"}), 400
        email = data.get("email")
        if not email:
            return jsonify({"message": "Email is required"}), 400
        existing_user = users_col.find_one({"email": email})
        if existing_user:
            return jsonify({"message": "This email is already registered"}), 400
        return jsonify({"message": "Email is available"}), 200
    except Exception:
        return jsonify({"message": "Server error. Please try again."}), 500

@main.route('/inscription', methods=['POST'])
def process_inscription():
    data = request.get_json()
    first_name = data.get("firstName")
    last_name = data.get("lastName")
    email = data.get("email")
    password = data.get("password")

    if not all([first_name, last_name, email, password]):
        return jsonify({"message": "All fields are required"}), 400
    existing_user = users_col.find_one({"email": email})
    if existing_user:
        return jsonify({"message": "This email is already registered"}), 400
    if len(password) < 8 or not re.search(r"[A-Z]", password) or not re.search(r"[a-z]", password) or not re.search(r"\d", password) or not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password) or first_name.lower() in password.lower() or last_name.lower() in password.lower():
        return jsonify({"message": "Password must be at least 8 characters with uppercase, lowercase, digit, symbol, and not contain your name"}), 400
    hashed_pw = generate_password_hash(password)
    user_id = str(uuid.uuid4())
    new_user = {
        "_id": user_id,
        "firstName": first_name,
        "lastName": last_name,
        "email": email,
        "password": hashed_pw,
        "created_at": datetime.utcnow(),
        "auth_method": "email",
        "preferences": {"theme": "light", "notifications": True},
        "last_login": None
    }
    users_col.insert_one(new_user)
    token_expiration = datetime.utcnow() + timedelta(days=JWT_EXPIRATION)
    token_data = {"user_id": user_id, "email": email, "exp": token_expiration}
    token = jwt.encode(token_data, JWT_SECRET, algorithm="HS256")
    send_welcome_email(email, first_name, user_id, token)
    return jsonify({"message": "User created successfully", "token": token}), 200

# INSCRIPTION VIA GOOGLE OAUTH
@main.route('/google_signup')
def google_signup():
    flow = Flow.from_client_config(
        {"web": {"client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": [GOOGLE_REDIRECT_URI]}},
        scopes=["openid", "email", "profile"]
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')
    session['google_oauth_state'] = state
    return redirect(authorization_url)

@main.route('/google_callback')
def google_callback():
    state = request.args.get('state')
    flow = Flow.from_client_config(
        {"web": {"client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": [GOOGLE_REDIRECT_URI]}},
        scopes=["https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile", "openid"],
        state=state
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    authorization_response = request.url
    flow.fetch_token(authorization_response=authorization_response)
    credentials = flow.credentials
    id_token_jwt = credentials.id_token
    try:
        id_info = id_token.verify_oauth2_token(id_token_jwt, google_requests.Request(), GOOGLE_CLIENT_ID)
    except ValueError:
        return jsonify({"message": "Invalid token"}), 400
    user_email = id_info.get("email")
    google_id = id_info.get("sub")
    first_name = id_info.get("given_name", "")
    last_name = id_info.get("family_name", "")
    if not user_email or not google_id:
        return jsonify({"message": "Cannot retrieve user info"}), 400
    user = users_col.find_one({"email": user_email})
    if user:
        users_col.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    else:
        user = {"_id": str(uuid.uuid4()), "firstName": first_name, "lastName": last_name, "email": user_email, "password": None, "googleId": google_id, "created_at": datetime.utcnow(), "auth_method": "google", "preferences": {"theme": "light", "notifications": True}, "last_login": datetime.utcnow()}
        users_col.insert_one(user)
        send_welcome_email(user_email, first_name)
    token_expiration = datetime.utcnow() + timedelta(days=JWT_EXPIRATION)
    token_data = {"user_id": user["_id"], "email": user["email"], "exp": token_expiration}
    token = jwt.encode(token_data, JWT_SECRET, algorithm="HS256")
    return redirect(f"/chat?token={token}")

# Routes - Connexion
@main.route('/connexion', methods=['GET'])
def show_connexion():
    return render_template('connexion.html')

@main.route('/connexion', methods=['POST'])
def process_connexion():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    if not all([email, password]):
        return jsonify({"message": "All fields are required"}), 400
    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"message": "Invalid email or password"}), 401
    if user.get("auth_method") == "email" and not check_password_hash(user.get("password", ""), password):
        return jsonify({"message": "Invalid email or password"}), 401
    elif user.get("auth_method") != "email":
        return jsonify({"message": "This account uses Google to sign in. Please use the 'Sign in with Google' button."}), 400
    users_col.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    token_expiration = datetime.utcnow() + timedelta(days=JWT_EXPIRATION)
    token_data = {"user_id": user["_id"], "email": user["email"], "exp": token_expiration}
    token = jwt.encode(token_data, JWT_SECRET, algorithm="HS256")
    response = jsonify({"message": "Login successful", "token": token})
    response.set_cookie('finnToken', token, httponly=True, max_age=30*24*60*60)
    return response, 200

# Route pour la réinitialisation de mot de passe
@main.route('/reset-password', methods=['GET'])
def show_reset_password():
    return render_template('reset_password.html')

@main.route('/reset-password', methods=['POST'])
def request_password_reset():
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"message": "Email is required"}), 400
    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"message": "If an account exists for this email, a reset link has been sent."}), 200
    reset_token = jwt.encode({"user_id": user["_id"], "email": email, "exp": datetime.utcnow() + timedelta(hours=1)}, JWT_SECRET, algorithm="HS256")
    reset_link = f"{SITE_URL}/reset-password/{reset_token}"
    message = Mail(from_email=SENDER_EMAILS, to_emails=email, subject="Finn - Reset Your Password", html_content=f"<p>Hello,</p><p>You requested a password reset. Click the link below to reset your password:</p><p><a href='{reset_link}'>{reset_link}</a></p><p>This link will expire in 1 hour. If you didn’t request this, ignore this email.</p><p>Best regards,<br>The Finn Team</p>")
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEYS)
        sg.send(message)
    except Exception:
        return jsonify({"message": "Server error. Please try again later."}), 500
    return jsonify({"message": "If an account exists for this email, a reset link has been sent."}), 200

@main.route('/reset-password/<token>', methods=['GET'])
def show_new_password(token):
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user = users_col.find_one({"_id": data["user_id"], "email": data["email"]})
        if not user:
            return render_template('reset_password_error.html')
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return render_template('reset_password_error.html')
    return render_template('new_password.html')

@main.route('/reset-password/<token>', methods=['POST'])
def reset_password(token):
    data = request.get_json()
    new_password = data.get("newPassword")
    if not new_password:
        return jsonify({"message": "New password is required"}), 400
    if len(new_password) < 8:
        return jsonify({"message": "Password must be at least 8 characters long"}), 400
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user = users_col.find_one({"_id": data["user_id"], "email": data["email"]})
        if not user:
            return jsonify({"message": "Invalid or expired reset link"}), 400
        hashed_password = generate_password_hash(new_password)
        users_col.update_one({"_id": user["_id"]}, {"$set": {"password": hashed_password}})
        return jsonify({"message": "Password reset successfully"}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"message": "This reset link has expired"}), 400
    except jwt.InvalidTokenError:
        return jsonify({"message": "Invalid reset link"}), 400
    except Exception:
        return jsonify({"message": "Server error. Please try again later."}), 500

# Fonction de décorateur pour l'authentification
def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        if not token:
            token = request.cookies.get('finnToken')
        if not token:
            return jsonify({'message': 'Authentication required'}), 401
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            current_user = users_col.find_one({"_id": data['user_id']})
            if not current_user:
                return jsonify({'message': 'User not found'}), 401
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return jsonify({'message': 'Token expired or invalid. Please sign in again.'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# Route de déconnexion
@main.route('/logout')
def logout():
    response = make_response(redirect(url_for('main.index')))
    response.delete_cookie('finnToken')
    return response

@main.route('/chat')
@main.route('/chat/<conversation_id>')
def chat(conversation_id=None):
    token = request.args.get('token')
    response = None
    
    # Gérer le token dans l'URL (cas de connexion via Google)
    if token:
        response = make_response(redirect(url_for('main.chat')))
        response.set_cookie('finnToken', token, httponly=True, max_age=30*24*60*60)
        return response
    
    # Vérifier le token dans les cookies
    token = request.cookies.get('finnToken')
    if not token:
        return redirect(url_for('main.show_connexion'))
    
    try:
        # Décoder le token et vérifier l'utilisateur
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = data['user_id']
        current_user = users_col.find_one({"_id": user_id})
        
        if not current_user:
            response = make_response(redirect(url_for('main.show_connexion')))
            response.delete_cookie('finnToken')
            return response
    except:
        response = make_response(redirect(url_for('main.show_connexion')))
        response.delete_cookie('finnToken')
        return response
    
    # Vérifier l'accès à la conversation spécifique
    if conversation_id:
        conversation = conversations_col.find_one({"_id": conversation_id, "user_id": user_id})
        if not conversation:
            return redirect(url_for('main.chat'))
    
    # Récupérer les conversations récentes
    recent_conversations = {}
    user_conversations = conversations_col.find(
        {"user_id": user_id, "messages": {"$exists": True, "$not": {"$size": 0}}}, 
        {"title": 1, "created_at": 1}
    ).sort("created_at", -1).limit(10)
    
    for conv in user_conversations:
        recent_conversations[conv["_id"]] = {
            "title": conv.get("title", "New Conversation"), 
            "created_at": conv.get("created_at")
        }
    
    return render_template('chat.html', active_page='chat', conversation_id=conversation_id, recent_conversations=recent_conversations, current_user=current_user)

@main.route('/api/chat', methods=['POST'])
@auth_required
def process_chat(current_user):
    data = request.json
    message = data.get('message', '')
    conversation_id = data.get('conversation_id', '')
    user_id = current_user["_id"]
    
    # Vérifier si le message est vide
    if not message.strip():
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    # Cas d'une nouvelle conversation
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        new_conv = {
            "_id": conversation_id, 
            "messages": [], 
            "created_at": datetime.utcnow(), 
            "title": "New Conversation", 
            "user_id": user_id
        }
        conversations_col.insert_one(new_conv)
    # Cas d'une conversation existante
    else:
        conversation = conversations_col.find_one({"_id": conversation_id, "user_id": user_id})
        if not conversation:
            return jsonify({'error': 'Unauthorized access to this conversation'}), 403
    
    # Ajouter le message de l'utilisateur à la conversation
    user_message = {'role': 'user', 'content': message, 'timestamp': datetime.utcnow()}
    conversations_col.update_one({"_id": conversation_id}, {"$push": {"messages": user_message}})
    
    # Générer un titre pour la nouvelle conversation
    conversation = conversations_col.find_one({"_id": conversation_id})
    if len(conversation.get('messages', [])) == 1:
        try:
            title_prompt = f"Génère un titre TRÈS court (maximum 3 mots) qui résume cette requête: '{message}'"
            title_response = claude_handler.process_query(title_prompt)
            title = title_response.strip()
            title = re.sub(r'^["\'«]|["\'.!?:,;»]$', '', title).strip()
            if len(title) > 30 or not title or any(title.lower() == mot for mot in ["titre", "voici", "le titre", "résumé"]):
                raise Exception("Titre non pertinent généré")
        except Exception:
            words = message.split()[:3]
            title = " ".join(words)
            if len(title) > 25:
                title = title[:22]
        conversations_col.update_one({"_id": conversation_id}, {"$set": {"title": title}})
    
    # Traiter la réponse de l'assistant
    response = claude_handler.process_query(message)
    assistant_message = response
    assistant_msg = {'role': 'assistant', 'content': assistant_message, 'timestamp': datetime.utcnow()}
    conversations_col.update_one({"_id": conversation_id}, {"$push": {"messages": assistant_msg}})
    
    return jsonify({'response': assistant_message, 'conversation_id': conversation_id})

@main.route('/api/conversations', methods=['GET'])
@auth_required
def get_conversations(current_user):
    user_id = current_user["_id"]
    # Assurez-vous que le tri est explicite et cohérent
    user_conversations = list(conversations_col.find(
        {"user_id": user_id, "messages": {"$exists": True, "$not": {"$size": 0}}}, 
        {"title": 1, "created_at": 1}
    ).sort("created_at", -1))
    
    conversations_dict = {}
    for conv in user_conversations:
        conv_id = conv["_id"]
        # Formatez la date de création de manière cohérente
        created_at = conv.get('created_at')
        if isinstance(created_at, datetime):
            created_at_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
            created_at_timestamp = created_at.timestamp()  # Ajouter un timestamp pour le tri côté client
        else:
            created_at_str = str(created_at)
            created_at_timestamp = 0
            
        conversations_dict[conv_id] = {
            'title': conv.get('title', 'New Conversation'), 
            'created_at': created_at_str,
            'timestamp': created_at_timestamp,  # Pour trier côté client
            'message_count': conv.get('message_count', 0)
        }
    
    return jsonify(conversations_dict)


@main.route('/api/conversations/<conversation_id>/messages', methods=['GET'])
@auth_required
def get_conversation_messages(current_user, conversation_id):
    """Récupère tous les messages d'une conversation spécifique."""
    user_id = current_user["_id"]
    conversation = conversations_col.find_one({"_id": conversation_id, "user_id": user_id})
    
    if not conversation:
        return jsonify({'error': 'Conversation not found or access denied'}), 404
    
    messages = conversation.get('messages', [])
    
    # Nettoyer les données pour la sérialisation JSON
    for msg in messages:
        if isinstance(msg.get('timestamp'), datetime):
            msg['timestamp'] = msg['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    
    return jsonify({'messages': messages})

@main.route('/api/conversations/new', methods=['POST'])
@auth_required
def new_conversation(current_user):
    return jsonify({'redirect': '/chat', 'status': 'redirect'})

@main.route('/api/conversations/<conversation_id>', methods=['DELETE'])
@auth_required
def delete_conversation(current_user, conversation_id):
    user_id = current_user["_id"]
    result = conversations_col.delete_one({"_id": conversation_id, "user_id": user_id})
    if result.deleted_count > 0:
        return jsonify({'status': 'deleted'})
    else:
        return jsonify({'status': 'error', 'message': 'Conversation introuvable'}), 404

@main.route('/notifications')
def notifications():
    token = request.cookies.get('finnToken')
    if not token:
        return redirect(url_for('main.show_connexion'))
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = data['user_id']
        current_user = users_col.find_one({"_id": user_id})
        if not current_user:
            response = make_response(redirect(url_for('main.show_connexion')))
            response.delete_cookie('finnToken')
            return response
    except:
        response = make_response(redirect(url_for('main.show_connexion')))
        response.delete_cookie('finnToken')
        return response
    return render_template('notifications.html', active_page='notifications', current_user=current_user)

@main.route('/news')
def news():
    token = request.cookies.get('finnToken')
    if not token:
        return redirect(url_for('main.show_connexion'))
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = data['user_id']
        current_user = users_col.find_one({"_id": user_id})
        if not current_user:
            response = make_response(redirect(url_for('main.show_connexion')))
            response.delete_cookie('finnToken')
            return response
            
        # Récupérer les conversations récentes pour afficher dans la barre latérale
        recent_conversations = get_user_conversations(user_id)
            
    except:
        response = make_response(redirect(url_for('main.show_connexion')))
        response.delete_cookie('finnToken')
        return response
        
    company = request.args.get('company', None)
    categories = ['Stock Markets', 'Cryptocurrencies', 'Macroeconomics', 'Commodities', 'Financial Tech', 'Financial Regulation', 'Forex & Currencies', 'Technical Analysis']
    category = request.args.get('category', 'Stock Markets')
    if company:
        news_data = news_handler.get_company_news(company)
        view_type = 'company'
    else:
        news_data = news_handler.get_global_news(category=category)
        view_type = 'category'
    if isinstance(news_data, str):
        error_message = news_data
        news_data = []
    else:
        error_message = None
    return render_template('news.html', active_page='news', news_data=news_data, categories=categories, current_category=category, company_name=company, view_type=view_type, error_message=error_message, current_user=current_user, recent_conversations=recent_conversations)

# ROUTE POUR A SIMULATION
@main.route('/simulator')

def simulator():

    # Vérification de l'authentification

    token = request.cookies.get('finnToken')

    if not token:

        logger.info("Tentative d'accès à la simulation sans token")

        return redirect(url_for('main.show_connexion'))

    

    try:

        # Décodage du token et récupération des informations utilisateur

        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])

        user_id = data['user_id']

        current_user = users_col.find_one({"_id": user_id})

        

        if not current_user:

            logger.warning(f"Utilisateur avec ID {user_id} non trouvé dans la base de données")

            response = make_response(redirect(url_for('main.show_connexion')))

            response.delete_cookie('finnToken')

            return response

        

        logger.info(f"Chargement du simulateur pour l'utilisateur {user_id}")

        return render_template('simulation.html', active_page='simulator', current_user=current_user)

        

    except jwt.ExpiredSignatureError:

        logger.warning("Token expiré lors de l'accès au simulateur")

        response = make_response(redirect(url_for('main.show_connexion')))

        response.delete_cookie('finnToken')

        return response

    except jwt.InvalidTokenError:

        logger.warning("Token invalide lors de l'accès au simulateur")

        response = make_response(redirect(url_for('main.show_connexion')))

        response.delete_cookie('finnToken')

        return response

    except Exception as e:

        logger.error(f"Erreur inattendue lors de l'accès au simulateur: {str(e)}")

        response = make_response(redirect(url_for('main.show_connexion')))

        response.delete_cookie('finnToken')

        return response

@main.route('/api/simulator/search', methods=['POST'])

@auth_required

def search_assets():

    """Search for assets based on a query."""

    data = request.get_json()

    query = data.get('query', '')

    if not query:

        return jsonify({'error': 'Query is required'}), 400

    try:

        logger.info(f"Recherche d'actifs pour: {query}")

        results = trading_simulator.search_assets(query)

        logger.info(f"Résultats trouvés: {len(results)}")

        return jsonify(results), 200

    except Exception as e:

        logger.error(f"Erreur lors de la recherche d'actifs: {str(e)}")

        return jsonify({'error': f'Search error: {str(e)}'}), 500

@main.route('/api/simulator/new-investment', methods=['POST'])

@auth_required

def new_investment():

    """Simulate a new investment."""

    data = request.get_json()

    symbol = data.get('symbol')

    amount = float(data.get('amount', 0))

    horizon = int(data.get('horizon', 0))

    risk_level = data.get('risk_level', 'moderate')

    asset_type = data.get('asset_type', 'crypto')

    logger.info(f"Nouvel investissement - Symbole: {symbol}, Montant: {amount}, Horizon: {horizon}, Risque: {risk_level}")

    

    if not all([symbol, amount > 0, horizon > 0]):

        logger.warning(f"Paramètres invalides - Symbole: {symbol}, Montant: {amount}, Horizon: {horizon}")

        return jsonify({'error': 'Invalid input parameters'}), 400

    try:

        # Convertir l'horizon en jours (comme attendu par simulator.py)

        horizon_days = horizon * 30

        results = trading_simulator.new_investment(symbol, amount, horizon_days, risk_level, asset_type, use_ai=True)

        logger.info(f"Simulation réussie pour {symbol}")

        return jsonify(results), 200

    except ValueError as e:

        logger.error(f"Erreur lors de la simulation: {str(e)}")

        return jsonify({'error': str(e)}), 400

    except Exception as e:

        logger.error(f"Erreur inattendue lors de la simulation: {str(e)}")

        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@main.route('/api/simulator/analyze-position', methods=['POST'])

@auth_required

def analyze_position():

    """Analyze an existing portfolio position."""

    data = request.get_json()

    symbol = data.get('symbol')

    quantity = float(data.get('quantity', 0))

    avg_purchase_price = float(data.get('avg_purchase_price', 0))

    horizon = int(data.get('horizon', 0))

    asset_type = data.get('asset_type', 'crypto')

    logger.info(f"Analyse portfolio - Symbole: {symbol}, Quantité: {quantity}, Prix d'achat: {avg_purchase_price}, Horizon: {horizon}")

    

    if not all([symbol, quantity > 0, avg_purchase_price > 0, horizon > 0]):

        logger.warning(f"Paramètres invalides - Symbole: {symbol}, Quantité: {quantity}, Prix: {avg_purchase_price}, Horizon: {horizon}")

        return jsonify({'error': 'Invalid input parameters'}), 400

    try:

        # Convertir l'horizon en jours (comme attendu par simulator.py)

        horizon_days = horizon * 30

        results = trading_simulator.analyze_portfolio(symbol, quantity, avg_purchase_price, horizon_days, asset_type, use_ai=True)

        logger.info(f"Analyse réussie pour {symbol}")

        return jsonify(results), 200

    except ValueError as e:

        logger.error(f"Erreur lors de l'analyse: {str(e)}")

        return jsonify({'error': str(e)}), 400

    except Exception as e:

        logger.error(f"Erreur inattendue lors de l'analyse: {str(e)}")

        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500



@main.route('/privacy')
def privacy():
    return render_template('privacy.html')

@main.route('/terms')
def terms():
    return render_template('terms.html')