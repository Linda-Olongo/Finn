import os
import re
import uuid
import html
import jwt
import secrets
from functools import wraps
from datetime import datetime, timedelta
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from werkzeug.security import generate_password_hash

from flask import (
    render_template, Blueprint, request, jsonify,
    session, redirect, url_for,make_response
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
from app.nlp.GeminiHandler import GeminiHandler

# Clé secrète pour JWT 
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_EXPIRATION = 30  # en jours

# Chargé SendGrid pour l'envoie des liens de reset mot de passe
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")

# 1) Configuration de MongoDB 
MONGO_URI = os.getenv("MONGO_URI") 
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["finn"]
users_col = db["users"] # Nouvelle collection pour les utilisateurs
conversations_col = db["conversations"]  # Nouvelle collection pour les conversations

# 2) Configuration de Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET") 
BASE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# Détection de l'environnement Render
if os.getenv("RENDER") == "true" or os.getenv("IS_PRODUCTION") == "true":
    # On est sur Render ou en production
    SITE_URL = os.getenv("RENDER_EXTERNAL_URL", "https://votre-app.onrender.com")
    GOOGLE_REDIRECT_URI = f"{SITE_URL}/google_callback"
else:
    # On est en développement local
    GOOGLE_REDIRECT_URI = BASE_REDIRECT_URI

main = Blueprint('main', __name__)
news_handler = NewsHandler()
gemini_handler = GeminiHandler()

# Dictionnaire pour stocker les conversations
# conversations = {} 

# PAGE D'ACCUEIL
@main.route('/')
def index():
    # Afficher la page loading.html comme page d'accueil
    return render_template('loading.html')


# INSCRIPTION (MANUELLE) - GET => Formulaire, POST => Création
@main.route('/inscription', methods=['GET'])
def show_inscription():
    # Renvoie le template inscription.html
    return render_template('inscription.html')

@main.route('/inscription', methods=['POST'])
def process_inscription():
    # Récupération des données JSON depuis le fetch() du front
    data = request.get_json()
    first_name = data.get("firstName")
    last_name = data.get("lastName")
    email = data.get("email")
    password = data.get("password")

    if not all([first_name, last_name, email, password]):
        return jsonify({"message": "All fields are required"}), 400

    # Vérifier si l'utilisateur existe déjà
    existing_user = users_col.find_one({"email": email})
    if existing_user:
        return jsonify({"message": "User already exists"}), 400

    # Hasher le mot de passe
    hashed_pw = generate_password_hash(password)

    # Créer le nouveau user
    new_user = {
        "_id": str(uuid.uuid4()),
        "firstName": first_name,
        "lastName": last_name,
        "email": email,
        "password": hashed_pw,
        "created_at": datetime.utcnow(),
        "auth_method": "email",
        "preferences": {
            "theme": "light",
            "notifications": True
        },
        "last_login": None
    }
    users_col.insert_one(new_user)

    return jsonify({"message": "User created successfully"}), 200


# INSCRIPTION VIA GOOGLE OAUTH
@main.route('/google_signup')
def google_signup():
    
    """
    1) Initialise le flow OAuth Google
    2) Redirige l'utilisateur vers l'écran de consentement
    """
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [GOOGLE_REDIRECT_URI]
            }
        },
        scopes=[
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "openid",
        ]
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )

    # On stocke 'state' dans la session pour vérification ultérieure
    session['google_oauth_state'] = state

    return redirect(authorization_url)


@main.route('/google_callback')
def google_callback():
    """
    1) Google redirige ici après le consentement
    2) On récupère le token, on vérifie l'user
    3) On l'insère en BDD si pas encore existant
    4) On le connecte (session) et on redirige
    """
    state = request.args.get('state') #  Récupérer l'état depuis la requête au lieu de la session
    
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [GOOGLE_REDIRECT_URI]
            }
        },
        scopes=[
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "openid",
        ],
        state=state
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI

    # Récupérer le code envoyé par Google dans la querystring
    authorization_response = request.url

    # Échanger le "code" contre un "token"
    flow.fetch_token(authorization_response=authorization_response)

    # Récupérer l'ID token
    credentials = flow.credentials
    id_token_jwt = credentials.id_token  # Jeton ID
    try:
        id_info = id_token.verify_oauth2_token(
            id_token_jwt,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )
    except ValueError:
        return jsonify({"message": "Invalid token"}), 400

    # Extraire l'email & Google ID
    user_email = id_info.get("email")
    google_id = id_info.get("sub")
    first_name = id_info.get("given_name", "")
    last_name = id_info.get("family_name", "")

    if not user_email or not google_id:
        return jsonify({"message": "Cannot retrieve user info"}), 400

    # Vérifier si l'utilisateur existe déjà en base
    user = users_col.find_one({"email": user_email})
    if user:
        # Mettre à jour last_login
        users_col.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
    else:
        # Créer le user
        user = {
            "_id": str(uuid.uuid4()),
            "firstName": first_name,
            "lastName": last_name,
            "email": user_email,
            "password": None,  # pas nécessaire pour Google
            "googleId": google_id,
            "created_at": datetime.utcnow(),
            "auth_method": "google",
            "preferences": {
                "theme": "light",
                "notifications": True
            },
            "last_login": datetime.utcnow()
        }
        users_col.insert_one(user)

    # Créer un token JWT valide pour 30 jours
    token_expiration = datetime.utcnow() + timedelta(days=JWT_EXPIRATION)
    token_data = {
        "user_id": user["_id"],
        "email": user["email"],
        "exp": token_expiration
    }
    token = jwt.encode(token_data, JWT_SECRET, algorithm="HS256")

    # Rediriger avec le token dans l'URL (vous pouvez aussi utiliser un cookie)
    return redirect(f"/chat?token={token}")

# Routes - Connxion
@main.route('/connexion', methods=['GET'])
def show_connexion():
    # Afficher la page de connexion
    return render_template('connexion.html')

# Route pour traiter la connexion (POST)
@main.route('/connexion', methods=['POST'])
def process_connexion():
    # Récupération des données
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not all([email, password]):
        return jsonify({"message": "All fields are required"}), 400

    # Vérifier si l'utilisateur existe
    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"message": "Invalid email or password"}), 401

    # Vérifier le mot de passe (uniquement pour les comptes créés par email)
    if user.get("auth_method") == "email":
        if not check_password_hash(user.get("password", ""), password):
            return jsonify({"message": "Invalid email or password"}), 401
    else:
        # Si l'utilisateur s'est inscrit avec Google et essaie de se connecter par email
        return jsonify({"message": "This account uses Google to sign in. Please use the 'Sign in with Google' button."}), 400

    # Mettre à jour la date de dernière connexion
    users_col.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )

    # Créer un token JWT valide pour 30 jours
    token_expiration = datetime.utcnow() + timedelta(days=JWT_EXPIRATION)
    token_data = {
        "user_id": user["_id"],
        "email": user["email"],
        "exp": token_expiration
    }
    token = jwt.encode(token_data, JWT_SECRET, algorithm="HS256")
    
    # Créer la réponse JSON
    response = jsonify({
        "message": "Login successful",
        "token": token
    })
    
    # Définir le cookie avec le token
    response.set_cookie('finnToken', token, httponly=True, max_age=30*24*60*60)  # 30 jours
    
    return response, 200
    
    
#Route pour la réinitialisation de mot de passe


# Fonction de décorateur pour l'authentification (
def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        # Vérifier si le token est présent dans l'en-tête
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        # Si pas de token dans l'en-tête, vérifier dans les cookies
        if not token:
            token = request.cookies.get('finnToken')
        
        if not token:
            return jsonify({'message': 'Authentication required'}), 401
        
        try:
            # Décoder le token
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            current_user = users_col.find_one({"_id": data['user_id']})
            
            if not current_user:
                return jsonify({'message': 'User not found'}), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expired. Please sign in again.'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token. Please sign in again.'}), 401
            
        # Passer l'utilisateur à la fonction décorée
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
    # Récupérer le token s'il est passé dans l'URL
    token = request.args.get('token')
    
    # Si on a un token dans l'URL, le stocker dans un cookie
    response = None
    if token:
        response = make_response(redirect(url_for('main.chat')))
        response.set_cookie('finnToken', token, httponly=True, max_age=30*24*60*60)  # 30 jours
        return response

    # Vérifier si l'utilisateur est authentifié, sinon rediriger vers la page de connexion
    token = request.cookies.get('finnToken')
    if not token:
        return redirect(url_for('main.show_connexion'))
    
    # Vérifier le token et récupérer les informations utilisateur
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = data['user_id']
        
        # Récupérer les informations complètes de l'utilisateur
        current_user = users_col.find_one({"_id": user_id})
        if not current_user:
            # User introuvable, rediriger vers la connexion
            response = make_response(redirect(url_for('main.show_connexion')))
            response.delete_cookie('finnToken')
            return response
            
    except:
        # Token invalide, rediriger vers la connexion
        response = make_response(redirect(url_for('main.show_connexion')))
        response.delete_cookie('finnToken')
        return response
    
    # Vérifier si la conversation existe et appartient à l'utilisateur
    if conversation_id:
        conversation = conversations_col.find_one({
            "_id": conversation_id,
            "user_id": user_id
        })
        if not conversation:
            # Conversation introuvable ou n'appartenant pas à l'utilisateur
            return redirect(url_for('main.chat'))
    
    # Récupérer les conversations récentes pour la barre latérale
    recent_conversations = {}
    user_conversations = conversations_col.find(
        {"user_id": user_id},
        {"title": 1, "created_at": 1}  # Ne récupérer que le titre et la date
    ).sort("created_at", -1).limit(10)  # Limiter aux 10 plus récentes
    
    for conv in user_conversations:
        recent_conversations[conv["_id"]] = {
            "title": conv.get("title", "New Conversation"),
            "created_at": conv.get("created_at")
        }
    
    return render_template(
        'chat.html', 
        active_page='chat',
        conversation_id=conversation_id,
        recent_conversations=recent_conversations,
        current_user=current_user
    )

@main.route('/api/chat', methods=['POST'])
@auth_required
def process_chat(current_user):
    data = request.json
    message = data.get('message', '')
    conversation_id = data.get('conversation_id', '')
    user_id = current_user["_id"]
    
    # Vérifier si l'ID de conversation est valide et que la conversation existe
    if not conversation_id:
        # Créer une nouvelle conversation uniquement lorsqu'un message est envoyé
        conversation_id = str(uuid.uuid4())
        new_conv = {
            "_id": conversation_id,
            "messages": [],
            "created_at": datetime.utcnow(),
            "title": "New Conversation",
            "user_id": user_id
        }
        conversations_col.insert_one(new_conv)
    else:
        # Vérifier que la conversation existe et appartient à l'utilisateur
        conversation = conversations_col.find_one({
            "_id": conversation_id,
            "user_id": user_id
        })
        if not conversation:
            return jsonify({'error': 'Accès non autorisé à cette conversation'}), 403
    
    # Ajouter le message de l'utilisateur à la conversation
    user_message = {
        'role': 'user',
        'content': message,
        'timestamp': datetime.utcnow()
    }
    
    conversations_col.update_one(
        {"_id": conversation_id},
        {"$push": {"messages": user_message}}
    )
    
    # Si c'est le premier message, générer un titre intelligent avec Gemini
    conversation = conversations_col.find_one({"_id": conversation_id})
    if len(conversation.get('messages', [])) == 1:
        try:
            # Demander à Gemini un titre court
            title_prompt = f"Génère un titre TRÈS court (maximum 3 mots) qui résume cette requête: '{message}'"
            title_response = gemini_handler.process_query(title_prompt)
            title = title_response['message'].strip()
            
            # Nettoyage du titre
            title = re.sub(r'^["\'«]|["\'.!?:,;»]$', '', title).strip()
            
            # Vérification supplémentaire de longueur
            if len(title) > 30:
                title = title[:28]
                
            # Vérifier que le titre est pertinent
            mots_non_pertinents = ["titre", "voici", "le titre", "résumé"]
            if not title or any(title.lower() == mot for mot in mots_non_pertinents):
                raise Exception("Titre non pertinent généré")
                
        except Exception as e:
            # Fallback avec limite stricte (3 mots maximum)
            words = message.split()[:3]
            title = " ".join(words)
            if len(title) > 25:
                title = title[:22] 
        
        conversations_col.update_one(
            {"_id": conversation_id},
            {"$set": {"title": title}}
        )
    
    # Traiter la requête avec GeminiHandler
    response = gemini_handler.process_query(message)
    assistant_message = response['message']
    
    # Ajouter la réponse à la conversation
    assistant_msg = {
        'role': 'assistant',
        'content': assistant_message,
        'timestamp': datetime.utcnow()
    }
    
    conversations_col.update_one(
        {"_id": conversation_id},
        {"$push": {"messages": assistant_msg}}
    )
    
    return jsonify({
        'response': assistant_message,
        'conversation_id': conversation_id
    })
    

# Récupérer les messages d'une conversation spécifique
@main.route('/api/conversations', methods=['GET'])
@auth_required
def get_conversations(current_user):
    user_id = current_user["_id"]
    
    # Récupérer toutes les conversations de l'utilisateur ayant au moins un message
    user_conversations = list(conversations_col.find({
        "user_id": user_id,
        "messages": {"$exists": True, "$not": {"$size": 0}}  # Ensure messages array exists and is not empty
    }, {
        "messages": 0  # Exclure les messages pour accélérer la récupération
    }).sort("created_at", -1))  # Trier par date (plus récentes d'abord)
    
    # Formatter la réponse
    conversations_dict = {}
    for conv in user_conversations:
        conv_id = conv["_id"]
        conversations_dict[conv_id] = {
            'title': conv.get('title', 'New Conversation'),
            'created_at': conv.get('created_at').strftime("%Y-%m-%d %H:%M:%S") if isinstance(conv.get('created_at'), datetime) else conv.get('created_at'),
            'message_count': conv.get('message_count', 0)
        }
    
    return jsonify(conversations_dict)

@main.route('/api/conversations/new', methods=['POST'])
@auth_required
def new_conversation(current_user):
    # Ne pas créer de conversation dans MongoDB, juste rediriger
    return jsonify({
        'redirect': '/chat',
        'status': 'redirect'
    })

@main.route('/api/conversations/<conversation_id>', methods=['DELETE'])
@auth_required
def delete_conversation(current_user, conversation_id):
    user_id = current_user["_id"]
    
    # Vérifier si la conversation existe et appartient à l'utilisateur
    result = conversations_col.delete_one({
        "_id": conversation_id,
        "user_id": user_id
    })
    
    if result.deleted_count > 0:
        return jsonify({'status': 'deleted'})
    else:
        return jsonify({'status': 'error', 'message': 'Conversation introuvable'}), 404
    
@main.route('/notifications')
def notifications():
    # Vérifier si l'utilisateur est authentifié
    token = request.cookies.get('finnToken')
    if not token:
        return redirect(url_for('main.show_connexion'))
    
    # Vérifier le token et récupérer les informations utilisateur
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = data['user_id']
        
        # Récupérer les informations complètes de l'utilisateur
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
    # Vérifier si l'utilisateur est authentifié
    token = request.cookies.get('finnToken')
    if not token:
        return redirect(url_for('main.show_connexion'))
    
    # Vérifier le token et récupérer les informations utilisateur
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = data['user_id']
        
        # Récupérer les informations complètes de l'utilisateur
        current_user = users_col.find_one({"_id": user_id})
        if not current_user:
            response = make_response(redirect(url_for('main.show_connexion')))
            response.delete_cookie('finnToken')
            return response
            
    except:
        response = make_response(redirect(url_for('main.show_connexion')))
        response.delete_cookie('finnToken')
        return response
    
    # Vérifier si une recherche d'entreprise est demandée
    company = request.args.get('company', None)
    
    # Définir les catégories
    categories = [
        'Stock Markets', 
        'Cryptocurrencies', 
        'Macroeconomics', 
        'Commodities', 
        'Financial Tech', 
        'Financial Regulation', 
        'Forex & Currencies', 
        'Technical Analysis'
    ]
    
    # Catégorie par défaut
    category = request.args.get('category', 'Stock Markets')
    
    # Récupérer les nouvelles
    if company:
        # Si une entreprise est spécifiée, récupérer ses nouvelles
        news_data = news_handler.get_company_news(company)
        view_type = 'company'
    else:
        # Récupérer les nouvelles par catégorie
        news_data = news_handler.get_global_news(category=category)
        view_type = 'category'
    
    # Gérer les cas où news_data est une chaîne d'erreur
    if isinstance(news_data, str):
        error_message = news_data
        news_data = []
    else:
        error_message = None
    
    return render_template(
        'news.html',
        active_page='news',
        news_data=news_data,
        categories=categories,
        current_category=category,
        company_name=company,
        view_type=view_type,
        error_message=error_message,
        current_user=current_user
    )
@main.route('/simulator')
def simulator():
    # Vérifier si l'utilisateur est authentifié
    token = request.cookies.get('finnToken')
    if not token:
        return redirect(url_for('main.show_connexion'))
    
    # Vérifier le token et récupérer les informations utilisateur
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = data['user_id']
        
        # Récupérer les informations complètes de l'utilisateur
        current_user = users_col.find_one({"_id": user_id})
        if not current_user:
            response = make_response(redirect(url_for('main.show_connexion')))
            response.delete_cookie('finnToken')
            return response
            
    except:
        response = make_response(redirect(url_for('main.show_connexion')))
        response.delete_cookie('finnToken')
        return response
    
    return render_template('simulation.html', active_page='simulator', current_user=current_user)

# POLITIQUE DE CONFIDENTIALITÉ
@main.route('/privacy')
def privacy():
    # Afficher la page privacy.html
    return render_template('privacy.html')

# CONDITIONS D'UTILISATION
@main.route('/terms')
def terms():
    # Afficher la page terms.html
    return render_template('terms.html')

# ROUTE POUR RÉNITIALISER LE MOT DE PASSE
@main.route('/reset-password', methods=['GET'])
def show_reset_password():
    return render_template('reset_password.html')


@main.route('/reset-password', methods=['POST'])
def request_password_reset():
    data = request.get_json()
    email = data.get("email")

    if not email:
        return jsonify({"message": "Email is required"}), 400

    # Vérifier si l'utilisateur existe
    user = users_col.find_one({"email": email})
    if not user:
        # On renvoie une réponse vague pour des raisons de sécurité
        return jsonify({"message": "If an account exists for this email, a reset link has been sent."}), 200

    # Générer un token de réinitialisation (valide 1 heure)
    reset_token = jwt.encode(
        {
            "user_id": user["_id"],
            "email": email,
            "exp": datetime.utcnow() + timedelta(hours=1)
        },
        JWT_SECRET,
        algorithm="HS256"
    )

    # Construire le lien de réinitialisation
    reset_link = f"{SITE_URL}/reset-password/{reset_token}"

    # Envoyer l'email via SendGrid
    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=email,
        subject="Finn - Reset Your Password",
        html_content=f"""
        <p>Hello,</p>
        <p>You requested a password reset. Click the link below to reset your password:</p>
        <p><a href="{reset_link}">{reset_link}</a></p>
        <p>This link will expire in 1 hour. If you didn’t request this, ignore this email.</p>
        <p>Best regards,<br>The Finn Team</p>
        """
    )

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(f"Email sent: {response.status_code}")
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return jsonify({"message": "Server error. Please try again later."}), 500

    return jsonify({"message": "If an account exists for this email, a reset link has been sent."}), 200


#Route pour Afficher le Formulaire de Nouveau Mot de Passe
@main.route('/reset-password/<token>', methods=['GET'])
def show_new_password(token):
    try:
        # Vérifier le token
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user = users_col.find_one({"_id": data["user_id"], "email": data["email"]})
        if not user:
            return render_template('reset_password_error.html')
    except jwt.ExpiredSignatureError:
        return render_template('reset_password_error.html')
    except jwt.InvalidTokenError:
        return render_template('reset_password_error.html')

    return render_template('new_password.html')

#Route pour Mettre à Jour le Mot de Passe
@main.route('/reset-password/<token>', methods=['POST'])
def reset_password(token):
    data = request.get_json()
    new_password = data.get("newPassword")

    if not new_password:
        return jsonify({"message": "New password is required"}), 400

    if len(new_password) < 8:
        return jsonify({"message": "Password must be at least 8 characters long"}), 400

    try:
        # Vérifier le token
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user = users_col.find_one({"_id": data["user_id"], "email": data["email"]})
        if not user:
            return jsonify({"message": "Invalid or expired reset link"}), 400

        # Mettre à jour le mot de passe dans MongoDB
        hashed_password = generate_password_hash(new_password)
        users_col.update_one(
            {"_id": user["_id"]},
            {"$set": {"password": hashed_password}}
        )

        return jsonify({"message": "Password reset successfully"}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"message": "This reset link has expired"}), 400
    except jwt.InvalidTokenError:
        return jsonify({"message": "Invalid reset link"}), 400
    except Exception as e:
        print(f"Error resetting password: {str(e)}")
        return jsonify({"message": "Server error. Please try again later."}), 500