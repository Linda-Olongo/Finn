# app/routes.py
from flask import render_template, Blueprint, request, jsonify, session, redirect, url_for
from app.data.news_handler import NewsHandler # Import des actualités
from app.nlp.GeminiHandler import GeminiHandler  # Import du GeminiHandler
import uuid
import datetime
import re
import html

main = Blueprint('main', __name__)
news_handler = NewsHandler()
gemini_handler = GeminiHandler()

# Dictionnaire pour stocker les conversations
conversations = {}

@main.route('/')
def index():
    # Rediriger vers le chat comme page d'accueil
    return redirect(url_for('main.chat'))


@main.route('/chat')
@main.route('/chat/<conversation_id>')
def chat(conversation_id=None):
    # Ne créer une nouvelle conversation que si l'ID est explicitement invalide
    # et pas automatiquement à chaque visite
    if conversation_id and conversation_id in conversations:
        # Conversation existante valide
        pass
    elif conversation_id:
        # ID non valide, rediriger vers la page de chat sans ID
        return redirect(url_for('main.chat'))
    
    # Si pas d'ID fourni, ne pas créer de conversation automatiquement
    # La conversation sera créée lors du premier message
    
    # Récupérer les conversations récentes pour le menu latéral
    recent_conversations = {k: v for k, v in sorted(
        conversations.items(), 
        key=lambda item: item[1]['created_at'], 
        reverse=True
    )}
    
    return render_template(
        'chat.html', 
        active_page='chat',
        conversation_id=conversation_id,
        recent_conversations=recent_conversations
    )

@main.route('/api/chat', methods=['POST'])
def process_chat():
    data = request.json
    message = data.get('message', '')
    conversation_id = data.get('conversation_id', '')
    
    # Vérifier si l'ID de conversation est valide et que la conversation existe
    # Sinon, créer une nouvelle conversation
    if not conversation_id or conversation_id not in conversations:
        conversation_id = str(uuid.uuid4())
        conversations[conversation_id] = {
            'messages': [],
            'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'title': 'New Conversation'  # Titre temporaire
        }
    
    message_safe = message
    
    # Ajouter le message de l'utilisateur à la conversation
    conversations[conversation_id]['messages'].append({
        'role': 'user',
        'content': message_safe,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Si c'est le premier message, générer un titre intelligent avec Gemini
    if len(conversations[conversation_id]['messages']) == 1:
        try:
            # Demander à Gemini un titre court
            title_prompt = f"Génère un titre TRÈS court (maximum 3 mots) qui résume cette requête: '{message}'"
            title_response = gemini_handler.process_query(title_prompt)
            title = title_response['message'].strip()
            
            # Nettoyage du titre
            title = re.sub(r'^["\'«]|["\'.!?:,;»]$', '', title).strip()
            
            # Vérification supplémentaire de longueur
            if len(title) > 25:  # Limite stricte à 25 caractères
                title = title[:22]
                
            # Vérifier que le titre est pertinent
            mots_non_pertinents = ["titre", "voici", "le titre", "résumé"]
            if not title or any(title.lower() == mot for mot in mots_non_pertinents):
                raise Exception("Titre non pertinent généré")
                
        except Exception as e:
            # Fallback avec limite stricte (3 mots maximum)
            words = message_safe.split()[:3]
            title = " ".join(words)
            if len(title) > 25:
                title = title[:22] 
        
        conversations[conversation_id]['title'] = title
    
    # Traiter la requête avec GeminiHandler
    response = gemini_handler.process_query(message)
    
    # Préserver le formatage de la réponse
    assistant_message = response['message']
    
    # Ajouter la réponse à la conversation
    conversations[conversation_id]['messages'].append({
        'role': 'assistant',
        'content': assistant_message,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    return jsonify({
        'response': assistant_message,
        'conversation_id': conversation_id
    })

# NOUVELLE ROUTE: récupérer les messages d'une conversation spécifique
@main.route('/api/conversations/<conversation_id>/messages', methods=['GET'])
def get_conversation_messages(conversation_id):
    """
    Récupère tous les messages d'une conversation spécifique.
    """
    # Vérifier si la conversation existe
    if conversation_id not in conversations:
        return jsonify({'error': 'Conversation not found'}), 404
    
    # Récupérer les messages de la conversation
    messages = conversations[conversation_id].get('messages', [])
    
    return jsonify({'messages': messages})

@main.route('/api/conversations', methods=['GET'])
def get_conversations():
    # Retourner uniquement les conversations non vides, triées par date
    non_empty_conversations = {
        k: v for k, v in conversations.items() 
        if 'messages' in v and len(v['messages']) > 0
    }
    sorted_conversations = {k: v for k, v in sorted(
        non_empty_conversations.items(), 
        key=lambda item: item[1]['created_at'], 
        reverse=True
    )}
    return jsonify(sorted_conversations)

@main.route('/api/conversations/new', methods=['POST'])
def new_conversation():
    # Créer une nouvelle conversation
    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = {
        'messages': [],
        'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'title': 'New Conversation'
    }
    return jsonify({
        'conversation_id': conversation_id,
        'status': 'created'
    })

@main.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    # Supprimer une conversation
    if conversation_id in conversations:
        del conversations[conversation_id]
        return jsonify({'status': 'deleted'})
    return jsonify({'status': 'error', 'message': 'Conversation not found'}), 404

# ROUTE POUR LES NOTIFICATIONS
@main.route('/notifications')
def notifications():
    return render_template('notifications.html', active_page='notifications')

# ROUTE POUR LES ACTUALITÉS
@main.route('/news')
def news():
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
        error_message=error_message
    )

# ROUTE POUR LES SIMULATIONS DE TRADING
@main.route('/simulator')
def simulator():
    return render_template('simulator.html', active_page='simulator')