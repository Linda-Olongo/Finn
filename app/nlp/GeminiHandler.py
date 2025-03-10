import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import langdetect
import google.generativeai as genai
from google.api_core import exceptions
from dateutil.parser import parse
from data.api_fetcher import DataCollector

# Charger les variables d'environnement
load_dotenv()

class ConversationMemory:
    """Gestion manuelle de l'historique des conversations."""
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
        self.repeated_queries = {}

    def add_interaction(self, query: str, response: str):
        self.history.append({"user": query, "assistant": response})
        if len(self.history) > self.max_history:
            self.history.pop(0)
        query_normalized = query.lower().strip()
        self.repeated_queries[query_normalized] = self.repeated_queries.get(query_normalized, 0) + 1

    def is_repeated_query(self, query: str) -> bool:
        return query.lower().strip() in self.repeated_queries

    def get_repetition_count(self, query: str) -> int:
        return self.repeated_queries.get(query.lower().strip(), 0)

    def get_recent_interactions(self, count=3) -> List[Dict[str, str]]:
        return self.history[-count:] if len(self.history) >= count else self.history

class ConversationMetadata:
    """Gestion des métadonnées conversationnelles."""
    def __init__(self):
        self.current_language = 'fr'
        self.user_level = 'débutant'
        self.session_stats = {'msg_count': 0, 'avg_msg_length': 0}
        self.last_update = datetime.now()
        self.is_first_message = True

    def update_stats(self, message: str):
        self.session_stats['msg_count'] += 1
        new_length = len(message)
        old_avg = self.session_stats['avg_msg_length']
        self.session_stats['avg_msg_length'] = (
            (old_avg * (self.session_stats['msg_count'] - 1) + new_length) / self.session_stats['msg_count']
        )
        self.is_first_message = False
        self.last_update = datetime.now()

    def detect_language(self, text: str) -> str:
        if not text or len(text.strip()) < 3:
            return self.current_language
        try:
            detected = langdetect.detect(text)
            if detected in ['fr', 'en']:
                self.current_language = detected
            return self.current_language
        except:
            return self.current_language

class GeminiHandler:
    def __init__(self):
        """Initialisation avec Gemini et DataCollector."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI API key not found.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.metadata = ConversationMetadata()
        self.memory = ConversationMemory(max_history=10)
        self.chat = self.model.start_chat(history=[])
        self.retry_delay = 60
        self.data_collector = DataCollector()

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Analyse une date dans différents formats naturels."""
        try:
            date_str = date_str.replace('/', ' ').replace('-', ' ').replace(',', ' ')
            return parse(date_str, dayfirst=True, fuzzy=True)
        except ValueError:
            return None

    def _fetch_price_data(self, symbol: str, date: Optional[datetime] = None, is_crypto: bool = False) -> Dict[str, Any]:
        """Récupère les données de prix et retourne un dictionnaire pour Gemini."""
        try:
            if date:  # Demande historique
                if is_crypto:
                    df = self.data_collector.get_crypto_historical(symbol)
                else:
                    df = self.data_collector.get_stock_historical(symbol)
                
                closest_date = df.index.get_indexer([date], method='nearest')[0]
                row = df.iloc[closest_date]
                date_str = date.strftime('%d/%m/%Y') if self.metadata.current_language == 'fr' else date.strftime('%m/%d/%Y')
                return {
                    'type': 'historical',
                    'symbol': symbol,
                    'name': df.attrs['name'],
                    'price': round(row['Close'], 2),
                    'date': date_str
                }
            else:  # Demande actuelle
                if is_crypto:
                    data = self.data_collector.get_crypto_current(symbol)
                    date_str = datetime.now().strftime('%d/%m/%Y') if self.metadata.current_language == 'fr' else datetime.now().strftime('%m/%d/%Y')
                    return {
                        'type': 'current',
                        'symbol': data['id'],
                        'name': data['name'],
                        'price': round(data['price'], 2),
                        'date': date_str
                    }
                else:
                    data = self.data_collector.get_stock_current(symbol)
                    date_str = datetime.now().strftime('%d/%m/%Y') if self.metadata.current_language == 'fr' else datetime.now().strftime('%m/%d/%Y')
                    return {
                        'type': 'current',
                        'symbol': data['symbol'],
                        'name': data['name'],
                        'price': round(data['price'], 2),
                        'date': date_str
                    }
        except Exception as e:
            return {
                'type': 'error',
                'symbol': symbol,
                'error': str(e)
            }

    def _generate_system_prompt(self, query: str, price_data: Optional[Dict[str, Any]] = None) -> str:
        """Construit un prompt système éducatif bilingue avec gestion des prix."""
        detected_language = self.metadata.detect_language(query)
        current_time = datetime.now()
        days_fr = {0: "lundi", 1: "mardi", 2: "mercredi", 3: "jeudi", 4: "vendredi", 5: "samedi", 6: "dimanche"}
        months_fr = {1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai", 6: "juin", 7: "juillet", 8: "août", 9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"}
        day_name = days_fr[current_time.weekday()]
        month_name = months_fr[current_time.month]
        formatted_date = f"{day_name} {current_time.day} {month_name} {current_time.year}"
        formatted_time = current_time.strftime("%H:%M")

        is_repeated = self.memory.is_repeated_query(query)
        repetition_count = self.memory.get_repetition_count(query)
        recent_interactions = self.memory.get_recent_interactions(2)
        
        recent_history = ""
        if recent_interactions:
            recent_history = "Recent history:\n" + "\n".join(
                f"Q{i+1}: {inter['user']}\nR{i+1}: {inter['assistant']}"
                for i, inter in enumerate(recent_interactions)
            )
        
        level_prompt = "expert" if self.metadata.user_level == 'expert' else "beginner with expert insights"
        
        price_info = ""
        if price_data:
            if price_data['type'] == 'error':
                price_info = f"Price data error for {price_data['symbol']}: {price_data['error']}"
            else:
                price_info = (
                    f"Price data for {price_data['name']} ({price_data['symbol']}): "
                    f"{price_data['price']}$ on {price_data['date']}"
                )
        
        system_prompt = f"""
        You are Finn, a bilingual financial expert (French and English), specialized in trading basics, technical terms, and financial concepts. Respond in {detected_language}.
        Current date and time: {formatted_date}, {formatted_time}

        1- Context:
        - Language: {detected_language}
        - User level: {level_prompt}
        - Repeated query: {is_repeated} (#{repetition_count})
        - First message: {self.metadata.is_first_message}
        
        2- History:
        {recent_history}

        3- RULES:
        a- Respond educationally, concisely, with expert precision, only to the question asked.
        b- Structure complex answers with numbered points for clarity.
        c- Adapt to the user level: {level_prompt}, providing deep insights even for beginners.
        d- If the question is repeated, vary the response naturally with a different phrasing, example, or perspective.
        e- Use practical examples to illustrate concepts when relevant.
        f- For greetings (e.g., 'yo', 'hello'), reply with a natural, engaging greeting in the detected language, then stop.
        g- For basic questions like time or date, reply directly (e.g., 'It’s {formatted_time}' or 'Today is {formatted_date}') without extra fluff.
        h- If the question is off-topic (not finance-related or basic), politely inform the user that you’re limited to finance and stop there.

        4- PRICE HANDLING:
        a- If the user asks for a price (current or historical) of a stock or crypto, use the provided price data and respond simply with the price and date (e.g., 'Tesla is 262.67$ today' or 'Bitcoin was 8500$ in January 2020').
        b- Do NOT add extra details (e.g., percentage changes, trends) unless the user explicitly asks for analysis or context.
        c- If the query involves analysis (e.g., 'highest price', 'average price', 'why this price'), use the price data as a starting point and provide a natural, insightful explanation.
        d- If price data contains an error, inform the user clearly and suggest retrying or clarifying.

        5- FORMAT:
        - Technical terms: definition + expert example + context.
        - Simple price questions: direct answer with price and date, naturally phrased.
        - Analysis questions: use price data and add concise, relevant financial insights.

        Price Data (if available):
        {price_info}

        Query: {query}
        """
        return system_prompt

    def process_query(self, query: str, max_retries=3) -> Dict[str, Any]:
        """Traite la requête avec gestion des prix via Gemini."""
        self.metadata.update_stats(query)
        
        # Détection minimale pour les prix
        query_lower = query.lower()
        price_data = None
        if 'price' in query_lower or 'prix' in query_lower or 'worth' in query_lower or 'combien' in query_lower or 'how much' in query_lower:
            words = query.split()
            symbol = None
            date = None
            is_crypto = 'crypto' in query_lower or 'bitcoin' in query_lower or 'eth' in query_lower
            
            for word in words:
                if len(word) > 1 and (word.isupper() or self.data_collector.mapper.get_stock_info(word) or self.data_collector.mapper.get_crypto_info(word)):
                    symbol = word
                parsed_date = self._parse_date(word)
                if parsed_date:
                    date = parsed_date
            
            if symbol:
                price_data = self._fetch_price_data(symbol, date, is_crypto)

        system_prompt = self._generate_system_prompt(query, price_data)
        
        for attempt in range(max_retries):
            try:
                response = self.chat.send_message(system_prompt)
                if response and response.text:
                    response_text = response.text.strip()
                    self.memory.add_interaction(query, response_text)
                    return {"message": response_text}
                else:
                    error_msg = "Désolé, reformulez votre question !" if self.metadata.current_language == 'fr' else "Sorry, please rephrase your question!"
                    self.memory.add_interaction(query, error_msg)
                    return {"message": error_msg}
            except exceptions.ResourceExhausted:
                if attempt < max_retries - 1:
                    print(f"Quota reached, retrying in {self.retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(self.retry_delay)
                else:
                    print("Quota exhausted after all attempts.")
                    error_msg = "Une erreur est survenue. Réessayez plus tard." if self.metadata.current_language == 'fr' else "An error occurred. Please try again later."
                    self.memory.add_interaction(query, error_msg)
                    return {"message": error_msg}
            except Exception as e:
                print(f"Error: {str(e)}")
                error_msg = "Une erreur est survenue. Réessayez plus tard." if self.metadata.current_language == 'fr' else "An error occurred. Please try again later."
                self.memory.add_interaction(query, error_msg)
                return {"message": error_msg}

if __name__ == "__main__":
    handler = GeminiHandler()
    print("Finn - Financial Education Assistant\nAsk your questions (type 'exit' to quit):")
    
    while True:
        query = input("\nYou: ").strip()
        if query.lower() == 'exit':
            print("Finn: À bientôt !" if handler.metadata.current_language == 'fr' else "Finn: See you soon!")
            break
        response = handler.process_query(query)
        print(f"\nFinn: {response['message']}")