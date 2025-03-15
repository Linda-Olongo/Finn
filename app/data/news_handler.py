import os
import requests
import time
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic  # Importer l'API Anthropic (Claude)

# Charger les variables d'environnement
load_dotenv()
NEWS_KEYS = os.getenv("NEWS_KEYS")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Récupère la clé API Anthropic

# Vérification des clés API
if not NEWS_KEYS:
    raise ValueError("La clé NEWS_KEYS est introuvable dans le fichier .env")
if not ANTHROPIC_API_KEY:
    raise ValueError("La clé ANTHROPIC_API_KEY est introuvable dans le fichier .env")

# URL de l'API NewsAPI
NEWS_API_URL = "https://newsapi.org/v2/everything"

class NewsHandler:
    def __init__(self):
        self.default_params = {
            "apiKey": NEWS_KEYS,
            "language": "en",  
            "sortBy": "publishedAt",
            "pageSize": 10,
            "from": self._get_date_minus_days(7)  # Actualités des 7 derniers jours seulement
        }
        self.cache = {}  # Initialisation du cache
        self.cache_duration = 600  # 10 minutes
        self.claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    def _get_date_minus_days(self, days):
        """Retourne la date d'il y a X jours au format YYYY-MM-DD pour l'API."""
        date = datetime.now() - timedelta(days=days)
        return date.strftime('%Y-%m-%d')

    def _summarize_with_claude(self, text, title):
        """Utilise l'API Claude pour générer un résumé détaillé et structuré de l'article."""
        try:
            # Vérifier si le texte semble tronqué
            is_truncated = "[+" in text or text.endswith("...") or "chars]" in text
            
            prompt = f"""
            You are an expert in media analysis and information synthesis. I will give you an article and I need a detailed, informative, and factual summary in English.

            ARTICLE TITLE: {title}
            
            CONTENT: {text}
            
            {'⚠️ ATTENTION: The content of this article appears to be truncated.' if is_truncated else ''}
            
            Here's what I want you to do:
            1. Write a summary of 5-7 sentences that captures the essence of the article.
            2. Make sure to include the essential facts, key actors, and context.
            3. Structure your summary coherently with an introduction, development, and conclusion.
            4. Remain strictly factual and neutral - no opinion or interpretation.
            5. Write in a professional and fluid journalistic style.
            6. NEVER invent information that is not in the provided text.
            7. CRITICAL INSTRUCTION: Even if the content is limited or truncated, DO NOT mention this fact in your summary. Focus only on the available facts.
            8. FORBIDDEN PHRASES: Do not use any of these phrases or variations of them:
               - "From the limited information available"
               - "Given the limited information"
               - "The available content is extremely limited"
               - "Note:"
               - "I can only provide"
               - "Without speculation"
               - "Since the available content"
               - "The article discusses"
               - "The article explains"
            9. Start directly with the facts - avoid any meta-commentary about the article or your process.
            10. If you cannot create a full 5-7 sentence summary with the information provided, simply create a shorter summary with what you know, but never apologize or explain why your summary is short.
            
            SUMMARY:
            """

            message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            summary = message.content[0].text.strip()
            return summary if summary else "Summary not available."
        except Exception as e:
            print(f"⚠️ Error generating summary with Claude: {e}")
            return "Summary not available due to a technical issue."
        
    def _get_news(self, query):
        """
        Récupère les actualités et les résume avec Claude.
        """
        if query in self.cache and time.time() - self.cache[query]["timestamp"] < self.cache_duration:
            return self.cache[query]["data"]

        params = self.default_params.copy()
        params["q"] = query
        
        # Augmenter le nombre de résultats pour les recherches d'entreprises spécifiques
        if len(query.split()) <= 3:  # Si la requête est courte (probablement un nom d'entreprise)
            params["pageSize"] = 20  # Récupérer plus d'articles pour avoir un meilleur filtrage

        try:
            response = requests.get(NEWS_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                raise Exception(f"Error with NewsAPI: {data.get('message', 'Unknown error')}")

            articles = []
            for article in data.get("articles", []):
                title = article["title"]
                url = article["url"]
                source = article["source"]["name"]
                date = article["publishedAt"][:10]

                # Pour les requêtes d'entreprises, vérifier que l'article est pertinent avant de résumer
                if len(query.split()) <= 3:  # Si c'est une requête d'entreprise
                    # Vérifier si au moins un des termes de la requête apparaît dans le titre ou la description
                    query_terms = [term.lower() for term in query.split() if len(term) > 2]  # Ignorer les petits mots
                    article_text = f"{title.lower()} {article.get('description', '').lower()}"
                    
                    # Si aucun terme significatif n'est trouvé, ignorer cet article
                    if not any(term in article_text for term in query_terms):
                        continue

                # Préparer le texte pour le résumé
                full_text = f"{title}. "
                
                # Ajouter la description si disponible
                if article.get('description'):
                    full_text += f"{article.get('description')} "
                
                # Ajouter le contenu si disponible
                if article.get('content'):
                    # Nettoyer le contenu des marqueurs de troncation
                    content = article.get('content')
                    # Si le contenu est tronqué, le nettoyer
                    if "[+" in content and "chars]" in content:
                        truncation_info = content[content.find("[+"):content.find("chars]")+6]
                        content = content.replace(truncation_info, " ")
                    full_text += content
                
                # Obtenir un résumé détaillé avec Claude
                summary = self._summarize_with_claude(full_text, title)

                articles.append({
                    "titre": title,
                    "resume": summary,
                    "url": url,
                    "source": source,
                    "date": date
                })

            # Si après filtrage il n'y a plus d'articles, retourner une liste vide
            self.cache[query] = {"data": articles, "timestamp": time.time()}
            return articles

        except requests.RequestException as e:
            print(f"❌ Error retrieving news: {e}")
            return []

    # Fonction de nettoyage des résumés avec expressions régulières
    def _clean_summary(self, summary_text):
        """Nettoie un résumé de toutes métaréférences et commentaires sur les limites de l'article."""
        # Supprimer des paragraphes/phrases entiers problématiques
        patterns_to_remove = [
            r'Note:.*?\.(?:\s|$)',
            r'Given the limited information.*?\.(?:\s|$)',
            r'From the limited information.*?\.(?:\s|$)',
            r'The available content is.*?\.(?:\s|$)',
            r'Since the available content.*?\.(?:\s|$)',
            r'I can only provide.*?\.(?:\s|$)',
            r'Without speculation.*?\.(?:\s|$)',
        ]
        
        cleaned = summary_text
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Supprimer des phrases contenant des termes problématiques
        problematic_terms = [
            r'limited information', r'extremely limited', r'without speculation',
            r'not enough information', r'available content', r'cannot responsibly',
            r'cannot expand', r'the article', r'article suggests', r'article implies',
            r'article states', r'article mentions', r'article describes'
        ]
        
        for term in problematic_terms:
            cleaned = re.sub(r'[^.!?]*\b' + term + r'\b[^.!?]*[.!?]', '', cleaned, flags=re.IGNORECASE)
        
        # Nettoyer les doubles espaces et les sauts de ligne
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Nettoyer la ponctuation
        cleaned = re.sub(r'\.{2,}', '.', cleaned).replace(' .', '.').strip()
        
        # Si le texte commence par des connecteurs logiques, les supprimer
        cleaned = re.sub(r'^(Additionally|Furthermore|However|Moreover|Therefore|Thus|In addition|As a result)\s*,?\s*', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()

    def get_global_news(self, category=None):
        """
        Récupère et formate les actualités économiques globales.
        Si une catégorie est spécifiée, filtre les actualités pour cette catégorie.
        """
        # Définir les requêtes par catégorie adaptées à un chatbot financier (en anglais)
        queries = {
            "Stock Markets": "'stock market' OR 'stock exchange' OR 'Dow Jones' OR 'NYSE' OR 'NASDAQ' OR 'S&P 500' OR 'stock index' OR 'stocks' OR 'shares' OR 'dividends'",
            "Cryptocurrencies": "'cryptocurrency' OR 'blockchain' OR 'NFT' OR 'decentralized finance' OR 'DeFi' OR 'crypto regulation' OR 'mining' OR 'staking'",
            "Macroeconomics": "'interest rates' OR 'inflation' OR 'GDP' OR 'central bank' OR 'federal reserve' OR 'ECB' OR 'monetary policy' OR 'recession' OR 'economic growth'",
            "Commodities": "'gold' OR 'oil' OR 'gas' OR 'commodities' OR 'precious metals' OR 'silver metal' OR 'uranium' OR 'lithium' OR 'energy'",
            "Financial Tech": "'fintech' OR 'digital payments' OR 'financial AI' OR 'algorithmic trading' OR 'roboadvisors' OR 'financial cybersecurity' OR 'big data finance'",
            "Financial Regulation": "'financial regulation' OR 'SEC' OR 'financial legislation' OR 'compliance' OR 'banking supervision' OR 'taxation' OR 'tax havens'",
            "Forex & Currencies": "'forex' OR 'exchange market' OR 'currency' OR 'dollar' OR 'euro' OR 'yen' OR 'pound sterling' OR 'yuan' OR 'digital currency' OR 'CBDC'",
            "Technical Analysis": "'technical analysis' OR 'trading' OR 'charts' OR 'technical indicators' OR 'support resistance' OR 'trends' OR 'volumes' OR 'momentum' OR 'high frequency trading'",
        }
        
        # Utiliser la requête appropriée selon la catégorie
        if category:
            if category not in queries:
                return f"⚠️ Category {category} not recognized."
            query = queries[category]
        else:
            # Si pas de catégorie, prendre Stock Markets par défaut
            query = queries["Stock Markets"]
        
        articles = self._get_news(query)
        if not articles:
            return f"⚠️ No news available for {category}. Please try another category."

        # Retourner directement la liste d'articles plutôt que du texte formaté
        return articles[:5]

    def get_company_news(self, company_name):
        """Récupère les actualités spécifiques à une entreprise avec un résumé détaillé."""
        articles = self._get_news(company_name)
        if not articles:
            return f"⚠️ No relevant news found for **{company_name}**. Please try another search term."

        # Tri des articles par date (du plus récent au plus ancien)
        sorted_articles = sorted(articles, key=lambda x: x['date'], reverse=True)

        # Retourner directement la liste d'articles plutôt que du texte formaté
        return sorted_articles[:5]


if __name__ == "__main__":
    news_handler = NewsHandler()

    while True:
        print("\n📢 MENU - FinanceBot News")
        print("1️⃣ - Global financial news")
        print("2️⃣ - Search news for a specific company/asset")
        print("3️⃣ - Exit")
        
        choice = input("➡️ Enter your choice: ").strip()

        if choice == "1":
            # Show subcategories for global news
            print("\n📊 Choose a financial category:")
            categories = {
                "a": "Stock Markets",
                "b": "Cryptocurrencies",
                "c": "Macroeconomics",
                "d": "Commodities",
                "e": "Financial Tech",
                "f": "Financial Regulation",
                "g": "Forex & Currencies",
                "h": "Technical Analysis"
            }
            
            for key, value in categories.items():
                print(f"{key} - {value}")
            
            cat_choice = input("➡️ Your choice: ").strip()
            
            if cat_choice in categories:
                news_data = news_handler.get_global_news(category=categories[cat_choice])
                
                # Format pour l'affichage console
                if isinstance(news_data, list):
                    for article in news_data:
                        print(f"\n** {article['titre']} **")
                        print(f"  {article['source']} - {article['date']}")
                        print(f"  {article['resume']}")
                        print(f"  [Read full article]({article['url']})")
                else:
                    print(news_data)
            else:
                print("❌ Invalid choice.")
        
        elif choice == "2":
            asset = input("\n🔍 Enter company or asset name : ").strip()
            if not asset:
                print("❌ Invalid name.")
                continue
                
            news_data = news_handler.get_company_news(asset)
            
            # Format pour l'affichage console
            if isinstance(news_data, list):
                for article in news_data:
                    print(f"\n** {article['titre']} **")
                    print(f"  {article['source']} - {article['date']}")
                    print(f"  {article['resume']}")
                    print(f"  [Read full article]({article['url']})")
            else:
                print(news_data)

        elif choice == "3":
            print("👋 See you soon for more financial analysis!")
            break

        else:
            print("❌ Invalid choice, please enter 1, 2 or 3.")