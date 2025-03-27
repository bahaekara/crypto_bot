import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob

# Définir les chemins des dossiers
DATA_DIR = r'C:\Users\bahik\Desktop\crypto_bot\data'
SENTIMENT_DIR = os.path.join(DATA_DIR, 'sentiment')
os.makedirs(SENTIMENT_DIR, exist_ok=True)

# Simulation de l'API client pour Twitter
class TwitterApiClient:
    def __init__(self):
        print("API Client simulé pour Twitter")
    
    def call_api(self, endpoint, query=None):
        print(f"Simulation d'appel API: {endpoint}")
        
        # Simuler des tweets pour l'analyse de sentiment
        if endpoint == 'Twitter/search_twitter' and query and 'query' in query:
            search_term = query['query']
            print(f"Recherche Twitter pour: {search_term}")
            
            # Générer des tweets simulés
            tweets = self.generate_simulated_tweets(search_term, query.get('count', 10))
            return tweets
        
        # Pour les autres types d'appels, retourner None
        return None
    
    def generate_simulated_tweets(self, search_term, count=10):
        """
        Génère des tweets simulés pour un terme de recherche donné
        """
        # Liste de sentiments possibles
        sentiments = ['positif', 'négatif', 'neutre']
        
        # Modèles de tweets selon le sentiment
        tweet_templates = {
            'positif': [
                "J'adore {crypto}! Le prix va certainement monter. #bullish #crypto",
                "{crypto} est l'avenir de la finance. Tellement d'applications potentielles! #crypto #innovation",
                "Vient d'acheter plus de {crypto}. Le projet est solide et l'équipe est excellente. #investissement",
                "Les nouvelles fonctionnalités de {crypto} sont impressionnantes. Adoption massive à venir! #crypto",
                "{crypto} résout des problèmes réels. C'est pourquoi j'y crois. #blockchain #crypto"
            ],
            'négatif': [
                "{crypto} est surévalué. Attendez-vous à une correction. #bearish #crypto",
                "Je viens de vendre tout mon {crypto}. Trop de risques en ce moment. #crypto #prudence",
                "Les problèmes techniques de {crypto} sont inquiétants. L'équipe doit agir vite. #crypto",
                "{crypto} a trop de concurrence. Difficile de rester pertinent. #crypto #marché",
                "La régulation va frapper {crypto} durement. Soyez prudents. #crypto #régulation"
            ],
            'neutre': [
                "Intéressant de voir l'évolution de {crypto} ces derniers temps. #crypto #marché",
                "Quelqu'un a des infos sur les prochaines mises à jour de {crypto}? #crypto #info",
                "Le volume d'échanges de {crypto} est stable cette semaine. #crypto #trading",
                "Comment voyez-vous {crypto} dans 5 ans? #crypto #futur",
                "Comparaison intéressante entre {crypto} et d'autres projets similaires. #crypto #analyse"
            ]
        }
        
        # Extraire le nom de la crypto du terme de recherche
        crypto_name = search_term.split()[0]
        
        # Générer des tweets simulés
        tweets = []
        for i in range(count):
            # Choisir un sentiment aléatoire avec une distribution réaliste
            sentiment_weights = [0.4, 0.3, 0.3]  # 40% positif, 30% négatif, 30% neutre
            sentiment = np.random.choice(sentiments, p=sentiment_weights)
            
            # Choisir un modèle de tweet aléatoire pour ce sentiment
            template = np.random.choice(tweet_templates[sentiment])
            
            # Remplacer {crypto} par le nom de la crypto
            text = template.replace('{crypto}', crypto_name)
            
            # Créer un tweet simulé
            tweet = {
                'id': f"tweet_{i}_{datetime.now().timestamp()}",
                'text': text,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user': {
                    'screen_name': f"crypto_user_{i}",
                    'followers_count': np.random.randint(100, 10000)
                },
                'retweet_count': np.random.randint(0, 100),
                'favorite_count': np.random.randint(0, 200)
            }
            
            tweets.append(tweet)
        
        return tweets

# Créer le client API
client = TwitterApiClient()

def analyze_crypto_sentiment(symbol, name):
    """
    Analyse le sentiment pour une cryptomonnaie donnée
    
    Args:
        symbol (str): Symbole de la cryptomonnaie (ex: BTC-USD)
        name (str): Nom de la cryptomonnaie (ex: Bitcoin)
    
    Returns:
        dict: Résultats de l'analyse de sentiment
    """
    print(f"Analyzing sentiment for {symbol}...")
    
    # Initialiser les résultats avec des valeurs par défaut
    results = {
        'symbol': symbol,
        'name': name,
        'sentiment_score': 0,
        'sentiment_label': 'Neutre',
        'tweet_count': 0,
        'positive_count': 0,
        'negative_count': 0,
        'neutral_count': 0,
        'sentiment_percentages': {
            'positive': 0,
            'negative': 0,
            'neutral': 100
        },
        'tweets': [],
        'popular_tweets': [],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # Liste des termes de recherche
        search_terms = [
            f"{name} crypto",
            f"{name} cryptocurrency",
            f"{name} price",
            f"${symbol.split('-')[0]}"
        ]
        
        all_tweets = []
        
        # Rechercher des tweets pour chaque terme
        for term in search_terms:
            print(f"Searching Twitter for: {term}")
            
            # Appeler l'API Twitter (ou simulée)
            tweets = client.call_api('Twitter/search_twitter', {
                'query': term,
                'count': 100,
                'lang': 'en',
                'result_type': 'recent'
            })
            
            # Si nous avons des tweets, les ajouter à la liste
            if tweets and isinstance(tweets, list):
                all_tweets.extend(tweets)
        
        # Supprimer les doublons
        unique_tweets = []
        tweet_ids = set()
        
        for tweet in all_tweets:
            if tweet and 'id' in tweet and tweet['id'] not in tweet_ids:
                tweet_ids.add(tweet['id'])
                unique_tweets.append(tweet)
        
        # Mettre à jour le nombre de tweets
        results['tweet_count'] = len(unique_tweets)
        
        # Si aucun tweet trouvé, retourner les résultats par défaut
        if not unique_tweets:
            print(f"No tweets found for {symbol}")
            # S'assurer que sentiment_percentages est défini
            results['sentiment_percentages'] = {
                'positive': 0,
                'negative': 0,
                'neutral': 100
            }
            return results
        
        # Analyser le sentiment de chaque tweet
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for tweet in unique_tweets:
            # Utiliser TextBlob pour l'analyse de sentiment
            text = tweet.get('text', '')
            analysis = TextBlob(text)
            
            # Déterminer le sentiment
            if analysis.sentiment.polarity > 0.1:
                sentiment = 'positive'
                positive_count += 1
            elif analysis.sentiment.polarity < -0.1:
                sentiment = 'negative'
                negative_count += 1
            else:
                sentiment = 'neutral'
                neutral_count += 1
            
            # Ajouter le sentiment au tweet
            tweet['sentiment'] = sentiment
            tweet['sentiment_score'] = analysis.sentiment.polarity
            
            # Ajouter le tweet à la liste des tweets analysés
            results['tweets'].append(tweet)
        
        # Trier les tweets par popularité (retweets + likes)
        popular_tweets = sorted(
            results['tweets'],
            key=lambda t: t.get('retweet_count', 0) + t.get('favorite_count', 0),
            reverse=True
        )
        
        # Prendre les 5 tweets les plus populaires
        results['popular_tweets'] = popular_tweets[:5]
        
        # Mettre à jour les compteurs
        results['positive_count'] = positive_count
        results['negative_count'] = negative_count
        results['neutral_count'] = neutral_count
        
        # Calculer les pourcentages
        total_count = positive_count + negative_count + neutral_count
        
        if total_count > 0:
            positive_percentage = (positive_count / total_count) * 100
            negative_percentage = (negative_count / total_count) * 100
            neutral_percentage = (neutral_count / total_count) * 100
        else:
            positive_percentage = 0
            negative_percentage = 0
            neutral_percentage = 100
        
        # S'assurer que sentiment_percentages est défini
        results['sentiment_percentages'] = {
            'positive': round(positive_percentage, 2),
            'negative': round(negative_percentage, 2),
            'neutral': round(neutral_percentage, 2)
        }
        
        # Calculer un score de sentiment entre -1 et 1
        if total_count > 0:
            results['sentiment_score'] = round((positive_count - negative_count) / total_count, 2)
        else:
            results['sentiment_score'] = 0
        
        # Déterminer le label de sentiment
        if results['sentiment_score'] > 0.1:
            results['sentiment_label'] = 'Positif'
        elif results['sentiment_score'] < -0.1:
            results['sentiment_label'] = 'Négatif'
        else:
            results['sentiment_label'] = 'Neutre'
        
        return results
    
    except Exception as e:
        print(f"Error analyzing sentiment for {symbol}: {e}")
        # S'assurer que sentiment_percentages est défini même en cas d'erreur
        results['sentiment_percentages'] = {
            'positive': 0,
            'negative': 0,
            'neutral': 100
        }
        return results

def analyze_all_cryptos(crypto_list, max_cryptos=10):
    """
    Analyse le sentiment pour toutes les cryptomonnaies de la liste
    
    Args:
        crypto_list (list): Liste des dictionnaires contenant les symboles et noms des cryptomonnaies
        max_cryptos (int): Nombre maximum de cryptomonnaies à analyser
    """
    print(f"Analyzing sentiment for {min(len(crypto_list), max_cryptos)} cryptocurrencies...")
    
    # Limiter le nombre de cryptomonnaies à analyser
    cryptos_to_analyze = crypto_list[:max_cryptos]
    
    # Analyser chaque cryptomonnaie
    for crypto in cryptos_to_analyze:
        symbol = crypto['symbol']
        name = crypto['name']
        
        # Analyser le sentiment
        results = analyze_crypto_sentiment(symbol, name)
        
        # Enregistrer les résultats en JSON
        file_path = os.path.join(SENTIMENT_DIR, f"{symbol.replace('-', '_')}_sentiment.json")
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Sentiment results saved for {symbol}")
    
    print("Sentiment analysis completed for all cryptocurrencies")

# Exécuter l'analyse de sentiment
if __name__ == "__main__":
    # Importer la liste des cryptomonnaies
    sys.path.append(r'C:\Users\bahik\Desktop\crypto_bot\data')
    from crypto_list import ALL_CRYPTOS
    
    # Analyser le sentiment pour les cryptomonnaies
    analyze_all_cryptos(ALL_CRYPTOS, max_cryptos=10)
