import sys
import os
import json
import pandas as pd
import requests
from datetime import datetime
import logging
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'sentiment_analysis.log'), 'a')
    ]
)
logger = logging.getLogger('sentiment_analysis')

# Créer le répertoire de logs s'il n'existe pas
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'), exist_ok=True)

# Définir les chemins des dossiers de manière relative
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SENTIMENT_DIR = os.path.join(DATA_DIR, 'sentiment')
os.makedirs(SENTIMENT_DIR, exist_ok=True)
os.makedirs(os.path.join(SENTIMENT_DIR, 'figures'), exist_ok=True)

# Client API réel pour Twitter
class TwitterClient:
    def __init__(self, api_key=None, api_secret=None, bearer_token=None):
        self.api_key = api_key or os.environ.get('TWITTER_API_KEY')
        self.api_secret = api_secret or os.environ.get('TWITTER_API_SECRET')
        self.bearer_token = bearer_token or os.environ.get('TWITTER_BEARER_TOKEN')
        
        # Vérifier si les clés API sont disponibles
        if not self.bearer_token:
            logger.warning("Aucun token Twitter fourni dans les variables d'environnement")
            self.use_data_api = True
        else:
            self.use_data_api = False
            logger.info("Client Twitter API initialisé avec bearer token")
    
    def call_api(self, endpoint, query=None):
        """
        Appelle l'API Twitter
        
        Args:
            endpoint (str): Point d'entrée de l'API
            query (dict): Paramètres de la requête
        
        Returns:
            list: Liste de tweets
        """
        if self.use_data_api:
            # Utiliser l'API de données
            try:
                import sys
                sys.path.append('/opt/.manus/.sandbox-runtime')
                from data_api import ApiClient
                
                data_client = ApiClient()
                
                if endpoint == 'Twitter/search_twitter':
                    logger.info(f"Recherche Twitter pour: {query.get('query', '')}")
                    result = data_client.call_api('Twitter/search_twitter', query=query)
                    
                    # Extraire les tweets du résultat
                    tweets = []
                    if result and 'result' in result and 'timeline' in result['result']:
                        timeline = result['result']['timeline']
                        if 'instructions' in timeline:
                            for instruction in timeline['instructions']:
                                if 'entries' in instruction:
                                    for entry in instruction['entries']:
                                        if 'content' in entry and 'items' in entry['content']:
                                            for item in entry['content']['items']:
                                                if 'item' in item and 'itemContent' in item['item']:
                                                    content = item['item']['itemContent']
                                                    if 'tweet_results' in content:
                                                        tweet_result = content['tweet_results']['result']
                                                        tweet = self._extract_tweet_data(tweet_result)
                                                        if tweet:
                                                            tweets.append(tweet)
                    
                    logger.info(f"Récupération de {len(tweets)} tweets via l'API de données")
                    return tweets
                
                return None
            except Exception as e:
                logger.error(f"Erreur lors de l'utilisation de l'API de données Twitter: {e}")
                return None
        else:
            # Utiliser l'API Twitter avec le bearer token
            headers = {
                'Authorization': f'Bearer {self.bearer_token}',
                'Content-Type': 'application/json'
            }
            
            if endpoint == 'Twitter/search_twitter':
                search_query = query.get('query', '')
                count = query.get('count', 10)
                
                # Construire l'URL de recherche Twitter v2
                url = f"https://api.twitter.com/2/tweets/search/recent?query={search_query}&max_results={count}"
                
                try:
                    logger.info(f"Appel à l'API Twitter avec URL: {url}")
                    response = requests.get(url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Convertir les données au format attendu
                        tweets = []
                        if 'data' in data:
                            for tweet_data in data['data']:
                                tweet = {
                                    'id': tweet_data.get('id'),
                                    'text': tweet_data.get('text'),
                                    'created_at': tweet_data.get('created_at'),
                                    'user': {
                                        'screen_name': tweet_data.get('author_id'),
                                        'followers_count': 0  # Non disponible dans cette réponse
                                    },
                                    'retweet_count': 0,  # Non disponible dans cette réponse
                                    'favorite_count': 0  # Non disponible dans cette réponse
                                }
                                tweets.append(tweet)
                        
                        logger.info(f"Récupération de {len(tweets)} tweets via l'API Twitter")
                        return tweets
                    else:
                        logger.error(f"Erreur API Twitter: {response.status_code} - {response.text}")
                        return None
                except Exception as e:
                    logger.error(f"Erreur lors de l'appel à l'API Twitter: {e}")
                    return None
            
            return None
    
    def _extract_tweet_data(self, tweet_result):
        """
        Extrait les données d'un tweet à partir du résultat de l'API
        """
        if not tweet_result:
            return None
        
        try:
            # Extraire les données de base du tweet
            tweet = {}
            
            if 'rest_id' in tweet_result:
                tweet['id'] = tweet_result['rest_id']
            
            if 'legacy' in tweet_result:
                legacy = tweet_result['legacy']
                tweet['text'] = legacy.get('full_text', '')
                tweet['created_at'] = legacy.get('created_at', '')
                tweet['retweet_count'] = legacy.get('retweet_count', 0)
                tweet['favorite_count'] = legacy.get('favorite_count', 0)
            
            # Extraire les données de l'utilisateur
            if 'core' in tweet_result and 'user_results' in tweet_result['core']:
                user_result = tweet_result['core']['user_results']['result']
                
                if 'legacy' in user_result:
                    user_legacy = user_result['legacy']
                    tweet['user'] = {
                        'screen_name': user_legacy.get('screen_name', ''),
                        'followers_count': user_legacy.get('followers_count', 0)
                    }
            
            return tweet
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des données du tweet: {e}")
            return None

# Créer le client API
client = TwitterClient()

def analyze_crypto_sentiment(symbol, name):
    """
    Analyse le sentiment pour une cryptomonnaie donnée
    
    Args:
        symbol (str): Symbole de la cryptomonnaie (ex: BTC-USD)
        name (str): Nom de la cryptomonnaie (ex: Bitcoin)
    
    Returns:
        dict: Résultats de l'analyse de sentiment
    """
    logger.info(f"Analyzing sentiment for {symbol}...")
    
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
            logger.info(f"Searching Twitter for: {term}")
            
            # Appeler l'API Twitter
            tweets = client.call_api('Twitter/search_twitter', {
                'query': term,
                'count': 100,
                'type': 'Latest'
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
            logger.warning(f"No tweets found for {symbol}")
            # S'assurer que sentiment_percentages est défini
            results['sentiment_percentages'] = {
                'positive': 0,
                'negative': 0,
                'neutral': 100
            }
            return results
        
        # Analyser le sentiment de chaque tweet
        from textblob import TextBlob
        
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
            results['sentiment_percentages'] = {
                'positive': round(positive_count / total_count * 100, 1),
                'negative': round(negative_count / total_count * 100, 1),
                'neutral': round(neutral_count / total_count * 100, 1)
            }
        
        # Calculer le score de sentiment global
        if total_count > 0:
            # Formule: (positive - negative) / total * 50 + 50
            # Cela donne un score de 0 à 100, où 50 est neutre
            sentiment_score = (positive_count - negative_count) / total_count * 50 + 50
            results['sentiment_score'] = round(sentiment_score, 1)
            
            # Déterminer le label de sentiment
            if sentiment_score >= 70:
                results['sentiment_label'] = 'Très positif'
            elif sentiment_score >= 60:
                results['sentiment_label'] = 'Positif'
            elif sentiment_score >= 40:
                results['sentiment_label'] = 'Neutre'
            elif sentiment_score >= 30:
                results['sentiment_label'] = 'Négatif'
            else:
                results['sentiment_label'] = 'Très négatif'
        
        logger.info(f"Sentiment analysis completed for {symbol}: Score={results['sentiment_score']}, Label={results['sentiment_label']}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error during sentiment analysis for {symbol}: {e}")
        return results

def analyze_all_cryptos():
    """
    Analyse le sentiment pour toutes les cryptomonnaies
    """
    # Importer la liste des cryptomonnaies
    sys.path.append(os.path.join(BASE_DIR, 'data'))
    from crypto_list import ALL_CRYPTOS
    
    # Créer un DataFrame pour stocker les résultats
    results_data = []
    
    # Analyser chaque cryptomonnaie
    for crypto in ALL_CRYPTOS:
        symbol = crypto['symbol']
        name = crypto['name']
        
        # Analyser le sentiment
        results = analyze_crypto_sentiment(symbol, name)
        
        # Enregistrer les résultats en JSON
        file_path = os.path.join(SENTIMENT_DIR, f"{symbol.replace('-', '_')}_sentiment.json")
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Ajouter les résultats au DataFrame
        results_data.append({
            'symbol': symbol,
            'name': name,
            'sentiment': results['sentiment_score'],
            'label': results['sentiment_label'],
            'positive': results['sentiment_percentages']['positive'],
            'neutral': results['sentiment_percentages']['neutral'],
            'negative': results['sentiment_percentages']['negative'],
            'tweet_count': results['tweet_count'],
            'performance': 0,  # Sera calculé plus tard
            'potential': 0,    # Sera calculé plus tard
            'score': 0,        # Sera calculé plus tard
            'rating': 'N/A'    # Sera calculé plus tard
        })
    
    # Créer un DataFrame
    df = pd.DataFrame(results_data)
    
    # Calculer les performances (basées sur le sentiment)
    if not df.empty:
        # Normaliser les scores de sentiment entre 0 et 100
        min_sentiment = df['sentiment'].min()
        max_sentiment = df['sentiment'].max()
        
        if max_sentiment > min_sentiment:
            df['performance'] = ((df['sentiment'] - min_sentiment) / (max_sentiment - min_sentiment)) * 100
        else:
            df['performance'] = 50  # Valeur par défaut si tous les sentiments sont identiques
        
        # Calculer le potentiel (basé sur le nombre de tweets et le sentiment)
        # Plus de tweets et un sentiment plus positif = plus de potentiel
        df['potential'] = df['sentiment'] * (1 + df['tweet_count'] / df['tweet_count'].max()) / 2
        
        # Normaliser le potentiel entre 0 et 100
        min_potential = df['potential'].min()
        max_potential = df['potential'].max()
        
        if max_potential > min_potential:
            df['potential'] = ((df['potential'] - min_potential) / (max_potential - min_potential)) * 100
        else:
            df['potential'] = 50  # Valeur par défaut si tous les potentiels sont identiques
        
        # Calculer le score global (moyenne du sentiment et du potentiel)
        df['score'] = (df['sentiment'] + df['potential']) / 2
        
        # Attribuer une notation
        df['rating'] = pd.cut(
            df['score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['F', 'D', 'C', 'B', 'A']
        )
    
    # Enregistrer les résultats en CSV
    df.to_csv(os.path.join(SENTIMENT_DIR, 'manual_sentiment_data.csv'), index=False)
    
    # Générer des visualisations
    generate_sentiment_visualizations(df)
    
    # Générer un rapport
    generate_sentiment_report(df)
    
    logger.info(f"Sentiment analysis completed for {len(df)} cryptocurrencies")
    
    return df

def generate_sentiment_visualizations(df):
    """
    Génère des visualisations pour l'analyse de sentiment
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les résultats de l'analyse de sentiment
    """
    if df.empty:
        logger.warning("No data available for sentiment visualizations")
        return
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Définir le style des graphiques
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Comparaison des sentiments
        plt.figure(figsize=(12, 8))
        
        # Trier par sentiment
        df_sorted = df.sort_values('sentiment', ascending=False)
        
        # Créer un graphique à barres
        bars = plt.bar(df_sorted['symbol'], df_sorted['sentiment'], color='skyblue')
        
        # Colorer les barres en fonction du sentiment
        for i, bar in enumerate(bars):
            if df_sorted.iloc[i]['sentiment'] >= 70:
                bar.set_color('darkgreen')
            elif df_sorted.iloc[i]['sentiment'] >= 60:
                bar.set_color('green')
            elif df_sorted.iloc[i]['sentiment'] >= 40:
                bar.set_color('gray')
            elif df_sorted.iloc[i]['sentiment'] >= 30:
                bar.set_color('red')
            else:
                bar.set_color('darkred')
        
        plt.axhline(y=50, color='black', linestyle='--')
        plt.title('Comparaison des sentiments par cryptomonnaie')
        plt.xlabel('Cryptomonnaie')
        plt.ylabel('Score de sentiment (0-100)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(SENTIMENT_DIR, 'figures', 'sentiment_comparison.png'))
        plt.close()
        
        # 2. Graphique de performance vs potentiel
        plt.figure(figsize=(10, 8))
        
        # Créer un nuage de points
        scatter = plt.scatter(
            df['performance'], 
            df['potential'], 
            s=df['tweet_count'] * 2,  # Taille proportionnelle au nombre de tweets
            alpha=0.7,
            c=df['sentiment'],
            cmap='RdYlGn'
        )
        
        # Ajouter des étiquettes pour chaque point
        for i, row in df.iterrows():
            plt.annotate(
                row['symbol'], 
                (row['performance'], row['potential']),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.colorbar(scatter, label='Sentiment')
        plt.title('Performance vs Potentiel')
        plt.xlabel('Performance')
        plt.ylabel('Potentiel')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(SENTIMENT_DIR, 'figures', 'performance_vs_potential.png'))
        plt.close()
        
        # 3. Heatmap de corrélation
        plt.figure(figsize=(10, 8))
        
        # Sélectionner les colonnes numériques
        numeric_cols = ['sentiment', 'positive', 'neutral', 'negative', 'tweet_count', 'performance', 'potential', 'score']
        corr = df[numeric_cols].corr()
        
        # Créer une heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Matrice de corrélation')
        plt.tight_layout()
        plt.savefig(os.path.join(SENTIMENT_DIR, 'figures', 'correlation_heatmap.png'))
        plt.close()
        
        # 4. Top 5 cryptomonnaies par score global
        plt.figure(figsize=(10, 6))
        
        # Trier par score
        top5 = df.sort_values('score', ascending=False).head(5)
        
        # Créer un graphique à barres
        bars = plt.bar(top5['symbol'], top5['score'], color='green')
        
        # Ajouter les valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1,
                f'{height:.1f}',
                ha='center',
                va='bottom'
            )
        
        plt.title('Top 5 cryptomonnaies par score global')
        plt.xlabel('Cryptomonnaie')
        plt.ylabel('Score global (0-100)')
        plt.ylim(0, 105)  # Pour laisser de la place aux étiquettes
        plt.tight_layout()
        plt.savefig(os.path.join(SENTIMENT_DIR, 'figures', 'global_score_ranking.png'))
        plt.close()
        
        # 5. Graphique radar pour le top 5
        plt.figure(figsize=(10, 8))
        
        # Préparer les données
        categories = ['Sentiment', 'Performance', 'Potentiel', 'Score']
        
        # Nombre de variables
        N = len(categories)
        
        # Angle de chaque axe
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]  # Fermer le graphique
        
        # Initialiser le graphique
        ax = plt.subplot(111, polar=True)
        
        # Ajouter chaque cryptomonnaie
        for i, row in top5.iterrows():
            values = [row['sentiment'], row['performance'], row['potential'], row['score']]
            values += values[:1]  # Fermer le graphique
            
            # Tracer la ligne
            ax.plot(angles, values, linewidth=2, label=row['symbol'])
            ax.fill(angles, values, alpha=0.1)
        
        # Fixer les étiquettes
        plt.xticks(angles[:-1], categories)
        
        # Ajouter une légende
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Analyse radar des 5 meilleures cryptomonnaies')
        plt.tight_layout()
        plt.savefig(os.path.join(SENTIMENT_DIR, 'figures', 'top5_radar.png'))
        plt.close()
        
        logger.info("Sentiment visualizations generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating sentiment visualizations: {e}")

def generate_sentiment_report(df):
    """
    Génère un rapport de sentiment en markdown
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les résultats de l'analyse de sentiment
    """
    if df.empty:
        logger.warning("No data available for sentiment report")
        return
    
    try:
        # Trier par score
        df_sorted = df.sort_values('score', ascending=False)
        
        # Créer le contenu du rapport
        report = f"""# Rapport d'Analyse de Sentiment des Cryptomonnaies

*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Résumé

Ce rapport présente l'analyse de sentiment pour {len(df)} cryptomonnaies, basée sur les données de Twitter.

## Top 5 Cryptomonnaies par Sentiment

"""
        
        # Ajouter les 5 meilleures cryptomonnaies
        top5 = df_sorted.head(5)
        
        for i, row in top5.iterrows():
            report += f"""### {i+1}. {row['name']} ({row['symbol']})

- **Score de sentiment**: {row['sentiment']:.1f}/100
- **Label**: {row['sentiment_label']}
- **Tweets analysés**: {row['tweet_count']}
- **Répartition**: {row['positive']:.1f}% positifs, {row['neutral']:.1f}% neutres, {row['negative']:.1f}% négatifs
- **Performance**: {row['performance']:.1f}/100
- **Potentiel**: {row['potential']:.1f}/100
- **Score global**: {row['score']:.1f}/100
- **Notation**: {row['rating']}

"""
        
        # Ajouter un tableau de toutes les cryptomonnaies
        report += """## Toutes les Cryptomonnaies Analysées

| Symbole | Nom | Sentiment | Label | Tweets | Positifs | Neutres | Négatifs | Score | Note |
|---------|-----|-----------|-------|--------|----------|---------|----------|-------|------|
"""
        
        for i, row in df_sorted.iterrows():
            report += f"| {row['symbol']} | {row['name']} | {row['sentiment']:.1f} | {row['label']} | {row['tweet_count']} | {row['positive']:.1f}% | {row['neutral']:.1f}% | {row['negative']:.1f}% | {row['score']:.1f} | {row['rating']} |\n"
        
        # Ajouter des notes méthodologiques
        report += """
## Méthodologie

L'analyse de sentiment est basée sur les tweets récents mentionnant chaque cryptomonnaie. Le processus comprend:

1. **Collecte de données**: Recherche de tweets contenant le nom ou le symbole de la cryptomonnaie
2. **Analyse de sentiment**: Utilisation de TextBlob pour déterminer la polarité du sentiment de chaque tweet
3. **Agrégation**: Calcul des scores globaux et des pourcentages
4. **Notation**: Attribution d'une note de A à F basée sur le score global

## Interprétation des Résultats

- **Score de sentiment**: 0-100, où 50 est neutre, au-dessus est positif, en-dessous est négatif
- **Performance**: Mesure relative du sentiment par rapport aux autres cryptomonnaies
- **Potentiel**: Combinaison du sentiment et de la popularité (nombre de tweets)
- **Score global**: Moyenne du sentiment et du potentiel
- **Notation**: A (excellent) à F (mauvais)

## Notes Importantes

- L'analyse de sentiment des réseaux sociaux n'est qu'un indicateur parmi d'autres
- Les sentiments peuvent changer rapidement en fonction des événements du marché
- Un sentiment positif ne garantit pas une performance future positive
- Cette analyse doit être combinée avec d'autres formes d'analyse pour une vue complète
"""
        
        # Enregistrer le rapport
        report_path = os.path.join(SENTIMENT_DIR, 'sentiment_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Sentiment report generated: {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating sentiment report: {e}")

# Exécuter l'analyse de sentiment
if __name__ == "__main__":
    analyze_all_cryptos()
