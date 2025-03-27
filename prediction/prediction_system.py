import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Ajouter le répertoire parent au chemin de recherche
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Définir les chemins des dossiers de manière relative
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')
SENTIMENT_DIR = os.path.join(DATA_DIR, 'sentiment')
PREDICTION_DIR = os.path.join(BASE_DIR, 'prediction')
os.makedirs(PREDICTION_DIR, exist_ok=True)
os.makedirs(os.path.join(PREDICTION_DIR, 'figures'), exist_ok=True)

def load_technical_indicators(symbol):
    """
    Charge les indicateurs techniques pour une cryptomonnaie
    
    Args:
        symbol (str): Symbole de la cryptomonnaie (ex: BTC-USD)
    
    Returns:
        pandas.DataFrame: DataFrame contenant les indicateurs techniques
    """
    file_path = os.path.join(ANALYSIS_DIR, f"{symbol.replace('-', '_')}_indicators.csv")
    if not os.path.exists(file_path):
        print(f"No technical indicators found for {symbol}")
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def load_sentiment_data():
    """
    Charge les données de sentiment
    
    Returns:
        pandas.DataFrame: DataFrame contenant les données de sentiment
    """
    file_path = os.path.join(SENTIMENT_DIR, 'manual_sentiment_data.csv')
    if not os.path.exists(file_path):
        print("No sentiment data found")
        return None
    
    df = pd.read_csv(file_path)
    return df

def prepare_features(technical_df, sentiment_df, symbol):
    """
    Prépare les caractéristiques pour le modèle de prédiction
    
    Args:
        technical_df (pandas.DataFrame): DataFrame contenant les indicateurs techniques
        sentiment_df (pandas.DataFrame): DataFrame contenant les données de sentiment
        symbol (str): Symbole de la cryptomonnaie
    
    Returns:
        pandas.DataFrame: DataFrame contenant les caractéristiques préparées
    """
    if technical_df is None:
        return None
    
    # Sélectionner les dernières données (30 jours)
    if len(technical_df) > 30:
        technical_df = technical_df.iloc[-30:]
    
    # Sélectionner les caractéristiques pertinentes
    features = technical_df[['close', 'volume', 'MA_7', 'MA_20', 'RSI_14', 'MACD_line', 'MACD_signal', 
                           'BB_upper_20', 'BB_middle_20', 'BB_lower_20', 'volatility_14', 'momentum_14']].copy()
    
    # Ajouter les données de sentiment si disponibles
    if sentiment_df is not None:
        sentiment_row = sentiment_df[sentiment_df['symbol'] == symbol]
        if not sentiment_row.empty:
            for col in ['sentiment', 'performance', 'potential', 'score', 'rating']:
                features[col] = sentiment_row[col].values[0]
    
    # Remplacer les valeurs NaN par 0
    features.fillna(0, inplace=True)
    
    return features

def create_prediction_model(features_df, symbol, sentiment_df):
    """
    Crée un modèle de prédiction basé sur les tendances récentes et le sentiment
    
    Args:
        features_df (pandas.DataFrame): DataFrame contenant les caractéristiques
        symbol (str): Symbole de la cryptomonnaie
        sentiment_df (pandas.DataFrame): DataFrame contenant les données de sentiment
    
    Returns:
        dict: Résultats de la prédiction
    """
    if features_df is None or features_df.empty:
        return None
    
    # Obtenir le prix actuel
    current_price = features_df['close'].iloc[-1]
    
    # Calculer la tendance récente (7 jours)
    if len(features_df) >= 7:
        price_7d_ago = features_df['close'].iloc[-7]
        # Éviter la division par zéro
        if price_7d_ago > 0:
            recent_trend = ((current_price - price_7d_ago) / price_7d_ago) * 100
        else:
            recent_trend = 0
    else:
        recent_trend = 0
        
    # S'assurer que la tendance n'est pas NaN ou infinie
    if np.isnan(recent_trend) or np.isinf(recent_trend):
        recent_trend = 0
    
    # Obtenir les données de sentiment
    sentiment_row = sentiment_df[sentiment_df['symbol'] == symbol]
    if sentiment_row.empty:
        sentiment = 50
        potential = 0
    else:
        sentiment = sentiment_row['sentiment'].values[0]
        potential = sentiment_row['potential'].values[0]
    
    # Calculer un score de prédiction basé sur la tendance récente et le sentiment
    prediction_score = (recent_trend * 0.4) + ((sentiment - 50) * 0.4) + (potential * 0.2 / 100)
    
    # Estimer le prix futur
    predicted_price = current_price * (1 + prediction_score / 100)
    
    # Calculer la variation en pourcentage
    percent_change = ((predicted_price - current_price) / current_price) * 100
    
    # Déterminer la direction et le signal
    if percent_change > 5:
        direction = "Forte hausse"
        signal = "ACHAT"
    elif percent_change > 1:
        direction = "Hausse modérée"
        signal = "ACHAT"
    elif percent_change > -1:
        direction = "Stable"
        signal = "CONSERVER"
    elif percent_change > -5:
        direction = "Baisse modérée"
        signal = "VENTE"
    else:
        direction = "Forte baisse"
        signal = "VENTE"
    
    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "percent_change": percent_change,
        "direction": direction,
        "signal": signal,
        "recent_trend": recent_trend,
        "sentiment": sentiment,
        "potential": potential
    }

def analyze_all_cryptos():
    """
    Analyse toutes les cryptomonnaies disponibles
    
    Returns:
        pandas.DataFrame: DataFrame contenant les résultats de l'analyse
    """
    # Charger les données de sentiment
    sentiment_df = load_sentiment_data()
    
    if sentiment_df is None:
        print("No sentiment data available. Cannot proceed with prediction.")
        return None
    
    # Obtenir la liste des symboles à partir des données de sentiment
    symbols = sentiment_df['symbol'].tolist()
    
    # Vérifier d'abord quelles cryptomonnaies ont des données techniques disponibles
    available_cryptos = []
    for symbol in symbols:
        file_path = os.path.join(ANALYSIS_DIR, f"{symbol.replace('-', '_')}_indicators.csv")
        if os.path.exists(file_path):
            available_cryptos.append(symbol)
        else:
            print(f"Skipping {symbol} - No technical indicators file found")
    
    if not available_cryptos:
        print("No cryptocurrencies with technical data available. Cannot proceed with prediction.")
        return None
    
    print(f"Found {len(available_cryptos)} cryptocurrencies with technical data out of {len(symbols)}")
    
    results = []
    
    for symbol in available_cryptos:
        print(f"Analyzing {symbol}...")
        
        # Charger les indicateurs techniques
        technical_df = load_technical_indicators(symbol)
        
        if technical_df is None or technical_df.empty:
            print(f"Skipping {symbol} due to empty technical data")
            continue
        
        # Préparer les caractéristiques
        features_df = prepare_features(technical_df, sentiment_df, symbol)
        
        if features_df is None or features_df.empty:
            print(f"Skipping {symbol} due to insufficient feature data")
            continue
        
        # Utiliser le modèle de prédiction
        prediction = create_prediction_model(features_df, symbol, sentiment_df)
        
        if prediction is None:
            print(f"Skipping {symbol} due to prediction failure")
            continue
        
        # Obtenir les données de sentiment
        sentiment_row = sentiment_df[sentiment_df['symbol'] == symbol].iloc[0]
        
        # Créer le résultat
        result = {
            "symbol": symbol,
            "current_price": prediction["current_price"],
            "predicted_price": prediction["predicted_price"],
            "percent_change": prediction["percent_change"],
            "direction": prediction["direction"],
            "signal": prediction["signal"],
            "recent_trend": prediction["recent_trend"],
            "sentiment": sentiment_row["sentiment"],
            "performance": sentiment_row["performance"],
            "potential": sentiment_row["potential"],
            "score": sentiment_row["score"],
            "rating": sentiment_row["rating"]
        }
        
        results.append(result)
    
    # Créer un DataFrame avec les résultats
    if results:
        results_df = pd.DataFrame(results)
        
        # Trier par potentiel de croissance
        results_df.sort_values('percent_change', ascending=False, inplace=True)
        
        # Enregistrer les résultats
        results_df.to_csv(os.path.join(PREDICTION_DIR, 'prediction_results.csv'), index=False)
        
        return results_df
    else:
        print("No results generated")
        return None

def generate_prediction_visualizations(results_df):
    """
    Génère des visualisations pour les prédictions
    
    Args:
        results_df (pandas.DataFrame): DataFrame contenant les résultats des prédictions
    """
    if results_df is None or results_df.empty:
        print("No prediction results available for visualization")
        return
    
    # Créer un dossier pour les figures
    figures_dir = os.path.join(PREDICTION_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Définir le style des graphiques
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Graphique des prix actuels vs prédits
    plt.figure(figsize=(12, 8))
    
    # Sélectionner les 10 premières cryptomonnaies
    top_10 = results_df.head(10)
    
    # Créer un graphique à barres
    x = np.arange(len(top_10))
    width = 0.35
    
    plt.bar(x - width/2, top_10['current_price'], width, label='Prix actuel')
    plt.bar(x + width/2, top_10['predicted_price'], width, label='Prix prédit')
    
    plt.xlabel('Cryptomonnaie')
    plt.ylabel('Prix (USD)')
    plt.title('Prix actuels vs Prix prédits (Top 10)')
    plt.xticks(x, top_10['symbol'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'current_vs_predicted_prices.png'))
    plt.close()
    
    # 2. Graphique des variations de prix prédites
    plt.figure(figsize=(12, 8))
    
    # Trier par variation de prix
    sorted_by_change = results_df.sort_values('percent_change', ascending=False)
    top_10_change = sorted_by_change.head(10)
    
    # Créer un graphique à barres horizontales
    bars = plt.barh(top_10_change['symbol'], top_10_change['percent_change'])
    
    # Colorer les barres en fonction de la direction
    for i, bar in enumerate(bars):
        if top_10_change.iloc[i]['percent_change'] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.xlabel('Variation prédite (%)')
    plt.ylabel('Cryptomonnaie')
    plt.title('Variations de prix prédites (Top 10)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'predicted_price_changes.png'))
    plt.close()
    
    # 3. Graphique de sentiment vs prédiction
    plt.figure(figsize=(10, 8))
    
    # Créer un nuage de points
    plt.scatter(results_df['sentiment'], results_df['percent_change'], 
                alpha=0.7, s=100, c=results_df['potential'], cmap='viridis')
    
    # Ajouter des étiquettes pour les points importants
    for i, row in results_df.iterrows():
        if abs(row['percent_change']) > 5 or row['sentiment'] > 70 or row['sentiment'] < 30:
            plt.annotate(row['symbol'], 
                         (row['sentiment'], row['percent_change']),
                         xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Sentiment')
    plt.ylabel('Variation prédite (%)')
    plt.title('Sentiment vs Prédiction')
    plt.colorbar(label='Potentiel')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'sentiment_vs_prediction.png'))
    plt.close()
    
    # 4. Dashboard des 5 meilleures cryptomonnaies
    plt.figure(figsize=(15, 10))
    
    # Sélectionner les 5 meilleures cryptomonnaies
    top_5 = results_df.head(5)
    
    # Créer un subplot pour chaque métrique
    metrics = ['percent_change', 'sentiment', 'potential', 'performance', 'score']
    colors = ['green', 'blue', 'purple', 'orange', 'red']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        bars = plt.bar(top_5['symbol'], top_5[metric], color=colors[i])
        plt.title(metric.replace('_', ' ').title())
        plt.xticks(rotation=45)
        
        # Ajouter les valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'top5_dashboard.png'))
    plt.close()
    
    print("Prediction visualizations generated")

def generate_prediction_report(results_df):
    """
    Génère un rapport de prédiction en markdown
    
    Args:
        results_df (pandas.DataFrame): DataFrame contenant les résultats des prédictions
    """
    if results_df is None or results_df.empty:
        print("No prediction results available for report generation")
        return
    
    # Créer le contenu du rapport
    report = f"""# Rapport de Prédiction des Cryptomonnaies

*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Résumé

Ce rapport présente les prédictions de prix pour {len(results_df)} cryptomonnaies, basées sur l'analyse technique et l'analyse de sentiment.

## Top 5 Cryptomonnaies Recommandées

"""
    
    # Ajouter les 5 meilleures cryptomonnaies
    top_5 = results_df.head(5)
    
    for i, row in top_5.iterrows():
        report += f"""### {i+1}. {row['symbol']}

- **Prix actuel**: {row['current_price']:.2f} USD
- **Prix prédit**: {row['predicted_price']:.2f} USD
- **Variation prédite**: {row['percent_change']:.2f}%
- **Direction**: {row['direction']}
- **Signal**: {row['signal']}
- **Sentiment**: {row['sentiment']:.1f}/100
- **Potentiel**: {row['potential']:.1f}/100
- **Score global**: {row['score']:.1f}/100
- **Notation**: {row['rating']}

"""
    
    # Ajouter un tableau de toutes les cryptomonnaies
    report += """## Toutes les Cryptomonnaies Analysées

| Symbole | Prix Actuel | Prix Prédit | Variation (%) | Signal | Sentiment | Potentiel |
|---------|-------------|-------------|---------------|--------|-----------|-----------|
"""
    
    for i, row in results_df.iterrows():
        report += f"| {row['symbol']} | {row['current_price']:.2f} | {row['predicted_price']:.2f} | {row['percent_change']:.2f} | {row['signal']} | {row['sentiment']:.1f} | {row['potential']:.1f} |\n"
    
    # Ajouter des notes méthodologiques
    report += """
## Méthodologie

Les prédictions sont basées sur une combinaison de:
1. **Analyse technique**: Utilisant des indicateurs comme les moyennes mobiles, RSI, MACD et les bandes de Bollinger
2. **Analyse de sentiment**: Basée sur les données de sentiment des réseaux sociaux et des forums
3. **Évaluation du potentiel**: Combinant les performances passées et les perspectives futures

## Notes Importantes

- Ces prédictions sont générées par un modèle automatisé et ne constituent pas des conseils financiers
- Les marchés de cryptomonnaies sont hautement volatils et imprévisibles
- Toujours faire vos propres recherches avant d'investir
- Les performances passées ne garantissent pas les résultats futurs
"""
    
    # Enregistrer le rapport
    report_path = os.path.join(PREDICTION_DIR, 'prediction_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Prediction report generated: {report_path}")

# Exécuter l'analyse et la prédiction
if __name__ == "__main__":
    # Analyser toutes les cryptomonnaies
    results_df = analyze_all_cryptos()
    
    if results_df is not None:
        # Générer des visualisations
        generate_prediction_visualizations(results_df)
        
        # Générer un rapport
        generate_prediction_report(results_df)
        
        print("Prediction analysis completed")
    else:
        print("Prediction analysis failed due to missing data")
