import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re

# Définir les chemins des dossiers
DATA_DIR = '/home/ubuntu/crypto_bot/data'
SENTIMENT_DIR = os.path.join(DATA_DIR, 'sentiment')
os.makedirs(SENTIMENT_DIR, exist_ok=True)
os.makedirs(os.path.join(SENTIMENT_DIR, 'figures'), exist_ok=True)

# Utiliser les données fournies par l'utilisateur pour l'analyse de sentiment
def create_manual_sentiment_data():
    """
    Crée des données de sentiment manuelles basées sur les informations fournies par l'utilisateur
    """
    # Données fournies par l'utilisateur
    crypto_data = [
        {"symbol": "BTC-USD", "performance": 3.63, "potential": 22.95, "risk": 0.00, "score": 9.55, "sentiment": 90.00, "rating": 5},
        {"symbol": "ETH-USD", "performance": 1.95, "potential": 11.83, "risk": 0.00, "score": 5.29, "sentiment": 90.00, "rating": 1},
        {"symbol": "SOL-USD", "performance": 11.78, "potential": 90.66, "risk": 0.00, "score": 8.45, "sentiment": 90.00, "rating": 5},
        {"symbol": "XRP-USD", "performance": 18.96, "potential": 173.41, "risk": 0.00, "score": 10.49, "sentiment": 90.00, "rating": 7},
        {"symbol": "ADA-USD", "performance": 15.68, "potential": 132.59, "risk": 0.00, "score": 6.41, "sentiment": 90.00, "rating": 5},
        {"symbol": "OM-USD", "performance": 28.38, "potential": 325.29, "risk": 0.00, "score": 11.09, "sentiment": 80.00, "rating": 3},
        {"symbol": "XLM-USD", "performance": 3.00, "potential": 18.66, "risk": 0.00, "score": 5.29, "sentiment": 90.00, "rating": 1},
        {"symbol": "XMR-USD", "performance": 6.40, "potential": 43.22, "risk": -0.11, "score": 11.84, "sentiment": 100.00, "rating": 4},
        {"symbol": "LINK-USD", "performance": 8.73, "potential": 62.39, "risk": 0.00, "score": 10.22, "sentiment": 90.00, "rating": 5},
        {"symbol": "AVAX-USD", "performance": 1.58, "potential": 9.51, "risk": 0.00, "score": 7.20, "sentiment": 90.00, "rating": 3},
        {"symbol": "LTC-USD", "performance": 12.02, "potential": 93.06, "risk": -0.47, "score": 7.92, "sentiment": 90.00, "rating": 5}
    ]
    
    # Convertir en DataFrame
    df = pd.DataFrame(crypto_data)
    
    # Enregistrer les données
    df.to_csv(os.path.join(SENTIMENT_DIR, 'manual_sentiment_data.csv'), index=False)
    
    return df

def calculate_sentiment_metrics(df):
    """
    Calcule des métriques de sentiment supplémentaires
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données de sentiment
    
    Returns:
        pandas.DataFrame: DataFrame avec des métriques supplémentaires
    """
    # Normaliser les scores de sentiment (0-1)
    df['sentiment_normalized'] = df['sentiment'] / 100.0
    
    # Calculer un score de sentiment pondéré (combinaison de sentiment et performance)
    df['weighted_sentiment'] = df['sentiment_normalized'] * 0.7 + (df['performance'] / df['performance'].max()) * 0.3
    
    # Calculer un score global (combinaison de sentiment, performance et potentiel)
    df['global_score'] = (
        df['sentiment_normalized'] * 0.4 + 
        (df['performance'] / df['performance'].max()) * 0.3 + 
        (df['potential'] / df['potential'].max()) * 0.3
    )
    
    # Classer les cryptomonnaies selon leur score global
    df['rank'] = df['global_score'].rank(ascending=False)
    
    return df

def generate_sentiment_visualizations(df):
    """
    Génère des visualisations pour l'analyse de sentiment
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données de sentiment
    """
    # Définir le style des graphiques
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Graphique de comparaison des sentiments
    plt.figure(figsize=(12, 8))
    
    # Trier par sentiment
    df_sorted = df.sort_values('sentiment', ascending=False)
    
    # Créer le graphique à barres
    bars = plt.bar(df_sorted['symbol'], df_sorted['sentiment'], color=plt.cm.viridis(df_sorted['sentiment']/100))
    
    plt.xlabel('Cryptomonnaie')
    plt.ylabel('Sentiment (%)')
    plt.title('Comparaison du Sentiment par Cryptomonnaie')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SENTIMENT_DIR, 'figures', 'sentiment_comparison.png'))
    plt.close()
    
    # 2. Graphique de performance vs potentiel
    plt.figure(figsize=(12, 10))
    
    scatter = plt.scatter(df['performance'], df['potential'], 
                         s=df['score']*20, 
                         c=df['sentiment'], 
                         cmap='viridis', 
                         alpha=0.7)
    
    # Ajouter les étiquettes pour chaque point
    for i, txt in enumerate(df['symbol']):
        plt.annotate(txt, (df['performance'].iloc[i], df['potential'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.colorbar(scatter, label='Sentiment (%)')
    plt.xlabel('Performance (%)')
    plt.ylabel('Potentiel (%)')
    plt.title('Performance vs Potentiel avec Sentiment')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(SENTIMENT_DIR, 'figures', 'performance_vs_potential.png'))
    plt.close()
    
    # 3. Graphique radar pour les top 5 cryptomonnaies
    top5 = df.sort_values('global_score', ascending=False).head(5)
    
    # Préparer les données pour le graphique radar
    categories = ['Performance', 'Potentiel', 'Sentiment', 'Score']
    
    # Normaliser les valeurs pour le graphique radar
    values = {
        'Performance': top5['performance'] / top5['performance'].max(),
        'Potentiel': top5['potential'] / top5['potential'].max(),
        'Sentiment': top5['sentiment_normalized'],
        'Score': top5['score'] / top5['score'].max()
    }
    
    # Nombre de variables
    N = len(categories)
    
    # Angle pour chaque axe
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fermer le graphique
    
    # Créer le graphique radar
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Ajouter les axes
    plt.xticks(angles[:-1], categories, size=12)
    
    # Dessiner les polygones pour chaque cryptomonnaie
    for i, symbol in enumerate(top5['symbol']):
        values_crypto = [values[cat].iloc[i] for cat in categories]
        values_crypto += values_crypto[:1]  # Fermer le polygone
        
        ax.plot(angles, values_crypto, linewidth=2, linestyle='solid', label=symbol)
        ax.fill(angles, values_crypto, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Analyse Comparative des Top 5 Cryptomonnaies', size=15)
    plt.tight_layout()
    plt.savefig(os.path.join(SENTIMENT_DIR, 'figures', 'top5_radar.png'))
    plt.close()
    
    # 4. Heatmap des corrélations
    plt.figure(figsize=(10, 8))
    
    # Sélectionner les colonnes numériques
    numeric_cols = ['performance', 'potential', 'risk', 'score', 'sentiment', 'rating', 
                   'sentiment_normalized', 'weighted_sentiment', 'global_score']
    
    # Calculer la matrice de corrélation
    corr_matrix = df[numeric_cols].corr()
    
    # Créer la heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matrice de Corrélation des Métriques')
    plt.tight_layout()
    plt.savefig(os.path.join(SENTIMENT_DIR, 'figures', 'correlation_heatmap.png'))
    plt.close()
    
    # 5. Graphique des top cryptomonnaies par score global
    plt.figure(figsize=(12, 8))
    
    # Trier par score global
    df_sorted = df.sort_values('global_score', ascending=False)
    
    # Créer le graphique à barres
    bars = plt.bar(df_sorted['symbol'], df_sorted['global_score'], color=plt.cm.plasma(df_sorted['global_score']))
    
    plt.xlabel('Cryptomonnaie')
    plt.ylabel('Score Global')
    plt.title('Classement des Cryptomonnaies par Score Global')
    plt.xticks(rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SENTIMENT_DIR, 'figures', 'global_score_ranking.png'))
    plt.close()

def generate_sentiment_report(df):
    """
    Génère un rapport d'analyse de sentiment
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données de sentiment
    """
    # Trier par score global
    df_sorted = df.sort_values('global_score', ascending=False)
    
    # Créer le rapport
    report = "# Rapport d'Analyse de Sentiment des Cryptomonnaies\n\n"
    report += f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Résumé\n\n"
    report += "Ce rapport présente une analyse du sentiment, de la performance et du potentiel des principales cryptomonnaies.\n\n"
    
    report += "## Top 5 Cryptomonnaies par Score Global\n\n"
    report += "| Rang | Symbole | Score Global | Sentiment | Performance | Potentiel | Score |\n"
    report += "|------|---------|-------------|-----------|-------------|-----------|-------|\n"
    
    for i, row in df_sorted.head(5).iterrows():
        report += f"| {int(row['rank'])} | {row['symbol']} | {row['global_score']:.3f} | {row['sentiment']:.1f}% | {row['performance']:.2f}% | {row['potential']:.2f}% | {row['score']:.2f} |\n"
    
    report += "\n## Analyse Détaillée\n\n"
    
    # Ajouter des analyses détaillées pour chaque cryptomonnaie du top 5
    for i, row in df_sorted.head(5).iterrows():
        report += f"### {row['symbol']}\n\n"
        report += f"- **Sentiment**: {row['sentiment']:.1f}%\n"
        report += f"- **Performance**: {row['performance']:.2f}%\n"
        report += f"- **Potentiel**: {row['potential']:.2f}%\n"
        report += f"- **Score**: {row['score']:.2f}\n"
        report += f"- **Rating**: {row['rating']}\n"
        report += f"- **Score Global**: {row['global_score']:.3f}\n\n"
        
        # Ajouter une analyse textuelle
        if row['sentiment'] >= 90:
            sentiment_text = "très positif"
        elif row['sentiment'] >= 70:
            sentiment_text = "positif"
        elif row['sentiment'] >= 50:
            sentiment_text = "neutre à positif"
        else:
            sentiment_text = "mitigé"
        
        if row['performance'] >= 15:
            performance_text = "excellente"
        elif row['performance'] >= 10:
            performance_text = "très bonne"
        elif row['performance'] >= 5:
            performance_text = "bonne"
        else:
            performance_text = "modérée"
        
        if row['potential'] >= 100:
            potential_text = "extrêmement élevé"
        elif row['potential'] >= 50:
            potential_text = "très élevé"
        elif row['potential'] >= 20:
            potential_text = "élevé"
        else:
            potential_text = "modéré"
        
        report += f"Le sentiment pour {row['symbol']} est {sentiment_text}, avec une {performance_text} performance récente. "
        report += f"Le potentiel de croissance est {potential_text}, ce qui en fait "
        
        if row['rank'] <= 3:
            report += "une option d'investissement très attractive à court terme.\n\n"
        elif row['rank'] <= 5:
            report += "une option d'investissement intéressante à surveiller de près.\n\n"
        else:
            report += "une option à considérer dans un portefeuille diversifié.\n\n"
    
    report += "## Corrélations et Observations\n\n"
    
    # Calculer quelques corrélations clés
    corr_perf_pot = df['performance'].corr(df['potential'])
    corr_sent_perf = df['sentiment'].corr(df['performance'])
    corr_sent_pot = df['sentiment'].corr(df['potential'])
    
    report += f"- Corrélation entre performance et potentiel: {corr_perf_pot:.3f}\n"
    report += f"- Corrélation entre sentiment et performance: {corr_sent_perf:.3f}\n"
    report += f"- Corrélation entre sentiment et potentiel: {corr_sent_pot:.3f}\n\n"
    
    report += "### Observations Générales\n\n"
    
    # Ajouter des observations générales
    if corr_perf_pot > 0.5:
        report += "- Les cryptomonnaies avec une bonne performance récente tendent également à avoir un potentiel de croissance plus élevé.\n"
    
    if corr_sent_perf > 0.5:
        report += "- Le sentiment positif est fortement corrélé avec la performance récente.\n"
    
    if corr_sent_pot > 0.5:
        report += "- Le sentiment positif est un bon indicateur du potentiel de croissance future.\n"
    
    # Identifier la cryptomonnaie avec le meilleur rapport risque/récompense
    best_risk_reward = df.loc[df['risk'].idxmin()]
    report += f"- {best_risk_reward['symbol']} présente le meilleur rapport risque/récompense avec un risque de {best_risk_reward['risk']:.2f}% et un potentiel de {best_risk_reward['potential']:.2f}%.\n"
    
    # Enregistrer le rapport
    with open(os.path.join(SENTIMENT_DIR, 'sentiment_report.md'), 'w') as f:
        f.write(report)
    
    return report

def main():
    """
    Fonction principale pour l'analyse de sentiment
    """
    print("Génération de l'analyse de sentiment basée sur les données fournies...")
    
    # Créer les données de sentiment manuelles
    df = create_manual_sentiment_data()
    
    # Calculer des métriques supplémentaires
    df = calculate_sentiment_metrics(df)
    
    # Générer des visualisations
    generate_sentiment_visualizations(df)
    
    # Générer un rapport
    generate_sentiment_report(df)
    
    print("Analyse de sentiment terminée avec succès.")
    print(f"Résultats enregistrés dans {SENTIMENT_DIR}")
    
    # Retourner les top 5 cryptomonnaies
    return df.sort_values('global_score', ascending=False).head(5)

if __name__ == "__main__":
    main()
