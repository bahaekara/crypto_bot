import pandas as pd
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import jinja2
import markdown
import base64
from io import BytesIO

# Définir les chemins des dossiers
DATA_DIR = r'C:\Users\bahik\Desktop\crypto_bot\data'
PREDICTION_DIR = r'C:\Users\bahik\Desktop\crypto_bot\prediction'
NEWSLETTER_DIR = r'C:\Users\bahik\Desktop\crypto_bot\newsletters'
os.makedirs(NEWSLETTER_DIR, exist_ok=True)
os.makedirs(os.path.join(NEWSLETTER_DIR, 'daily'), exist_ok=True)
os.makedirs(os.path.join(NEWSLETTER_DIR, 'weekly'), exist_ok=True)

def load_prediction_results():
    """
    Charge les résultats de prédiction
    
    Returns:
        pandas.DataFrame: DataFrame contenant les résultats de prédiction
    """
    file_path = os.path.join(PREDICTION_DIR, 'prediction_results.csv')
    if not os.path.exists(file_path):
        print("No prediction results found")
        return None
    
    df = pd.read_csv(file_path)
    return df

def fig_to_base64(fig):
    """
    Convertit une figure matplotlib en image base64 pour l'inclusion dans HTML
    
    Args:
        fig: Figure matplotlib
    
    Returns:
        str: Image encodée en base64
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def create_performance_chart(results_df, top_n=5):
    """
    Crée un graphique de performance pour les top N cryptomonnaies
    
    Args:
        results_df (pandas.DataFrame): DataFrame contenant les résultats de prédiction
        top_n (int): Nombre de cryptomonnaies à inclure
    
    Returns:
        str: Image encodée en base64
    """
    # Trier par variation de prix prédite
    df_sorted = results_df.sort_values('percent_change', ascending=False).head(top_n)
    
    # Définir les couleurs en fonction du signal
    colors = []
    for signal in df_sorted['signal']:
        if signal == "ACHAT":
            colors.append('#4CAF50')  # Vert
        elif signal == "CONSERVER":
            colors.append('#FFC107')  # Jaune
        else:
            colors.append('#F44336')  # Rouge
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_sorted['symbol'], df_sorted['percent_change'], color=colors)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Cryptomonnaie')
    plt.ylabel('Variation de prix prédite (%)')
    plt.title('Top Cryptomonnaies par Potentiel de Croissance')
    plt.xticks(rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Convertir en base64
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    
    return img_str

def create_sentiment_chart(results_df, top_n=5):
    """
    Crée un graphique de sentiment pour les top N cryptomonnaies
    
    Args:
        results_df (pandas.DataFrame): DataFrame contenant les résultats de prédiction
        top_n (int): Nombre de cryptomonnaies à inclure
    
    Returns:
        str: Image encodée en base64
    """
    # Trier par sentiment
    df_sorted = results_df.sort_values('sentiment', ascending=False).head(top_n)
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_sorted['symbol'], df_sorted['sentiment'], color=plt.cm.viridis(df_sorted['sentiment']/100))
    
    plt.xlabel('Cryptomonnaie')
    plt.ylabel('Sentiment (%)')
    plt.title('Top Cryptomonnaies par Sentiment')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Convertir en base64
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    
    return img_str

def generate_daily_newsletter(results_df, date=None):
    """
    Génère une newsletter quotidienne
    
    Args:
        results_df (pandas.DataFrame): DataFrame contenant les résultats de prédiction
        date (str): Date de la newsletter (format YYYY-MM-DD)
    
    Returns:
        str: Contenu HTML de la newsletter
    """
    if results_df is None or results_df.empty:
        print("No prediction results for newsletter generation")
        return None
    
    # Utiliser la date actuelle si non spécifiée
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Charger le template Jinja2
    template_loader = jinja2.FileSystemLoader(searchpath=os.path.join(NEWSLETTER_DIR, 'templates'))
    template_env = jinja2.Environment(loader=template_loader)
    
    # Créer le template s'il n'existe pas
    template_dir = os.path.join(NEWSLETTER_DIR, 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    template_path = os.path.join(template_dir, 'daily_template.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Newsletter Quotidienne Crypto - {{ date }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        .header h1 {
            color: #4CAF50;
            margin-bottom: 5px;
        }
        .header p {
            color: #777;
            font-size: 14px;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            color: #2196F3;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        .crypto-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f9f9f9;
        }
        .buy {
            border-left: 5px solid #4CAF50;
        }
        .hold {
            border-left: 5px solid #FFC107;
        }
        .sell {
            border-left: 5px solid #F44336;
        }
        .crypto-card h3 {
            margin-top: 0;
            color: #333;
        }
        .crypto-card p {
            margin: 5px 0;
        }
        .crypto-card .signal {
            font-weight: bold;
            font-size: 14px;
            padding: 3px 8px;
            border-radius: 3px;
            display: inline-block;
            margin-top: 5px;
        }
        .buy-signal {
            background-color: #4CAF50;
            color: white;
        }
        .hold-signal {
            background-color: #FFC107;
            color: black;
        }
        .sell-signal {
            background-color: #F44336;
            color: white;
        }
        .chart {
            margin: 20px 0;
            text-align: center;
        }
        .chart img {
            max-width: 100%;
            height: auto;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 12px;
            color: #777;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Newsletter Quotidienne Crypto</h1>
        <p>Analyse et prédictions pour le {{ date }}</p>
    </div>
    
    <div class="section">
        <h2>Résumé du Marché</h2>
        <p>{{ market_summary }}</p>
    </div>
    
    <div class="section">
        <h2>Top Recommandations du Jour</h2>
        
        {% for crypto in top_cryptos %}
        <div class="crypto-card {% if crypto.signal == 'ACHAT' %}buy{% elif crypto.signal == 'CONSERVER' %}hold{% else %}sell{% endif %}">
            <h3>{{ crypto.symbol }}</h3>
            <p><strong>Prix actuel:</strong> {{ "%.2f"|format(crypto.current_price) }} USD</p>
            <p><strong>Prix prédit (7j):</strong> {{ "%.2f"|format(crypto.predicted_price) }} USD</p>
            <p><strong>Variation prédite:</strong> {{ "%.2f"|format(crypto.percent_change) }}%</p>
            <p><strong>Sentiment:</strong> {{ "%.1f"|format(crypto.sentiment) }}%</p>
            <div class="signal {% if crypto.signal == 'ACHAT' %}buy-signal{% elif crypto.signal == 'CONSERVER' %}hold-signal{% else %}sell-signal{% endif %}">
                {{ crypto.signal }}
            </div>
        </div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Graphique des Performances Prédites</h2>
        <div class="chart">
            <img src="data:image/png;base64,{{ performance_chart }}" alt="Performance Chart">
        </div>
    </div>
    
    <div class="section">
        <h2>Analyse de Sentiment</h2>
        <div class="chart">
            <img src="data:image/png;base64,{{ sentiment_chart }}" alt="Sentiment Chart">
        </div>
    </div>
    
    <div class="section">
        <h2>Tableau Récapitulatif</h2>
        <table>
            <tr>
                <th>Symbole</th>
                <th>Prix Actuel</th>
                <th>Variation Prédite</th>
                <th>Signal</th>
                <th>Sentiment</th>
            </tr>
            {% for crypto in all_cryptos %}
            <tr>
                <td>{{ crypto.symbol }}</td>
                <td>{{ "%.2f"|format(crypto.current_price) }} USD</td>
                <td>{{ "%.2f"|format(crypto.percent_change) }}%</td>
                <td>{{ crypto.signal }}</td>
                <td>{{ "%.1f"|format(crypto.sentiment) }}%</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="footer">
        <p>Cette newsletter est générée automatiquement par un bot d'analyse de cryptomonnaies. Les prédictions sont basées sur des indicateurs techniques et l'analyse de sentiment.</p>
        <p>Les informations fournies ne constituent pas des conseils d'investissement. Investissez de manière responsable et à vos propres risques.</p>
        <p>© {{ current_year }} Crypto Bot - Tous droits réservés</p>
    </div>
</body>
</html>""")
    
    # Charger le template
    template = template_env.get_template('daily_template.html')
    
    # Préparer les données pour le template
    top_cryptos = results_df.sort_values('percent_change', ascending=False).head(5).to_dict('records')
    all_cryptos = results_df.to_dict('records')
    
    # Générer un résumé du marché
    buy_count = len(results_df[results_df['signal'] == 'ACHAT'])
    hold_count = len(results_df[results_df['signal'] == 'CONSERVER'])
    sell_count = len(results_df[results_df['signal'] == 'VENTE'])
    
    market_summary = f"Aujourd'hui, notre analyse a identifié {buy_count} cryptomonnaies avec un signal d'achat, "
    market_summary += f"{hold_count} avec un signal de conservation et {sell_count} avec un signal de vente. "
    
    # Ajouter des informations sur les tendances
    avg_change = results_df['percent_change'].mean()
    if avg_change > 5:
        market_summary += "Le marché montre une forte tendance haussière avec un potentiel de croissance moyen de "
    elif avg_change > 0:
        market_summary += "Le marché montre une légère tendance haussière avec un potentiel de croissance moyen de "
    else:
        market_summary += "Le marché montre une tendance baissière avec un potentiel de croissance moyen de "
    
    market_summary += f"{avg_change:.2f}% sur les 7 prochains jours."
    
    # Créer les graphiques
    performance_chart = create_performance_chart(results_df)
    sentiment_chart = create_sentiment_chart(results_df)
    
    # Rendre le template
    html_content = template.render(
        date=date,
        market_summary=market_summary,
        top_cryptos=top_cryptos,
        all_cryptos=all_cryptos,
        performance_chart=performance_chart,
        sentiment_chart=sentiment_chart,
        current_year=datetime.now().year
    )
    
    # Enregistrer la newsletter
    output_path = os.path.join(NEWSLETTER_DIR, 'daily', f"newsletter_{date}.html")
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    # Créer également une version Markdown
    md_content = f"# Newsletter Quotidienne Crypto - {date}\n\n"
    md_content += "## Résumé du Marché\n\n"
    md_content += f"{market_summary}\n\n"
    md_content += "## Top Recommandations du Jour\n\n"
    
    for crypto in top_cryptos:
        md_content += f"### {crypto['symbol']}\n\n"
        md_content += f"- **Prix actuel:** {crypto['current_price']:.2f} USD\n"
        md_content += f"- **Prix prédit (7j):** {crypto['predicted_price']:.2f} USD\n"
        md_content += f"- **Variation prédite:** {crypto['percent_change']:.2f}%\n"
        md_content += f"- **Sentiment:** {crypto['sentiment']:.1f}%\n"
        md_content += f"- **Signal:** {crypto['signal']}\n\n"
    
    md_content += "## Tableau Récapitulatif\n\n"
    md_content += "| Symbole | Prix Actuel | Variation Prédite | Signal | Sentiment |\n"
    md_content += "|---------|-------------|-------------------|--------|------------|\n"
    
    for crypto in all_cryptos:
        md_content += f"| {crypto['symbol']} | {crypto['current_price']:.2f} USD | {crypto['percent_change']:.2f}% | {crypto['signal']} | {crypto['sentiment']:.1f}% |\n"
    
    md_content += "\n\n*Cette newsletter est générée automatiquement par un bot d'analyse de cryptomonnaies. Les prédictions sont basées sur des indicateurs techniques et l'analyse de sentiment.*\n\n"
    md_content += "*Les informations fournies ne constituent pas des conseils d'investissement. Investissez de manière responsable et à vos propres risques.*\n"
    
    md_output_path = os.path.join(NEWSLETTER_DIR, 'daily', f"newsletter_{date}.md")
    with open(md_output_path, 'w') as f:
        f.write(md_content)
    
    print(f"Daily newsletter generated for {date}")
    return html_content, md_content

def generate_weekly_newsletter(results_df, date=None, week_number=None):
    """
    Génère une newsletter hebdomadaire
    
    Args:
        results_df (pandas.DataFrame): DataFrame contenant les résultats de prédiction
        date (str): Date de la newsletter (format YYYY-MM-DD)
        week_number (int): Numéro de la semaine
    
    Returns:
        str: Contenu HTML de la newsletter
    """
    if results_df is None or results_df.empty:
        print("No prediction results for newsletter generation")
        return None
    
    # Utiliser la date actuelle si non spécifiée
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Calculer le numéro de la semaine si non spécifié
    if week_number is None:
        week_number = datetime.now().isocalendar()[1]
    
    # Charger le template Jinja2
    template_loader = jinja2.FileSystemLoader(searchpath=os.path.join(NEWSLETTER_DIR, 'templates'))
    template_env = jinja2.Environment(loader=template_loader)
    
    # Créer le template s'il n'existe pas
    template_dir = os.path.join(NEWSLETTER_DIR, 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    template_path = os.path.join(template_dir, 'weekly_template.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Newsletter Hebdomadaire Crypto - Semaine {{ week_number }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 10px;
        }
        .header h1 {
            color: #2196F3;
            margin-bottom: 5px;
        }
        .header p {
            color: #777;
            font-size: 14px;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            color: #2196F3;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        .crypto-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f9f9f9;
        }
        .buy {
            border-left: 5px solid #4CAF50;
        }
        .hold {
            border-left: 5px solid #FFC107;
        }
        .sell {
            border-left: 5px solid #F44336;
        }
        .crypto-card h3 {
            margin-top: 0;
            color: #333;
        }
        .crypto-card p {
            margin: 5px 0;
        }
        .crypto-card .signal {
            font-weight: bold;
            font-size: 14px;
            padding: 3px 8px;
            border-radius: 3px;
            display: inline-block;
            margin-top: 5px;
        }
        .buy-signal {
            background-color: #4CAF50;
            color: white;
        }
        .hold-signal {
            background-color: #FFC107;
            color: black;
        }
        .sell-signal {
            background-color: #F44336;
            color: white;
        }
        .chart {
            margin: 20px 0;
            text-align: center;
        }
        .chart img {
            max-width: 100%;
            height: auto;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 12px;
            color: #777;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .market-insight {
            background-color: #E3F2FD;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .market-insight h3 {
            color: #1565C0;
            margin-top: 0;
        }
        .portfolio-recommendation {
            background-color: #E8F5E9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .portfolio-recommendation h3 {
            color: #2E7D32;
            margin-top: 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Newsletter Hebdomadaire Crypto</h1>
        <p>Analyse et prédictions pour la semaine {{ week_number }} ({{ date }})</p>
    </div>
    
    <div class="section">
        <h2>Aperçu du Marché</h2>
        <div class="market-insight">
            <h3>Tendances de la Semaine</h3>
            <p>{{ market_summary }}</p>
        </div>
    </div>
    
    <div class="section">
        <h2>Top Opportunités d'Investissement</h2>
        
        {% for crypto in top_cryptos %}
        <div class="crypto-card {% if crypto.signal == 'ACHAT' %}buy{% elif crypto.signal == 'CONSERVER' %}hold{% else %}sell{% endif %}">
            <h3>{{ crypto.symbol }}</h3>
            <p><strong>Prix actuel:</strong> {{ "%.2f"|format(crypto.current_price) }} USD</p>
            <p><strong>Prix prédit (7j):</strong> {{ "%.2f"|format(crypto.predicted_price) }} USD</p>
            <p><strong>Variation prédite:</strong> {{ "%.2f"|format(crypto.percent_change) }}%</p>
            <p><strong>Sentiment:</strong> {{ "%.1f"|format(crypto.sentiment) }}%</p>
            <p><strong>Performance récente:</strong> {{ "%.2f"|format(crypto.performance) }}%</p>
            <p><strong>Potentiel:</strong> {{ "%.2f"|format(crypto.potential) }}%</p>
            <div class="signal {% if crypto.signal == 'ACHAT' %}buy-signal{% elif crypto.signal == 'CONSERVER' %}hold-signal{% else %}sell-signal{% endif %}">
                {{ crypto.signal }}
            </div>
        </div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Analyse Approfondie</h2>
        <div class="portfolio-recommendation">
            <h3>Recommandation de Portefeuille</h3>
            <p>{{ portfolio_recommendation }}</p>
        </div>
        
        <div class="chart">
            <h3>Potentiel de Croissance à 7 Jours</h3>
            <img src="data:image/png;base64,{{ performance_chart }}" alt="Performance Chart">
        </div>
        
        <div class="chart">
            <h3>Analyse de Sentiment</h3>
            <img src="data:image/png;base64,{{ sentiment_chart }}" alt="Sentiment Chart">
        </div>
    </div>
    
    <div class="section">
        <h2>Tableau Récapitulatif</h2>
        <table>
            <tr>
                <th>Symbole</th>
                <th>Prix Actuel</th>
                <th>Variation Prédite</th>
                <th>Signal</th>
                <th>Sentiment</th>
                <th>Potentiel</th>
            </tr>
            {% for crypto in all_cryptos %}
            <tr>
                <td>{{ crypto.symbol }}</td>
                <td>{{ "%.2f"|format(crypto.current_price) }} USD</td>
                <td>{{ "%.2f"|format(crypto.percent_change) }}%</td>
                <td>{{ crypto.signal }}</td>
                <td>{{ "%.1f"|format(crypto.sentiment) }}%</td>
                <td>{{ "%.2f"|format(crypto.potential) }}%</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Perspectives à Long Terme</h2>
        <p>{{ long_term_outlook }}</p>
    </div>
    
    <div class="footer">
        <p>Cette newsletter est générée automatiquement par un bot d'analyse de cryptomonnaies. Les prédictions sont basées sur des indicateurs techniques et l'analyse de sentiment.</p>
        <p>Les informations fournies ne constituent pas des conseils d'investissement. Investissez de manière responsable et à vos propres risques.</p>
        <p>© {{ current_year }} Crypto Bot - Tous droits réservés</p>
    </div>
</body>
</html>""")
    
    # Charger le template
    template = template_env.get_template('weekly_template.html')
    
    # Préparer les données pour le template
    top_cryptos = results_df.sort_values('percent_change', ascending=False).head(5).to_dict('records')
    all_cryptos = results_df.to_dict('records')
    
    # Générer un résumé du marché
    buy_count = len(results_df[results_df['signal'] == 'ACHAT'])
    hold_count = len(results_df[results_df['signal'] == 'CONSERVER'])
    sell_count = len(results_df[results_df['signal'] == 'VENTE'])
    
    market_summary = f"Pour la semaine {week_number}, notre analyse a identifié {buy_count} cryptomonnaies avec un signal d'achat, "
    market_summary += f"{hold_count} avec un signal de conservation et {sell_count} avec un signal de vente. "
    
    # Ajouter des informations sur les tendances
    avg_change = results_df['percent_change'].mean()
    if avg_change > 5:
        market_summary += "Le marché montre une forte tendance haussière avec un potentiel de croissance moyen de "
    elif avg_change > 0:
        market_summary += "Le marché montre une légère tendance haussière avec un potentiel de croissance moyen de "
    else:
        market_summary += "Le marché montre une tendance baissière avec un potentiel de croissance moyen de "
    
    market_summary += f"{avg_change:.2f}% sur les 7 prochains jours. "
    
    # Ajouter des informations sur les cryptomonnaies les plus performantes
    top_performer = results_df.loc[results_df['percent_change'].idxmax()]
    market_summary += f"La cryptomonnaie la plus prometteuse cette semaine est {top_performer['symbol']} avec un potentiel de croissance de {top_performer['percent_change']:.2f}%."
    
    # Générer une recommandation de portefeuille
    portfolio_recommendation = "Basé sur notre analyse, nous recommandons la répartition de portefeuille suivante pour la semaine à venir:\n\n"
    
    # Calculer les allocations recommandées
    total_allocation = 100
    top_cryptos_df = results_df.sort_values('percent_change', ascending=False).head(5)
    
    # Normaliser les variations prédites pour obtenir des allocations
    total_change = top_cryptos_df['percent_change'].sum()
    
    if total_change > 0:
        allocations = (top_cryptos_df['percent_change'] / total_change * total_allocation).round().astype(int)
        
        # Ajuster pour s'assurer que la somme est exactement 100%
        diff = total_allocation - allocations.sum()
        if diff != 0:
            # Ajouter la différence à l'allocation la plus importante
            max_idx = allocations.idxmax()
            allocations.loc[max_idx] += diff
        
        for i, (_, row) in enumerate(top_cryptos_df.iterrows()):
            portfolio_recommendation += f"- **{row['symbol']}**: {allocations.iloc[i]}%\n"
    else:
        portfolio_recommendation = "En raison des conditions de marché actuelles, nous recommandons de conserver principalement des stablecoins cette semaine et d'attendre de meilleures opportunités d'entrée."
    
    # Générer des perspectives à long terme
    long_term_outlook = "Pour les investisseurs à long terme, nous recommandons de se concentrer sur les cryptomonnaies ayant des fondamentaux solides et une adoption croissante. "
    
    # Identifier les cryptomonnaies avec un bon sentiment mais pas nécessairement une variation à court terme élevée
    long_term_candidates = results_df[(results_df['sentiment'] > 80) & (results_df['potential'] > 50)]
    
    if not long_term_candidates.empty:
        long_term_outlook += "Parmi les cryptomonnaies analysées, les suivantes présentent un bon potentiel à long terme:\n\n"
        for i, row in long_term_candidates.iterrows():
            long_term_outlook += f"- **{row['symbol']}**: Sentiment de {row['sentiment']:.1f}%, potentiel de {row['potential']:.2f}%\n"
    else:
        long_term_outlook += "Actuellement, aucune des cryptomonnaies analysées ne présente un profil particulièrement attractif pour l'investissement à long terme. Nous recommandons de rester prudent et d'attendre de meilleures opportunités."
    
    # Créer les graphiques
    performance_chart = create_performance_chart(results_df)
    sentiment_chart = create_sentiment_chart(results_df)
    
    # Rendre le template
    html_content = template.render(
        date=date,
        week_number=week_number,
        market_summary=market_summary,
        top_cryptos=top_cryptos,
        all_cryptos=all_cryptos,
        performance_chart=performance_chart,
        sentiment_chart=sentiment_chart,
        portfolio_recommendation=portfolio_recommendation,
        long_term_outlook=long_term_outlook,
        current_year=datetime.now().year
    )
    
    # Enregistrer la newsletter
    output_path = os.path.join(NEWSLETTER_DIR, 'weekly', f"newsletter_week_{week_number}.html")
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    # Créer également une version Markdown
    md_content = f"# Newsletter Hebdomadaire Crypto - Semaine {week_number}\n\n"
    md_content += "## Aperçu du Marché\n\n"
    md_content += f"{market_summary}\n\n"
    md_content += "## Top Opportunités d'Investissement\n\n"
    
    for crypto in top_cryptos:
        md_content += f"### {crypto['symbol']}\n\n"
        md_content += f"- **Prix actuel:** {crypto['current_price']:.2f} USD\n"
        md_content += f"- **Prix prédit (7j):** {crypto['predicted_price']:.2f} USD\n"
        md_content += f"- **Variation prédite:** {crypto['percent_change']:.2f}%\n"
        md_content += f"- **Sentiment:** {crypto['sentiment']:.1f}%\n"
        md_content += f"- **Performance récente:** {crypto['performance']:.2f}%\n"
        md_content += f"- **Potentiel:** {crypto['potential']:.2f}%\n"
        md_content += f"- **Signal:** {crypto['signal']}\n\n"
    
    md_content += "## Recommandation de Portefeuille\n\n"
    md_content += f"{portfolio_recommendation}\n\n"
    
    md_content += "## Tableau Récapitulatif\n\n"
    md_content += "| Symbole | Prix Actuel | Variation Prédite | Signal | Sentiment | Potentiel |\n"
    md_content += "|---------|-------------|-------------------|--------|-----------|----------|\n"
    
    for crypto in all_cryptos:
        md_content += f"| {crypto['symbol']} | {crypto['current_price']:.2f} USD | {crypto['percent_change']:.2f}% | {crypto['signal']} | {crypto['sentiment']:.1f}% | {crypto['potential']:.2f}% |\n"
    
    md_content += "\n## Perspectives à Long Terme\n\n"
    md_content += f"{long_term_outlook}\n\n"
    
    md_content += "\n\n*Cette newsletter est générée automatiquement par un bot d'analyse de cryptomonnaies. Les prédictions sont basées sur des indicateurs techniques et l'analyse de sentiment.*\n\n"
    md_content += "*Les informations fournies ne constituent pas des conseils d'investissement. Investissez de manière responsable et à vos propres risques.*\n"
    
    md_output_path = os.path.join(NEWSLETTER_DIR, 'weekly', f"newsletter_week_{week_number}.md")
    with open(md_output_path, 'w') as f:
        f.write(md_content)
    
    print(f"Weekly newsletter generated for week {week_number}")
    return html_content, md_content

def main():
    """
    Fonction principale pour la génération de newsletters
    """
    print("Génération des newsletters...")
    
    # Charger les résultats de prédiction
    results_df = load_prediction_results()
    
    if results_df is not None:
        # Générer la newsletter quotidienne
        today = datetime.now().strftime('%Y-%m-%d')
        daily_html, daily_md = generate_daily_newsletter(results_df, date=today)
        
        # Générer la newsletter hebdomadaire
        week_number = datetime.now().isocalendar()[1]
        weekly_html, weekly_md = generate_weekly_newsletter(results_df, date=today, week_number=week_number)
        
        print("Newsletters générées avec succès.")
        print(f"Newsletter quotidienne: {os.path.join(NEWSLETTER_DIR, 'daily', f'newsletter_{today}.html')}")
        print(f"Newsletter hebdomadaire: {os.path.join(NEWSLETTER_DIR, 'weekly', f'newsletter_week_{week_number}.html')}")
    else:
        print("Impossible de générer les newsletters: aucun résultat de prédiction disponible.")

if __name__ == "__main__":
    main()
