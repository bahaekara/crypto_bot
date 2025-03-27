# Newsletter Hebdomadaire Crypto - Semaine {{ week }}

## Aperçu du Marché

Pour la semaine {{ week }}, notre analyse a identifié {{ buy_signals|length }} cryptomonnaies avec un signal d'achat, {{ hold_signals|length }} avec un signal de conservation et {{ sell_signals|length }} avec un signal de vente. Le marché montre une forte tendance {{ market_trend }} avec un potentiel de croissance moyen de {{ "%.2f"|format(avg_growth) }}% sur les 7 prochains jours. La cryptomonnaie la plus prometteuse cette semaine est {{ top_performers[0].symbol }} avec un potentiel de croissance de {{ "%.2f"|format(top_performers[0].predicted_change) }}%.

## Top Opportunités d'Investissement

{% for crypto in top_performers[:5] %}
### {{ crypto.symbol }}

- **Prix actuel:** {{ "%.2f"|format(crypto.current_price) }} USD
- **Prix prédit (7j):** {{ "%.2f"|format(crypto.predicted_price) }} USD
- **Variation prédite:** {{ "%.2f"|format(crypto.predicted_change) }}%
- **Sentiment:** {{ "%.1f"|format(sentiment.get(crypto.symbol.split('-')[0], 50)) }}%
- **Performance récente:** {{ "%.2f"|format(crypto.recent_performance) }}%
- **Potentiel:** {{ "%.2f"|format(crypto.potential) }}%
- **Signal:** {{ crypto.signal }}

{% endfor %}

## Recommandation de Portefeuille

Basé sur notre analyse, nous recommandons la répartition de portefeuille suivante pour la semaine à venir:

{% for item in portfolio[:5] %}
- **{{ item.symbol }}**: {{ item.weight }}%
{% endfor %}


## Tableau Récapitulatif

| Symbole | Prix Actuel | Variation Prédite | Signal | Sentiment | Potentiel |
|---------|-------------|-------------------|--------|-----------|----------|
{% for crypto in predictions %}
| {{ crypto.symbol }} | {{ "%.2f"|format(crypto.current_price) }} USD | {{ "%.2f"|format(crypto.predicted_change) }}% | {{ crypto.signal }} | {{ "%.1f"|format(sentiment.get(crypto.symbol.split('-')[0], 50)) }}% | {{ "%.2f"|format(crypto.potential) }}% |
{% endfor %}

## Perspectives à Long Terme

Pour les investisseurs à long terme, nous recommandons de se concentrer sur les cryptomonnaies ayant des fondamentaux solides et une adoption croissante. Parmi les cryptomonnaies analysées, les suivantes présentent un bon potentiel à long terme:

{% for crypto in predictions|sort(attribute='potential', reverse=True)[:4] %}
- **{{ crypto.symbol }}**: Sentiment de {{ "%.1f"|format(sentiment.get(crypto.symbol.split('-')[0], 50)) }}%, potentiel de {{ "%.2f"|format(crypto.potential) }}%
{% endfor %}




*Cette newsletter est générée automatiquement par un bot d'analyse de cryptomonnaies. Les prédictions sont basées sur des indicateurs techniques et l'analyse de sentiment.*

*Les informations fournies ne constituent pas des conseils d'investissement. Investissez de manière responsable et à vos propres risques.*