# Bot d'Analyse de Cryptomonnaies avec API Réelles

Ce README explique comment configurer et utiliser le bot d'analyse de cryptomonnaies avec des API réelles pour obtenir des données de marché, des analyses de sentiment et des données on-chain.

## Prérequis

- Windows 10 ou plus récent
- Python 3.8 ou plus récent
- Connexion Internet
- Clés API (optionnelles mais recommandées pour des données réelles)

## Installation

1. Extrayez le contenu du fichier zip dans le dossier `C:\Users\bahik\Desktop\crypto_bot`
2. Double-cliquez sur `install.bat` pour installer les dépendances requises
3. Renommez le fichier `env.template` en `.env` et ajoutez vos clés API

## Configuration des API

Pour utiliser des données réelles, vous devez obtenir des clés API gratuites auprès des services suivants:

1. **Twitter API** (pour l'analyse de sentiment)
   - Créez un compte développeur sur https://developer.twitter.com/
   - Créez un projet et une application pour obtenir les clés API
   - Ajoutez les clés dans le fichier `.env`

2. **Etherscan API** (pour les données on-chain Ethereum)
   - Inscrivez-vous sur https://etherscan.io/register
   - Créez une clé API dans votre compte
   - Ajoutez la clé dans le fichier `.env`

3. **Blockchair API** (pour les données on-chain Bitcoin)
   - Inscrivez-vous sur https://blockchair.com/api
   - Obtenez une clé API
   - Ajoutez la clé dans le fichier `.env`

Si vous n'ajoutez pas ces clés API, le bot fonctionnera quand même mais utilisera des données simulées ou limitées.

## Utilisation

### Exécution complète

Pour exécuter toutes les fonctionnalités du bot en séquence:

1. Double-cliquez sur `run_all.bat`
2. Le bot effectuera les opérations suivantes:
   - Collecte des données historiques de prix via Yahoo Finance
   - Analyse technique des indicateurs
   - Analyse de sentiment via Twitter
   - Analyse on-chain via Etherscan/Blockchair/CoinGecko
   - Génération de prédictions
   - Création de newsletters quotidiennes et hebdomadaires

### Exécution des composants individuels

Vous pouvez également exécuter chaque composant séparément:

- `run_technical_analysis.bat` - Exécute uniquement l'analyse technique
- `run_sentiment_analysis.bat` - Exécute uniquement l'analyse de sentiment
- `run_onchain_analysis.bat` - Exécute uniquement l'analyse on-chain
- `run_prediction.bat` - Exécute uniquement les prédictions
- `run_newsletters.bat` - Génère uniquement les newsletters

## Structure des fichiers

```
crypto_bot/
├── analysis/
│   ├── technical_indicators.py
│   ├── real_sentiment_analysis.py
│   ├── real_onchain_analysis.py
│   └── results/
├── data/
│   ├── collect_data.py
│   ├── crypto_list.py
│   ├── historical/
│   ├── sentiment/
│   └── onchain/
├── prediction/
│   └── prediction_system.py
├── newsletters/
│   ├── newsletter_generator.py
│   ├── daily/
│   └── weekly/
├── yahoo_finance_client.py
├── install.bat
├── run_all.bat
├── run_technical_analysis.bat
├── run_sentiment_analysis.bat
├── run_onchain_analysis.bat
├── run_prediction.bat
├── run_newsletters.bat
├── requirements.txt
├── .env
└── README.md
```

## Résultats

Les résultats sont générés dans les dossiers suivants:

- Données historiques: `data/historical/`
- Analyse de sentiment: `data/sentiment/`
- Analyse on-chain: `data/onchain/`
- Indicateurs techniques: `analysis/results/`
- Prédictions: `prediction/`
- Newsletters: `newsletters/daily/` et `newsletters/weekly/`

## Dépannage

Si vous rencontrez des problèmes:

1. Vérifiez que Python est correctement installé et dans le PATH
2. Assurez-vous que toutes les dépendances sont installées via `install.bat`
3. Vérifiez votre connexion Internet
4. Vérifiez que les clés API sont correctement configurées dans le fichier `.env`
5. Consultez les messages d'erreur dans la console pour plus d'informations

## Remarques importantes

- Les API gratuites ont souvent des limites de taux. Si vous atteignez ces limites, le bot basculera automatiquement vers des données simulées.
- Pour une utilisation intensive, envisagez d'obtenir des API premium.
- Les prédictions générées ne constituent pas des conseils financiers. Utilisez-les à vos propres risques.
