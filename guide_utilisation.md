# Guide d'utilisation du Bot d'Analyse de Cryptomonnaies

## Introduction

Ce guide explique comment installer, configurer et utiliser le Bot d'Analyse de Cryptomonnaies amélioré. Ce bot vous permet d'analyser les tendances du marché des cryptomonnaies, d'effectuer des analyses techniques et de sentiment, et de générer des prédictions et des newsletters.

## Prérequis

- **Système d'exploitation** : Windows 10/11 ou Linux
- **Python** : Version 3.8 ou supérieure (compatible avec les versions récentes)
- **Connexion Internet** : Nécessaire pour récupérer les données en temps réel
- **Clés API** : Facultatives mais recommandées pour des données complètes (voir section Configuration des API)

## Installation

### Étape 1 : Téléchargement et extraction

1. Téléchargez le fichier zip du bot
2. Extrayez le contenu dans un dossier de votre choix, par exemple : `C:\Users\[votre_nom]\Documents\crypto_bot`

### Étape 2 : Installation des dépendances

#### Sous Windows

1. Ouvrez une invite de commande (cmd) ou PowerShell
2. Naviguez vers le dossier du bot :
   ```
   cd C:\Users\[votre_nom]\Documents\crypto_bot
   ```
3. Exécutez le script d'installation :
   ```
   install.bat
   ```
   
   Ou manuellement :
   ```
   python -m pip install -r requirements.txt
   ```

#### Sous Linux

1. Ouvrez un terminal
2. Naviguez vers le dossier du bot :
   ```
   cd /chemin/vers/crypto_bot
   ```
3. Installez les dépendances :
   ```
   pip3 install -r requirements.txt
   ```

## Configuration des API

Pour obtenir des données réelles et complètes, vous devez configurer les clés API suivantes :

### 1. Configuration du fichier .env

1. Renommez le fichier `.env.example` en `.env`
2. Ouvrez le fichier `.env` avec un éditeur de texte
3. Ajoutez vos clés API comme indiqué ci-dessous

### 2. API Twitter (pour l'analyse de sentiment)

1. Créez un compte développeur sur [Twitter Developer Portal](https://developer.twitter.com/)
2. Créez un projet et une application pour obtenir les clés API
3. Ajoutez les clés dans le fichier `.env` :
   ```
   TWITTER_API_KEY=votre_clé_api_twitter
   TWITTER_API_SECRET=votre_secret_api_twitter
   TWITTER_BEARER_TOKEN=votre_token_bearer_twitter
   ```

### 3. API Etherscan (pour les données on-chain Ethereum)

1. Inscrivez-vous sur [Etherscan](https://etherscan.io/register)
2. Créez une clé API dans votre compte
3. Ajoutez la clé dans le fichier `.env` :
   ```
   ETHERSCAN_API_KEY=votre_clé_api_etherscan
   ```

### 4. API Blockchair (pour les données on-chain Bitcoin)

1. Inscrivez-vous sur [Blockchair API](https://blockchair.com/api)
2. Obtenez une clé API
3. Ajoutez la clé dans le fichier `.env` :
   ```
   BLOCKCHAIR_API_KEY=votre_clé_api_blockchair
   ```

## Utilisation

### Exécution complète

Pour exécuter toutes les fonctionnalités du bot en séquence :

#### Sous Windows

1. Double-cliquez sur `run_all.bat`

   Ou via ligne de commande :
   ```
   python scripts/run_all.py
   ```

#### Sous Linux

```
python3 scripts/run_all.py
```

Le bot effectuera les opérations suivantes :
- Collecte des données historiques de prix via Yahoo Finance
- Analyse technique des indicateurs
- Analyse de sentiment via Twitter
- Analyse on-chain via Etherscan/Blockchair
- Génération de prédictions
- Création de newsletters quotidiennes et hebdomadaires

### Exécution des composants individuels

Vous pouvez également exécuter chaque composant séparément :

#### Sous Windows

- Analyse technique : `run_technical_analysis.bat` ou `python scripts/run_technical.py`
- Analyse de sentiment : `run_sentiment_analysis.bat` ou `python scripts/run_sentiment.py`
- Analyse on-chain : `run_onchain_analysis.bat` ou `python scripts/run_onchain.py`
- Prédictions : `run_prediction.bat` ou `python scripts/run_prediction.py`
- Newsletters : `run_newsletters.bat` ou `python scripts/run_newsletters.py`

#### Sous Linux

- Analyse technique : `python3 scripts/run_technical.py`
- Analyse de sentiment : `python3 scripts/run_sentiment.py`
- Analyse on-chain : `python3 scripts/run_onchain.py`
- Prédictions : `python3 scripts/run_prediction.py`
- Newsletters : `python3 scripts/run_newsletters.py`

## Personnalisation

### Modification de la liste des cryptomonnaies

Pour modifier la liste des cryptomonnaies analysées :

1. Ouvrez le fichier `data/crypto_list.py`
2. Modifiez les listes `MAJOR_CRYPTOS`, `EMERGING_CRYPTOS`, `DEFI_CRYPTOS` ou `LAYER2_CRYPTOS`
3. Sauvegardez le fichier

### Personnalisation des paramètres d'analyse

Pour modifier les paramètres d'analyse technique :

1. Ouvrez le fichier `analysis/technical.py`
2. Modifiez les paramètres des indicateurs (périodes des moyennes mobiles, RSI, etc.)
3. Sauvegardez le fichier

### Personnalisation des newsletters

Pour personnaliser les newsletters :

1. Modifiez les templates HTML dans le dossier `newsletters/templates/`
2. Ajustez les paramètres de génération dans `newsletters/generator.py`

## Résultats et fichiers générés

Les résultats sont générés dans les dossiers suivants :

- **Données historiques** : `data/historical/`
  - Fichiers CSV contenant les données de prix historiques

- **Analyse technique** : `analysis/results/`
  - Fichiers CSV contenant les indicateurs techniques
  - Graphiques dans `analysis/figures/`

- **Analyse de sentiment** : `data/sentiment/`
  - Fichiers JSON contenant les analyses de sentiment
  - Graphiques dans `data/sentiment/figures/`

- **Analyse on-chain** : `data/onchain/`
  - Fichiers JSON contenant les données on-chain

- **Prédictions** : `prediction/`
  - Fichier CSV `prediction_results.csv` contenant les prédictions
  - Rapport Markdown `prediction_report.md`
  - Graphiques dans `prediction/figures/`

- **Newsletters** : `newsletters/daily/` et `newsletters/weekly/`
  - Fichiers HTML et Markdown des newsletters générées

## Dépannage

### Problèmes courants

1. **Erreur d'installation des dépendances**
   - Vérifiez que Python est correctement installé et dans le PATH
   - Essayez d'installer les dépendances une par une :
     ```
     pip install pandas numpy matplotlib seaborn scikit-learn
     pip install jinja2 markdown textblob yfinance
     pip install python-dotenv tweepy etherscan-python blockchair
     ```

2. **Erreur d'accès aux API**
   - Vérifiez que les clés API sont correctement configurées dans le fichier `.env`
   - Assurez-vous que les clés API sont valides et actives
   - Vérifiez votre connexion Internet

3. **Erreur lors de l'exécution**
   - Vérifiez les messages d'erreur dans la console
   - Assurez-vous que tous les dossiers nécessaires existent
   - Vérifiez les permissions d'écriture dans les dossiers de sortie

### Logs et débogage

Les logs sont générés dans le dossier `logs/` et peuvent aider à diagnostiquer les problèmes.

## Remarques importantes

- Les API gratuites ont souvent des limites de taux. Si vous atteignez ces limites, certaines fonctionnalités pourraient être temporairement indisponibles.
- Pour une utilisation intensive, envisagez d'obtenir des API premium.
- Les prédictions générées ne constituent pas des conseils financiers. Utilisez-les à vos propres risques.
- Le bot utilise exclusivement des données réelles provenant des API configurées.

## Support et contact

Si vous rencontrez des problèmes ou avez des questions, veuillez consulter la documentation ou contacter le support technique.

---

© 2025 Bot d'Analyse de Cryptomonnaies - Tous droits réservés
