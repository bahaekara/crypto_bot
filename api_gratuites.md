# Liste des API gratuites pour le bot crypto

## API Twitter
- **Description** : Utilisée pour l'analyse de sentiment des cryptomonnaies
- **Site d'inscription** : https://developer.twitter.com/
- **Limites gratuites** : Accès limité aux tweets récents, 500,000 tweets par mois
- **Configuration** : Ajouter les clés dans le fichier `.env` :
  ```
  TWITTER_API_KEY=votre_clé_api_twitter
  TWITTER_API_SECRET=votre_secret_api_twitter
  TWITTER_BEARER_TOKEN=votre_token_bearer_twitter
  ```

## API Etherscan
- **Description** : Utilisée pour obtenir des données on-chain Ethereum
- **Site d'inscription** : https://etherscan.io/register
- **Limites gratuites** : 100,000 requêtes par jour
- **Configuration** : Ajouter la clé dans le fichier `.env` :
  ```
  ETHERSCAN_API_KEY=votre_clé_api_etherscan
  ```

## API Blockchair
- **Description** : Utilisée pour obtenir des données on-chain Bitcoin et autres cryptomonnaies
- **Site d'inscription** : https://blockchair.com/api
- **Limites gratuites** : 1,000 requêtes par jour
- **Configuration** : Ajouter la clé dans le fichier `.env` :
  ```
  BLOCKCHAIR_API_KEY=votre_clé_api_blockchair
  ```

## API CoinGecko
- **Description** : Utilisée pour obtenir des données de prix et de marché
- **Site d'inscription** : Pas d'inscription nécessaire pour le plan gratuit
- **URL** : https://www.coingecko.com/en/api
- **Limites gratuites** : 10-50 appels par minute
- **Configuration** : Aucune clé nécessaire pour le plan gratuit

## API Yahoo Finance (via yfinance)
- **Description** : Utilisée pour obtenir des données historiques de prix
- **Site d'inscription** : Pas d'inscription nécessaire
- **Limites gratuites** : Utilisation raisonnable
- **Configuration** : Aucune clé nécessaire

## Alternatives gratuites recommandées

### API CoinMarketCap
- **Description** : Alternative à CoinGecko pour les données de marché
- **Site d'inscription** : https://coinmarketcap.com/api/
- **Limites gratuites** : 10,000 appels par mois
- **Configuration** : Ajouter la clé dans le fichier `.env` :
  ```
  COINMARKETCAP_API_KEY=votre_clé_api_coinmarketcap
  ```

### API Binance
- **Description** : Données de marché en temps réel
- **Site d'inscription** : https://www.binance.com/en/register
- **Limites gratuites** : 1,200 requêtes par minute
- **Configuration** : Ajouter les clés dans le fichier `.env` :
  ```
  BINANCE_API_KEY=votre_clé_api_binance
  BINANCE_API_SECRET=votre_secret_api_binance
  ```
