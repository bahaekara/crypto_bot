import sys
import os
import json
import pandas as pd
import requests
from datetime import datetime
import logging
from dotenv import load_dotenv

from analysis.confidence_indicators import ANALYSIS_DIR

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'onchain_analysis.log'), 'a')
    ]
)
logger = logging.getLogger('onchain_analysis')

# Créer le répertoire de logs s'il n'existe pas
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'), exist_ok=True)

# Définir les chemins des dossiers de manière relative
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ONCHAIN_DIR = os.path.join(DATA_DIR, 'onchain')
os.makedirs(ONCHAIN_DIR, exist_ok=True)
os.makedirs(os.path.join(ANALYSIS_DIR, 'onchain_figures'), exist_ok=True)

# Client API réel pour Etherscan
class EtherscanClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('ETHERSCAN_API_KEY')
        
        # Vérifier si la clé API est disponible
        if not self.api_key:
            logger.warning("Aucune clé Etherscan fournie dans les variables d'environnement")
        else:
            logger.info("Client Etherscan API initialisé avec clé API")
    
    def call_api(self, endpoint, params=None):
        """
        Appelle l'API Etherscan
        
        Args:
            endpoint (str): Point d'entrée de l'API
            params (dict): Paramètres de la requête
        
        Returns:
            dict: Résultat de l'API
        """
        if not self.api_key:
            logger.error("Clé API Etherscan non disponible")
            return None
        
        base_url = "https://api.etherscan.io/api"
        
        # Ajouter la clé API aux paramètres
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        try:
            logger.info(f"Appel à l'API Etherscan: {endpoint} avec paramètres {params}")
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Vérifier si l'API a retourné une erreur
                if 'status' in data and data['status'] == '0':
                    logger.error(f"Erreur API Etherscan: {data.get('message', 'Unknown error')}")
                    return None
                
                logger.info("Données Etherscan récupérées avec succès")
                return data
            else:
                logger.error(f"Erreur API Etherscan: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API Etherscan: {e}")
            return None

# Client API réel pour Blockchair
class BlockchairClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('BLOCKCHAIR_API_KEY')
        
        # Vérifier si la clé API est disponible
        if not self.api_key or self.api_key == "YOUR_BLOCKCHAIR_API_KEY":
            logger.warning("Aucune clé Blockchair valide fournie dans les variables d'environnement")
            self.use_data_api = True
        else:
            self.use_data_api = False
            logger.info("Client Blockchair API initialisé avec clé API")
    
    def call_api(self, endpoint, params=None):
        """
        Appelle l'API Blockchair
        
        Args:
            endpoint (str): Point d'entrée de l'API
            params (dict): Paramètres de la requête
        
        Returns:
            dict: Résultat de l'API
        """
        if self.use_data_api:
            # Utiliser l'API de données
            try:
                import sys
                sys.path.append('/opt/.manus/.sandbox-runtime')
                from data_api import ApiClient
                
                data_client = ApiClient()
                
                # Adapter l'endpoint et les paramètres au format de l'API de données
                # Ceci est un exemple, à adapter selon les besoins réels
                result = data_client.call_api(endpoint, query=params)
                
                logger.info(f"Données récupérées via l'API de données pour {endpoint}")
                return result
            except Exception as e:
                logger.error(f"Erreur lors de l'utilisation de l'API de données Blockchair: {e}")
                return None
        else:
            # Utiliser l'API Blockchair avec la clé API
            base_url = "https://api.blockchair.com"
            
            # Construire l'URL complète
            url = f"{base_url}/{endpoint}"
            
            # Ajouter la clé API aux paramètres
            if params is None:
                params = {}
            params['key'] = self.api_key
            
            try:
                logger.info(f"Appel à l'API Blockchair: {url} avec paramètres {params}")
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Vérifier si l'API a retourné une erreur
                    if 'error' in data and data['error']:
                        logger.error(f"Erreur API Blockchair: {data['error']}")
                        return None
                    
                    logger.info("Données Blockchair récupérées avec succès")
                    return data
                else:
                    logger.error(f"Erreur API Blockchair: {response.status_code} - {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Erreur lors de l'appel à l'API Blockchair: {e}")
                return None

# Créer les clients API
etherscan_client = EtherscanClient()
blockchair_client = BlockchairClient()

def get_ethereum_data(address=None):
    """
    Récupère les données on-chain Ethereum
    
    Args:
        address (str, optional): Adresse Ethereum à analyser. Si None, utilise des adresses prédéfinies.
    
    Returns:
        dict: Données on-chain Ethereum
    """
    # Si aucune adresse n'est fournie, utiliser des adresses de contrats populaires
    if address is None:
        # Adresses de contrats populaires (exemples)
        addresses = [
            "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap V2 Router
            "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",  # UNI Token
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
            "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"   # WBTC
        ]
    else:
        addresses = [address]
    
    results = {}
    
    for addr in addresses:
        try:
            # Récupérer le solde ETH
            params = {
                'module': 'account',
                'action': 'balance',
                'address': addr,
                'tag': 'latest'
            }
            balance_data = etherscan_client.call_api('', params)
            
            if balance_data and 'result' in balance_data:
                eth_balance = int(balance_data['result']) / 1e18  # Convertir de wei à ETH
            else:
                eth_balance = 0
            
            # Récupérer les transactions
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': addr,
                'startblock': 0,
                'endblock': 99999999,
                'page': 1,
                'offset': 10,
                'sort': 'desc'
            }
            tx_data = etherscan_client.call_api('', params)
            
            transactions = []
            if tx_data and 'result' in tx_data and isinstance(tx_data['result'], list):
                transactions = tx_data['result']
            
            # Récupérer les transferts de tokens ERC-20
            params = {
                'module': 'account',
                'action': 'tokentx',
                'address': addr,
                'page': 1,
                'offset': 10,
                'sort': 'desc'
            }
            token_tx_data = etherscan_client.call_api('', params)
            
            token_transfers = []
            if token_tx_data and 'result' in token_tx_data and isinstance(token_tx_data['result'], list):
                token_transfers = token_tx_data['result']
            
            # Stocker les résultats
            results[addr] = {
                'eth_balance': eth_balance,
                'transactions': transactions,
                'token_transfers': token_transfers
            }
            
            logger.info(f"Données Ethereum récupérées pour l'adresse {addr}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données Ethereum pour l'adresse {addr}: {e}")
            results[addr] = {
                'eth_balance': 0,
                'transactions': [],
                'token_transfers': []
            }
    
    return results

def get_bitcoin_data(address=None):
    """
    Récupère les données on-chain Bitcoin
    
    Args:
        address (str, optional): Adresse Bitcoin à analyser. Si None, utilise des adresses prédéfinies.
    
    Returns:
        dict: Données on-chain Bitcoin
    """
    # Si aucune adresse n'est fournie, utiliser des adresses populaires
    if address is None:
        # Adresses Bitcoin populaires (exemples)
        addresses = [
            "1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ",  # Binance Cold Wallet
            "3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64",  # Huobi Cold Wallet
            "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",  # Largest BTC Wallet
            "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",  # Grayscale Bitcoin Trust
            "385cR5DM96n1HvBDMzLHPYcw89fZAXULJP"   # Bitfinex Cold Wallet
        ]
    else:
        addresses = [address]
    
    results = {}
    
    for addr in addresses:
        try:
            # Récupérer les données de l'adresse
            endpoint = f"bitcoin/dashboards/address/{addr}"
            address_data = blockchair_client.call_api(endpoint)
            
            if address_data and 'data' in address_data:
                # Extraire les données pertinentes
                addr_info = address_data['data'].get(addr, {})
                
                # Stocker les résultats
                results[addr] = addr_info
                
                logger.info(f"Données Bitcoin récupérées pour l'adresse {addr}")
            else:
                logger.warning(f"Aucune donnée Bitcoin disponible pour l'adresse {addr}")
                results[addr] = {}
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données Bitcoin pour l'adresse {addr}: {e}")
            results[addr] = {}
    
    return results

def analyze_crypto_onchain(symbol, name):
    """
    Analyse on-chain pour une cryptomonnaie
    
    Args:
        symbol (str): Symbole de la cryptomonnaie (ex: BTC-USD)
        name (str): Nom de la cryptomonnaie (ex: Bitcoin)
    
    Returns:
        dict: Résultats de l'analyse on-chain
    """
    logger.info(f"Analyzing on-chain data for {symbol}...")
    
    # Initialiser les résultats
    results = {
        'symbol': symbol,
        'name': name,
        'blockchain': 'Unknown',
        'metrics': {},
        'addresses': {},
        'transactions': [],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # Déterminer la blockchain en fonction du symbole
        if symbol.startswith('BTC-'):
            results['blockchain'] = 'Bitcoin'
            
            # Récupérer les données on-chain Bitcoin
            btc_data = get_bitcoin_data()
            
            if btc_data:
                # Extraire et traiter les métriques
                metrics = {
                    'active_addresses': 0,
                    'transaction_count': 0,
                    'average_transaction_value': 0,
                    'total_fees': 0,
                    'mempool_size': 0
                }
                
                # Calculer les métriques à partir des données
                for addr, data in btc_data.items():
                    if 'address' in data:
                        addr_info = data['address']
                        
                        # Mettre à jour les métriques
                        metrics['transaction_count'] += addr_info.get('transaction_count', 0)
                        
                        # Stocker les informations d'adresse
                        results['addresses'][addr] = {
                            'balance': addr_info.get('balance', 0) / 1e8,  # Convertir de satoshi à BTC
                            'received': addr_info.get('received', 0) / 1e8,
                            'spent': addr_info.get('spent', 0) / 1e8,
                            'transaction_count': addr_info.get('transaction_count', 0)
                        }
                    
                    if 'transactions' in data:
                        # Ajouter les transactions
                        for tx in data['transactions'][:10]:  # Limiter à 10 transactions
                            results['transactions'].append({
                                'hash': tx.get('hash', ''),
                                'time': tx.get('time', ''),
                                'size': tx.get('size', 0),
                                'weight': tx.get('weight', 0),
                                'fee': tx.get('fee', 0) / 1e8,
                                'input_count': tx.get('input_count', 0),
                                'output_count': tx.get('output_count', 0),
                                'input_total': tx.get('input_total', 0) / 1e8,
                                'output_total': tx.get('output_total', 0) / 1e8
                            })
                
                # Mettre à jour les métriques
                results['metrics'] = metrics
            
        elif symbol.startswith('ETH-') or symbol in ['UNI-USD', 'LINK-USD', 'AAVE-USD', 'MKR-USD', 'COMP-USD', 'SNX-USD', 'BAL-USD', 'CRV-USD', 'SUSHI-USD', '1INCH-USD', 'YFI-USD', 'LRC-USD']:
            results['blockchain'] = 'Ethereum'
            
            # Récupérer les données on-chain Ethereum
            eth_data = get_ethereum_data()
            
            if eth_data:
                # Extraire et traiter les métriques
                metrics = {
                    'active_addresses': 0,
                    'transaction_count': 0,
                    'average_gas_price': 0,
                    'total_fees': 0,
                    'pending_transactions': 0
                }
                
                # Calculer les métriques à partir des données
                for addr, data in eth_data.items():
                    # Mettre à jour les métriques
                    metrics['transaction_count'] += len(data.get('transactions', []))
                    
                    # Stocker les informations d'adresse
                    results['addresses'][addr] = {
                        'balance': data.get('eth_balance', 0),
                        'transaction_count': len(data.get('transactions', [])),
                        'token_transfer_count': len(data.get('token_transfers', []))
                    }
                    
                    # Ajouter les transactions
                    for tx in data.get('transactions', [])[:10]:  # Limiter à 10 transactions
                        results['transactions'].append({
                            'hash': tx.get('hash', ''),
                            'timeStamp': tx.get('timeStamp', ''),
                            'from': tx.get('from', ''),
                            'to': tx.get('to', ''),
                            'value': int(tx.get('value', 0)) / 1e18,  # Convertir de wei à ETH
                            'gas': int(tx.get('gas', 0)),
                            'gasPrice': int(tx.get('gasPrice', 0)) / 1e9,  # Convertir en Gwei
                            'isError': tx.get('isError', '0')
                        })
                
                # Mettre à jour les métriques
                results['metrics'] = metrics
        
        else:
            logger.warning(f"Blockchain non prise en charge pour {symbol}")
        
        logger.info(f"On-chain analysis completed for {symbol}")
        
    except Exception as e:
        logger.error(f"Error during on-chain analysis for {symbol}: {e}")
    
    return results

def analyze_all_cryptos():
    """
    Analyse on-chain pour toutes les cryptomonnaies
    """
    # Importer la liste des cryptomonnaies
    sys.path.append(os.path.join(BASE_DIR, 'data'))
    from crypto_list import ALL_CRYPTOS
    
    # Analyser chaque cryptomonnaie
    for crypto in ALL_CRYPTOS:
        symbol = crypto['symbol']
        name = crypto['name']
        
        # Analyser les données on-chain
        results = analyze_crypto_onchain(symbol, name)
        
        # Enregistrer les résultats en JSON
        file_path = os.path.join(ONCHAIN_DIR, f"{symbol.replace('-', '_')}_onchain.json")
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"On-chain data saved to {file_path}")
    
    logger.info(f"On-chain analysis completed for all cryptocurrencies")

# Exécuter l'analyse on-chain
if __name__ == "__main__":
    analyze_all_cryptos()
