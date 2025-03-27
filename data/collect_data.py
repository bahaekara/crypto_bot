import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin de recherche
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les clients API réels
from yahoo_finance_client import YahooFinanceClient

# Définir les chemins des dossiers de manière relative
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
HISTORICAL_DIR = os.path.join(DATA_DIR, 'historical')
os.makedirs(HISTORICAL_DIR, exist_ok=True)

def collect_historical_data(symbol, client, period='3mo', interval='1d'):
    """
    Collecte les données historiques pour une cryptomonnaie
    
    Args:
        symbol (str): Symbole de la cryptomonnaie
        client: Client API
        period (str): Période de temps ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval (str): Intervalle de temps ('1m', '2m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo')
    
    Returns:
        pandas.DataFrame: DataFrame contenant les données historiques
    """
    print(f"Collecting data for {symbol}...")
    
    try:
        # Appeler l'API Yahoo Finance
        data = client.call_api('YahooFinance/get_stock_chart', {
            'symbol': symbol,
            'range': period,
            'interval': interval
        })
        
        if data is None or 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            print(f"API call failed for {symbol}")
            return None
        
        # Extraire les données
        chart_data = data['chart']['result'][0]
        timestamps = chart_data['timestamp']
        quote = chart_data['indicators']['quote'][0]
        
        # Créer un DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': quote['open'],
            'high': quote['high'],
            'low': quote['low'],
            'close': quote['close'],
            'volume': quote['volume']
        })
        
        # Convertir les timestamps en dates
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('date', inplace=True)
        
        # Supprimer la colonne timestamp
        df.drop('timestamp', axis=1, inplace=True)
        
        # Calculer des indicateurs supplémentaires
        df['returns'] = df['close'].pct_change(fill_method=None)
        df['volatility'] = df['returns'].rolling(window=7).std() * np.sqrt(252)
        
        return df
    
    except Exception as e:
        print(f"Error collecting data for {symbol}: {e}")
        return None

def collect_data_for_all_cryptos(crypto_list):
    """
    Collecte les données pour toutes les cryptomonnaies de la liste
    
    Args:
        crypto_list (list): Liste des dictionnaires contenant les symboles et noms des cryptomonnaies
    """
    print(f"Starting data collection for {len(crypto_list)} cryptocurrencies...")
    
    # Créer le client Yahoo Finance
    client = YahooFinanceClient()
    
    # Compteurs pour le résumé
    success_count = 0
    fail_count = 0
    failed_cryptos = []
    
    # Collecter les données pour chaque cryptomonnaie
    for crypto in crypto_list:
        symbol = crypto['symbol']
        
        # Collecter les données historiques
        df = collect_historical_data(symbol, client)
        
        if df is not None and not df.empty:
            # Enregistrer les données en CSV
            file_path = os.path.join(HISTORICAL_DIR, f"{symbol.replace('-', '_')}.csv")
            df.to_csv(file_path)
            print(f"Data saved to {file_path}")
            success_count += 1
        else:
            print(f"No data collected for {symbol}")
            fail_count += 1
            failed_cryptos.append(symbol)
    
    # Afficher un résumé
    print("\nData collection summary:")
    print(f"Successfully collected data for {success_count} cryptocurrencies")
    print(f"Failed to collect data for {fail_count} cryptocurrencies")
    
    if failed_cryptos:
        print("\nFailed cryptocurrencies:")
        for crypto in failed_cryptos:
            print(f"- {crypto}")
    
    print("Data collection completed.")

# Exécuter la collecte de données
if __name__ == "__main__":
    # Importer la liste des cryptomonnaies
    from crypto_list import ALL_CRYPTOS
    
    # Collecter les données
    collect_data_for_all_cryptos(ALL_CRYPTOS)
