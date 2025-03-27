# Simulation de données pour le bot d'analyse de cryptomonnaies

import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def generate_crypto_data(symbol, days=90, base_price=None, volatility=None, trend=None):
    """
    Génère des données historiques simulées pour une cryptomonnaie
    
    Args:
        symbol (str): Symbole de la cryptomonnaie
        days (int): Nombre de jours d'historique à générer
        base_price (float): Prix de base (si None, généré aléatoirement)
        volatility (float): Volatilité (si None, générée aléatoirement)
        trend (float): Tendance (si None, générée aléatoirement)
    
    Returns:
        pandas.DataFrame: DataFrame contenant les données simulées
    """
    # Définir les paramètres de simulation
    if base_price is None:
        base_price = np.random.uniform(100, 10000)
    
    if volatility is None:
        volatility = np.random.uniform(0.01, 0.05)
    
    if trend is None:
        trend = np.random.uniform(-0.005, 0.01)
    
    # Générer les dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(days)]
    dates.reverse()  # Ordre chronologique
    
    # Générer les prix
    prices = []
    current_price = base_price
    
    for _ in range(days):
        # Ajouter un bruit aléatoire
        noise = np.random.normal(0, volatility * current_price)
        # Ajouter une tendance
        current_price = current_price * (1 + trend) + noise
        # S'assurer que le prix reste positif
        current_price = max(current_price, 0.1)
        prices.append(current_price)
    
    # Créer les listes pour chaque type de prix
    opens = prices.copy()
    highs = [p * (1 + np.random.uniform(0, 0.05)) for p in prices]
    lows = [p * (1 - np.random.uniform(0, 0.05)) for p in prices]
    closes = [p * (1 + np.random.uniform(-0.03, 0.03)) for p in prices]
    volumes = [int(p * np.random.uniform(1000, 10000)) for p in prices]
    
    # Créer un DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'adj_close': closes,
        'symbol': symbol,
        'currency': 'USD'
    })
    
    # Définir la date comme index
    df.set_index('date', inplace=True)
    
    return df

def generate_all_crypto_data(crypto_list, output_dir):
    """
    Génère des données historiques simulées pour toutes les cryptomonnaies de la liste
    
    Args:
        crypto_list (list): Liste des dictionnaires contenant les symboles et noms des cryptomonnaies
        output_dir (str): Répertoire de sortie pour les fichiers CSV
    """
    # Créer le dossier de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Générer les données pour chaque cryptomonnaie
    for crypto in crypto_list:
        symbol = crypto['symbol']
        name = crypto['name']
        
        print(f"Generating data for {name} ({symbol})...")
        
        # Générer les données
        df = generate_crypto_data(symbol)
        
        # Enregistrer les données en CSV
        file_path = os.path.join(output_dir, f"{symbol.replace('-', '_')}.csv")
        df.to_csv(file_path)
        print(f"Data saved to {file_path}")
    
    print(f"Generated data for {len(crypto_list)} cryptocurrencies")

# Exemple d'utilisation
if __name__ == "__main__":
    # Liste des cryptomonnaies
    cryptos = [
        {'symbol': 'BTC-USD', 'name': 'Bitcoin'},
        {'symbol': 'ETH-USD', 'name': 'Ethereum'},
        {'symbol': 'SOL-USD', 'name': 'Solana'},
        # Ajouter d'autres cryptomonnaies selon les besoins
    ]
    
    # Générer les données
    generate_all_crypto_data(cryptos, 'data/historical')
