import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Ajouter le répertoire parent au chemin de recherche
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Définir les chemins des dossiers de manière relative
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')
HISTORICAL_DIR = os.path.join(DATA_DIR, 'historical')
RESULTS_DIR = os.path.join(ANALYSIS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(ANALYSIS_DIR, 'figures'), exist_ok=True)

def calculate_technical_indicators(df):
    """
    Calcule les indicateurs techniques pour un DataFrame de prix
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données de prix
    
    Returns:
        pandas.DataFrame: DataFrame avec les indicateurs techniques ajoutés
    """
    # Copier le DataFrame pour éviter de modifier l'original
    result = df.copy()
    
    # Moyennes mobiles
    result['MA_7'] = result['close'].rolling(window=7).mean()
    result['MA_20'] = result['close'].rolling(window=20).mean()
    result['MA_50'] = result['close'].rolling(window=50).mean()
    result['MA_100'] = result['close'].rolling(window=100).mean()
    
    # Écarts-types et bandes de Bollinger
    result['std_20'] = result['close'].rolling(window=20).std()
    result['BB_upper_20'] = result['MA_20'] + (result['std_20'] * 2)
    result['BB_middle_20'] = result['MA_20']
    result['BB_lower_20'] = result['MA_20'] - (result['std_20'] * 2)
    
    # RSI (Relative Strength Index)
    delta = result['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    result['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    result['EMA_12'] = result['close'].ewm(span=12, adjust=False).mean()
    result['EMA_26'] = result['close'].ewm(span=26, adjust=False).mean()
    result['MACD_line'] = result['EMA_12'] - result['EMA_26']
    result['MACD_signal'] = result['MACD_line'].ewm(span=9, adjust=False).mean()
    result['MACD_histogram'] = result['MACD_line'] - result['MACD_signal']
    
    # Momentum
    result['momentum_14'] = result['close'] / result['close'].shift(14) * 100
    
    # Volatilité
    result['volatility_14'] = result['close'].rolling(window=14).std() / result['close'].rolling(window=14).mean() * 100
    
    # Retourner le DataFrame avec les indicateurs
    return result

def analyze_crypto(symbol):
    """
    Analyse technique pour une cryptomonnaie
    
    Args:
        symbol (str): Symbole de la cryptomonnaie (ex: BTC-USD)
    
    Returns:
        pandas.DataFrame: DataFrame contenant les indicateurs techniques
    """
    print(f"Analyzing {symbol}...")
    
    # Charger les données historiques
    file_path = os.path.join(HISTORICAL_DIR, f"{symbol.replace('-', '_')}.csv")
    if not os.path.exists(file_path):
        print(f"No historical data found for {symbol}")
        return None
    
    # Lire les données
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Calculer les indicateurs techniques
    result = calculate_technical_indicators(df)
    
    # Enregistrer les résultats
    output_path = os.path.join(ANALYSIS_DIR, f"{symbol.replace('-', '_')}_indicators.csv")
    result.to_csv(output_path)
    print(f"Technical indicators saved to {output_path}")
    
    return result

def generate_technical_analysis_visualizations(symbol, df):
    """
    Génère des visualisations pour l'analyse technique
    
    Args:
        symbol (str): Symbole de la cryptomonnaie
        df (pandas.DataFrame): DataFrame contenant les indicateurs techniques
    """
    if df is None or df.empty:
        print(f"No data available for {symbol} visualizations")
        return
    
    # Créer un dossier pour les figures
    figures_dir = os.path.join(ANALYSIS_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Définir le style des graphiques
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Graphique des prix et moyennes mobiles
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Prix de clôture', color='blue')
    plt.plot(df.index, df['MA_7'], label='MA 7 jours', color='red')
    plt.plot(df.index, df['MA_20'], label='MA 20 jours', color='green')
    plt.plot(df.index, df['MA_50'], label='MA 50 jours', color='purple')
    plt.title(f'Prix et moyennes mobiles pour {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Prix (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{symbol.replace('-', '_')}_moving_averages.png"))
    plt.close()
    
    # 2. Bandes de Bollinger
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Prix de clôture', color='blue')
    plt.plot(df.index, df['BB_upper_20'], label='Bande supérieure', color='red')
    plt.plot(df.index, df['BB_middle_20'], label='Moyenne mobile 20 jours', color='green')
    plt.plot(df.index, df['BB_lower_20'], label='Bande inférieure', color='red')
    plt.fill_between(df.index, df['BB_upper_20'], df['BB_lower_20'], alpha=0.1, color='gray')
    plt.title(f'Bandes de Bollinger pour {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Prix (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{symbol.replace('-', '_')}_bollinger_bands.png"))
    plt.close()
    
    # 3. RSI
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['RSI_14'], color='purple')
    plt.axhline(y=70, color='red', linestyle='--')
    plt.axhline(y=30, color='green', linestyle='--')
    plt.fill_between(df.index, df['RSI_14'], 70, where=(df['RSI_14'] >= 70), color='red', alpha=0.3)
    plt.fill_between(df.index, df['RSI_14'], 30, where=(df['RSI_14'] <= 30), color='green', alpha=0.3)
    plt.title(f'RSI pour {symbol}')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{symbol.replace('-', '_')}_rsi.png"))
    plt.close()
    
    # 4. MACD
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['MACD_line'], label='MACD', color='blue')
    plt.plot(df.index, df['MACD_signal'], label='Signal', color='red')
    plt.bar(df.index, df['MACD_histogram'], label='Histogramme', color='gray', alpha=0.3)
    plt.title(f'MACD pour {symbol}')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{symbol.replace('-', '_')}_macd.png"))
    plt.close()
    
    print(f"Visualizations generated for {symbol}")

def analyze_all_cryptos():
    """
    Analyse technique pour toutes les cryptomonnaies disponibles
    """
    # Obtenir la liste des fichiers de données historiques
    historical_files = [f for f in os.listdir(HISTORICAL_DIR) if f.endswith('.csv')]
    
    if not historical_files:
        print("No historical data files found")
        return
    
    print(f"Found {len(historical_files)} historical data files")
    
    # Analyser chaque cryptomonnaie
    for file in historical_files:
        symbol = file.replace('.csv', '').replace('_', '-')
        df = analyze_crypto(symbol)
        
        if df is not None:
            generate_technical_analysis_visualizations(symbol, df)
    
    print("Technical analysis completed for all cryptocurrencies")

# Exécuter l'analyse technique
if __name__ == "__main__":
    # Vérifier si les données historiques existent
    if not os.path.exists(HISTORICAL_DIR):
        print(f"Historical data directory not found: {HISTORICAL_DIR}")
        print("Please run collect_data.py first")
        sys.exit(1)
    
    # Analyser toutes les cryptomonnaies
    analyze_all_cryptos()
