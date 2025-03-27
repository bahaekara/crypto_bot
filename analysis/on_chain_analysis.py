import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Définir les chemins des dossiers
DATA_DIR = r'C:\Users\bahik\Desktop\crypto_bot\data'
ONCHAIN_DIR = os.path.join(DATA_DIR, 'onchain')
os.makedirs(ONCHAIN_DIR, exist_ok=True)

class OnChainAnalyzer:
    """
    Classe pour l'analyse on-chain des cryptomonnaies
    """
    def __init__(self):
        self.data_dir = ONCHAIN_DIR
    
    def generate_simulated_data(self, symbol):
        """
        Génère des données on-chain simulées pour une cryptomonnaie
        Cette fonction évite les problèmes de dépassement d'entier
        
        Args:
            symbol (str): Symbole de la cryptomonnaie
            
        Returns:
            dict: Données on-chain simulées
        """
        print(f"Génération de données on-chain simulées pour {symbol}...")
        
        # Date actuelle
        end_date = datetime.now()
        
        # Date de début (90 jours avant)
        start_date = end_date - timedelta(days=90)
        
        # Générer des dates
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Nombre de jours
        days = len(dates)
        
        # Paramètres de base pour la simulation
        # Utiliser des valeurs plus petites pour éviter les dépassements d'entier
        base_active_addresses = np.random.randint(1000, 10000)
        base_transactions = np.random.randint(5000, 50000)
        base_volume = np.random.randint(10000, 100000)
        base_fees = np.random.randint(100, 1000)
        
        # Tendances (entre -0.5% et +1% par jour)
        address_trend = np.random.uniform(-0.005, 0.01)
        transaction_trend = np.random.uniform(-0.005, 0.01)
        volume_trend = np.random.uniform(-0.005, 0.01)
        fee_trend = np.random.uniform(-0.005, 0.01)
        
        # Volatilité (entre 1% et 5%)
        address_volatility = np.random.uniform(0.01, 0.05)
        transaction_volatility = np.random.uniform(0.01, 0.05)
        volume_volatility = np.random.uniform(0.01, 0.05)
        fee_volatility = np.random.uniform(0.01, 0.05)
        
        # Générer les données
        active_addresses = []
        transactions = []
        volumes = []
        fees = []
        
        current_addresses = base_active_addresses
        current_transactions = base_transactions
        current_volume = base_volume
        current_fees = base_fees
        
        for _ in range(days):
            # Adresses actives
            address_noise = np.random.normal(0, address_volatility * current_addresses)
            address_trend_component = address_trend * current_addresses
            current_addresses = max(100, current_addresses + address_trend_component + address_noise)
            active_addresses.append(int(current_addresses))
            
            # Transactions
            transaction_noise = np.random.normal(0, transaction_volatility * current_transactions)
            transaction_trend_component = transaction_trend * current_transactions
            current_transactions = max(100, current_transactions + transaction_trend_component + transaction_noise)
            transactions.append(int(current_transactions))
            
            # Volumes
            volume_noise = np.random.normal(0, volume_volatility * current_volume)
            volume_trend_component = volume_trend * current_volume
            current_volume = max(1000, current_volume + volume_trend_component + volume_noise)
            volumes.append(int(current_volume))
            
            # Frais
            fee_noise = np.random.normal(0, fee_volatility * current_fees)
            fee_trend_component = fee_trend * current_fees
            current_fees = max(10, current_fees + fee_trend_component + fee_noise)
            fees.append(int(current_fees))
        
        # Créer un DataFrame
        df = pd.DataFrame({
            'date': dates,
            'active_addresses': active_addresses,
            'transactions': transactions,
            'volume': volumes,
            'fees': fees,
            'symbol': symbol
        })
        
        # Définir la date comme index
        df.set_index('date', inplace=True)
        
        return df
    
    def analyze_onchain_data(self, symbol):
        """
        Analyse les données on-chain d'une cryptomonnaie
        
        Args:
            symbol (str): Symbole de la cryptomonnaie
            
        Returns:
            dict: Résultats de l'analyse on-chain
        """
        print(f"Analyse on-chain pour {symbol}...")
        
        try:
            # Générer des données simulées
            df = self.generate_simulated_data(symbol)
            
            # Enregistrer les données en CSV
            file_path = os.path.join(self.data_dir, f"{symbol.replace('-', '_')}_onchain.csv")
            df.to_csv(file_path)
            
            # Calculer des métriques
            latest_date = df.index.max()
            week_ago = latest_date - timedelta(days=7)
            month_ago = latest_date - timedelta(days=30)
            
            # Données récentes
            latest_data = df.loc[latest_date]
            
            # Données d'il y a une semaine
            week_ago_data = df.loc[df.index >= week_ago].iloc[0] if week_ago in df.index else df.iloc[0]
            
            # Données d'il y a un mois
            month_ago_data = df.loc[df.index >= month_ago].iloc[0] if month_ago in df.index else df.iloc[0]
            
            # Calculer les variations
            weekly_change_addresses = ((latest_data['active_addresses'] - week_ago_data['active_addresses']) / 
                                      week_ago_data['active_addresses']) * 100
            
            weekly_change_transactions = ((latest_data['transactions'] - week_ago_data['transactions']) / 
                                         week_ago_data['transactions']) * 100
            
            weekly_change_volume = ((latest_data['volume'] - week_ago_data['volume']) / 
                                   week_ago_data['volume']) * 100
            
            monthly_change_addresses = ((latest_data['active_addresses'] - month_ago_data['active_addresses']) / 
                                       month_ago_data['active_addresses']) * 100
            
            monthly_change_transactions = ((latest_data['transactions'] - month_ago_data['transactions']) / 
                                          month_ago_data['transactions']) * 100
            
            monthly_change_volume = ((latest_data['volume'] - month_ago_data['volume']) / 
                                    month_ago_data['volume']) * 100
            
            # Calculer des moyennes
            avg_daily_transactions = df['transactions'].mean()
            avg_daily_volume = df['volume'].mean()
            avg_daily_active_addresses = df['active_addresses'].mean()
            
            # Créer un résumé
            summary = {
                'symbol': symbol,
                'latest_date': latest_date.strftime('%Y-%m-%d'),
                'active_addresses': int(latest_data['active_addresses']),
                'transactions': int(latest_data['transactions']),
                'volume': int(latest_data['volume']),
                'fees': int(latest_data['fees']),
                'weekly_change': {
                    'active_addresses': round(weekly_change_addresses, 2),
                    'transactions': round(weekly_change_transactions, 2),
                    'volume': round(weekly_change_volume, 2)
                },
                'monthly_change': {
                    'active_addresses': round(monthly_change_addresses, 2),
                    'transactions': round(monthly_change_transactions, 2),
                    'volume': round(monthly_change_volume, 2)
                },
                'averages': {
                    'daily_transactions': round(avg_daily_transactions, 2),
                    'daily_volume': round(avg_daily_volume, 2),
                    'daily_active_addresses': round(avg_daily_active_addresses, 2)
                }
            }
            
            # Calculer un score on-chain (0-100)
            # Basé sur les variations et les moyennes
            address_score = min(100, max(0, 50 + weekly_change_addresses / 2))
            transaction_score = min(100, max(0, 50 + weekly_change_transactions / 2))
            volume_score = min(100, max(0, 50 + weekly_change_volume / 2))
            
            # Score global
            onchain_score = (address_score + transaction_score + volume_score) / 3
            
            # Ajouter le score au résumé
            summary['onchain_score'] = round(onchain_score, 2)
            
            # Déterminer un signal basé sur le score
            if onchain_score >= 70:
                summary['signal'] = 'FORT'
            elif onchain_score >= 50:
                summary['signal'] = 'POSITIF'
            elif onchain_score >= 30:
                summary['signal'] = 'NEUTRE'
            else:
                summary['signal'] = 'FAIBLE'
            
            # Enregistrer le résumé en JSON
            summary_path = os.path.join(self.data_dir, f"{symbol.replace('-', '_')}_onchain_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Analyse on-chain terminée pour {symbol}")
            return summary
            
        except Exception as e:
            print(f"Erreur lors de l'analyse on-chain pour {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'onchain_score': 50,
                'signal': 'NEUTRE'
            }

def analyze_all_cryptos(crypto_list, max_cryptos=5):
    """
    Analyse on-chain pour toutes les cryptomonnaies de la liste
    
    Args:
        crypto_list (list): Liste des dictionnaires contenant les symboles et noms des cryptomonnaies
        max_cryptos (int): Nombre maximum de cryptomonnaies à analyser
    """
    print(f"Analyse on-chain pour {min(len(crypto_list), max_cryptos)} cryptomonnaies...")
    
    # Créer l'analyseur
    analyzer = OnChainAnalyzer()
    
    # Limiter le nombre de cryptomonnaies à analyser
    cryptos_to_analyze = crypto_list[:max_cryptos]
    
    # Analyser chaque cryptomonnaie
    results = []
    for crypto in cryptos_to_analyze:
        symbol = crypto['symbol']
        
        # Analyser les données on-chain
        summary = analyzer.analyze_onchain_data(symbol)
        results.append(summary)
    
    # Créer un rapport global
    report = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'cryptos_analyzed': len(results),
        'results': results
    }
    
    # Enregistrer le rapport en JSON
    report_path = os.path.join(ONCHAIN_DIR, 'onchain_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Analyse on-chain terminée pour toutes les cryptomonnaies")
    return report

# Exécuter l'analyse on-chain
if __name__ == "__main__":
    # Importer la liste des cryptomonnaies
    sys.path.append(r'C:\Users\bahik\Desktop\crypto_bot\data')
    from crypto_list import ALL_CRYPTOS
    
    # Analyser les données on-chain
    analyze_all_cryptos(ALL_CRYPTOS, max_cryptos=5)
