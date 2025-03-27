import yfinance as yf
import pandas as pd
from datetime import datetime

class YahooFinanceClient:
    """
    Client API réel pour Yahoo Finance utilisant la bibliothèque yfinance
    """
    def __init__(self):
        print("Client Yahoo Finance initialisé")
    
    def call_api(self, endpoint, query=None):
        """
        Appelle l'API Yahoo Finance via yfinance
        
        Args:
            endpoint (str): Point d'entrée de l'API (ignoré, gardé pour compatibilité)
            query (dict): Paramètres de la requête
                symbol (str): Symbole de la cryptomonnaie
                interval (str): Intervalle de temps
                range (str): Période
        
        Returns:
            dict: Données formatées de façon compatible avec le code existant
        """
        if endpoint != 'YahooFinance/get_stock_chart' or not query:
            print(f"Endpoint non supporté ou paramètres manquants: {endpoint}")
            return None
        
        try:
            symbol = query.get('symbol')
            interval = query.get('interval', '1d')
            period = query.get('range', '3mo')
            
            # Conversion des intervalles et périodes au format yfinance
            interval_map = {'1d': '1d', '1wk': '1wk', '1mo': '1mo'}
            period_map = {
                '1d': '1d', '5d': '5d', '1mo': '1mo',
                '3mo': '3mo', '6mo': '6mo', '1y': '1y',
                '2y': '2y', '5y': '5y', '10y': '10y',
                'ytd': 'ytd', 'max': 'max'
            }
            
            yf_interval = interval_map.get(interval, '1d')
            yf_period = period_map.get(period, '3mo')
            
            print(f"Récupération des données pour {symbol} avec interval={yf_interval}, period={yf_period}")
            
            # Télécharger les données depuis Yahoo Finance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=yf_period, interval=yf_interval)
            
            if hist.empty:
                print(f"Aucune donnée trouvée pour {symbol}")
                return None
            
            # Convertir au format attendu par le code existant
            timestamps = [int(dt.timestamp()) for dt in hist.index.to_pydatetime()]
            
            result = {
                'chart': {
                    'result': [{
                        'meta': {
                            'currency': 'USD',
                            'symbol': symbol,
                            'exchangeName': 'CRYPTO',
                            'instrumentType': 'CRYPTOCURRENCY',
                            'firstTradeDate': timestamps[0],
                            'regularMarketTime': timestamps[-1],
                            'gmtoffset': 0,
                            'timezone': 'UTC',
                            'exchangeTimezoneName': 'UTC',
                        },
                        'timestamp': timestamps,
                        'indicators': {
                            'quote': [{
                                'open': hist['Open'].tolist(),
                                'high': hist['High'].tolist(),
                                'low': hist['Low'].tolist(),
                                'close': hist['Close'].tolist(),
                                'volume': hist['Volume'].tolist()
                            }],
                            'adjclose': [{
                                'adjclose': hist['Close'].tolist()  # Yahoo Finance ne fournit pas de close ajusté pour les cryptos
                            }]
                        }
                    }],
                    'error': None
                }
            }
            
            print(f"Données récupérées avec succès pour {symbol}: {len(timestamps)} points")
            return result
            
        except Exception as e:
            print(f"Erreur lors de la récupération des données pour {symbol}: {e}")
            return None
