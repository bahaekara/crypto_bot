import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Définir les chemins des dossiers
DATA_DIR = '/home/ubuntu/crypto_bot/data'
ANALYSIS_DIR = '/home/ubuntu/crypto_bot/analysis'
CONFIDENCE_DIR = os.path.join(ANALYSIS_DIR, 'confidence')
os.makedirs(CONFIDENCE_DIR, exist_ok=True)
os.makedirs(os.path.join(CONFIDENCE_DIR, 'figures'), exist_ok=True)

class ConfidenceIndicatorSystem:
    """
    Système d'indicateurs de confiance pour les prédictions de cryptomonnaies
    """
    
    def __init__(self):
        """
        Initialise le système d'indicateurs de confiance
        """
        # Facteurs qui influencent la confiance dans les prédictions
        self.confidence_factors = {
            'volatility': {
                'weight': 0.20,
                'description': 'Volatilité récente de la cryptomonnaie'
            },
            'volume': {
                'weight': 0.15,
                'description': 'Volume de transactions récent'
            },
            'trend_strength': {
                'weight': 0.20,
                'description': 'Force de la tendance actuelle'
            },
            'market_correlation': {
                'weight': 0.10,
                'description': 'Corrélation avec le marché global'
            },
            'technical_consensus': {
                'weight': 0.20,
                'description': 'Consensus des indicateurs techniques'
            },
            'sentiment_stability': {
                'weight': 0.15,
                'description': 'Stabilité du sentiment sur les réseaux sociaux'
            }
        }
        
        # Seuils pour les niveaux de confiance
        self.confidence_thresholds = {
            'very_high': 0.80,
            'high': 0.65,
            'medium': 0.50,
            'low': 0.35,
            'very_low': 0.20
        }
    
    def load_historical_data(self, symbol):
        """
        Charge les données historiques pour une cryptomonnaie
        
        Args:
            symbol (str): Symbole de la cryptomonnaie
        
        Returns:
            pandas.DataFrame: DataFrame contenant les données historiques
        """
        file_path = os.path.join(DATA_DIR, 'historical', f"{symbol.replace('-', '_')}.csv")
        if not os.path.exists(file_path):
            print(f"No historical data found for {symbol}")
            return None
        
        df = pd.read_csv(file_path)
        
        # Adapter le code pour gérer différents formats de données
        date_column = None
        for col in df.columns:
            if col.lower() in ['date', 'time', 'datetime']:
                date_column = col
                break
        
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        else:
            # Si aucune colonne de date n'est trouvée, créer un index de date
            df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
        
        return df
    
    def load_technical_indicators(self, symbol):
        """
        Charge les indicateurs techniques pour une cryptomonnaie
        
        Args:
            symbol (str): Symbole de la cryptomonnaie
        
        Returns:
            pandas.DataFrame: DataFrame contenant les indicateurs techniques
        """
        file_path = os.path.join(ANALYSIS_DIR, f"{symbol.replace('-', '_')}_indicators.csv")
        if not os.path.exists(file_path):
            print(f"No technical indicators found for {symbol}")
            return None
        
        df = pd.read_csv(file_path)
        
        # Adapter le code pour gérer différents formats de données
        date_column = None
        for col in df.columns:
            if col.lower() in ['date', 'time', 'datetime']:
                date_column = col
                break
        
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        else:
            # Si aucune colonne de date n'est trouvée, créer un index de date
            df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
        
        return df
    
    def load_sentiment_data(self, symbol):
        """
        Charge les données de sentiment pour une cryptomonnaie
        
        Args:
            symbol (str): Symbole de la cryptomonnaie
        
        Returns:
            pandas.DataFrame: DataFrame contenant les données de sentiment
        """
        file_path = os.path.join(ANALYSIS_DIR, f"{symbol.replace('-', '_')}_sentiment.csv")
        if not os.path.exists(file_path):
            print(f"No sentiment data found for {symbol}, using simulated data")
            # Créer des données de sentiment simulées
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            sentiment_data = {
                'date': dates,
                'sentiment_score': np.random.normal(0.2, 0.3, size=30),
                'sentiment_volume': np.random.randint(100, 1000, size=30),
                'positive_ratio': np.random.uniform(0.4, 0.7, size=30),
                'negative_ratio': np.random.uniform(0.1, 0.4, size=30),
                'neutral_ratio': np.random.uniform(0.1, 0.3, size=30)
            }
            df = pd.DataFrame(sentiment_data)
            df.set_index('date', inplace=True)
            return df
        
        df = pd.read_csv(file_path)
        
        # Adapter le code pour gérer différents formats de données
        date_column = None
        for col in df.columns:
            if col.lower() in ['date', 'time', 'datetime']:
                date_column = col
                break
        
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        else:
            # Si aucune colonne de date n'est trouvée, créer un index de date
            df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
        
        return df
    
    def calculate_volatility_confidence(self, historical_df, window=14):
        """
        Calcule le score de confiance basé sur la volatilité
        
        Args:
            historical_df (pandas.DataFrame): DataFrame contenant les données historiques
            window (int): Fenêtre pour le calcul de la volatilité
        
        Returns:
            float: Score de confiance pour la volatilité (0-1)
        """
        if historical_df is None:
            return 0.5
        
        # Identifier la colonne de prix de clôture
        close_column = None
        for col in historical_df.columns:
            if col.lower() in ['close', 'price', 'closing_price', 'adj close', 'adj_close']:
                close_column = col
                break
        
        if close_column is None:
            return 0.5
        
        # Calculer les rendements journaliers
        returns = historical_df[close_column].pct_change(fill_method=None).dropna()
        
        if len(returns) < window:
            return 0.5
        
        # Calculer la volatilité récente (écart-type des rendements)
        recent_volatility = returns[-window:].std()
        
        # Calculer la volatilité historique (sur toute la période disponible)
        historical_volatility = returns.std()
        
        # Normaliser la volatilité récente par rapport à la volatilité historique
        if historical_volatility > 0:
            relative_volatility = recent_volatility / historical_volatility
        else:
            relative_volatility = 1.0
        
        # Convertir en score de confiance (volatilité plus faible = confiance plus élevée)
        # Utiliser une fonction sigmoïde pour mapper la volatilité relative à un score entre 0 et 1
        confidence_score = 1 / (1 + np.exp(5 * (relative_volatility - 1)))
        
        return confidence_score
    
    def calculate_volume_confidence(self, historical_df, window=14):
        """
        Calcule le score de confiance basé sur le volume de transactions
        
        Args:
            historical_df (pandas.DataFrame): DataFrame contenant les données historiques
            window (int): Fenêtre pour le calcul du volume moyen
        
        Returns:
            float: Score de confiance pour le volume (0-1)
        """
        if historical_df is None:
            return 0.5
        
        # Vérifier si la colonne de volume existe
        volume_column = None
        for col in historical_df.columns:
            if col.lower() in ['volume', 'vol', 'transaction_volume']:
                volume_column = col
                break
        
        if volume_column is None:
            return 0.5
        
        # Extraire les volumes
        volumes = historical_df[volume_column]
        
        if len(volumes) < window:
            return 0.5
        
        # Calculer le volume moyen récent
        recent_volume = volumes[-window:].mean()
        
        # Calculer le volume moyen historique
        historical_volume = volumes.mean()
        
        # Normaliser le volume récent par rapport au volume historique
        if historical_volume > 0:
            relative_volume = recent_volume / historical_volume
        else:
            relative_volume = 1.0
        
        # Convertir en score de confiance (volume plus élevé = confiance plus élevée)
        # Utiliser une fonction sigmoïde pour mapper le volume relatif à un score entre 0 et 1
        confidence_score = 1 / (1 + np.exp(-5 * (relative_volume - 1)))
        
        return min(1.0, max(0.0, confidence_score))
    
    def calculate_trend_strength_confidence(self, indicators_df, window=14):
        """
        Calcule le score de confiance basé sur la force de la tendance
        
        Args:
            indicators_df (pandas.DataFrame): DataFrame contenant les indicateurs techniques
            window (int): Fenêtre pour le calcul de la force de la tendance
        
        Returns:
            float: Score de confiance pour la force de la tendance (0-1)
        """
        if indicators_df is None:
            return 0.5
        
        # Vérifier si les colonnes nécessaires existent
        required_columns = ['MA_20', 'MA_50', 'RSI_14']
        available_columns = [col for col in required_columns if col in indicators_df.columns]
        
        if not available_columns:
            return 0.5
        
        # Calculer la force de la tendance en fonction des moyennes mobiles et du RSI
        trend_strength = 0.5
        
        if 'MA_20' in indicators_df.columns and 'MA_50' in indicators_df.columns:
            # Calculer le ratio MA_20 / MA_50
            recent_data = indicators_df.iloc[-window:]
            ma_ratio = recent_data['MA_20'] / recent_data['MA_50']
            
            # Calculer la tendance du ratio (pente)
            ma_ratio_trend = np.polyfit(range(len(ma_ratio)), ma_ratio, 1)[0]
            
            # Normaliser la tendance
            normalized_trend = 1 / (1 + np.exp(-50 * ma_ratio_trend))
            
            trend_strength = normalized_trend
        
        if 'RSI_14' in indicators_df.columns:
            # Utiliser le RSI pour ajuster la force de la tendance
            recent_rsi = indicators_df['RSI_14'].iloc[-1]
            
            # RSI proche de 50 indique une tendance faible
            # RSI proche de 0 ou 100 indique une tendance forte (mais potentiellement surachetée/survendue)
            rsi_strength = 2 * abs(recent_rsi - 50) / 50
            
            # Ajuster la force de la tendance en fonction du RSI
            trend_strength = 0.7 * trend_strength + 0.3 * rsi_strength
        
        return min(1.0, max(0.0, trend_strength))
    
    def calculate_market_correlation_confidence(self, symbol, historical_df, window=30):
        """
        Calcule le score de confiance basé sur la corrélation avec le marché global
        
        Args:
            symbol (str): Symbole de la cryptomonnaie
            historical_df (pandas.DataFrame): DataFrame contenant les données historiques
            window (int): Fenêtre pour le calcul de la corrélation
        
        Returns:
            float: Score de confiance pour la corrélation avec le marché (0-1)
        """
        if historical_df is None:
            return 0.5
        
        # Identifier la colonne de prix de clôture
        close_column = None
        for col in historical_df.columns:
            if col.lower() in ['close', 'price', 'closing_price', 'adj close', 'adj_close']:
                close_column = col
                break
        
        if close_column is None:
            return 0.5
        
        # Charger les données de BTC comme référence du marché (si le symbole n'est pas BTC)
        if 'BTC' not in symbol:
            market_file_path = os.path.join(DATA_DIR, 'historical', "BTC_USD.csv")
            if not os.path.exists(market_file_path):
                return 0.5
            
            market_df = pd.read_csv(market_file_path)
            
            # Adapter le code pour gérer différents formats de données
            date_column = None
            for col in market_df.columns:
                if col.lower() in ['date', 'time', 'datetime']:
                    date_column = col
                    break
            
            if date_column:
                market_df[date_column] = pd.to_datetime(market_df[date_column])
                market_df.set_index(date_column, inplace=True)
            else:
                # Si aucune colonne de date n'est trouvée, créer un index de date
                market_df.index = pd.date_range(end=datetime.now(), periods=len(market_df), freq='D')
            
            # Identifier la colonne de prix de clôture pour le marché
            market_close_column = None
            for col in market_df.columns:
                if col.lower() in ['close', 'price', 'closing_price', 'adj close', 'adj_close']:
                    market_close_column = col
                    break
            
            if market_close_column is None:
                return 0.5
            
            # Calculer les rendements
            crypto_returns = historical_df[close_column].pct_change(fill_method=None).dropna()
            market_returns = market_df[market_close_column].pct_change(fill_method=None).dropna()
            
            # Aligner les indices
            common_index = crypto_returns.index.intersection(market_returns.index)
            if len(common_index) < window:
                return 0.5
            
            crypto_returns = crypto_returns.loc[common_index]
            market_returns = market_returns.loc[common_index]
            
            # Calculer la corrélation récente
            if len(crypto_returns) >= window:
                correlation = crypto_returns[-window:].corr(market_returns[-window:])
            else:
                correlation = crypto_returns.corr(market_returns)
            
            # Convertir la corrélation en score de confiance
            # Une corrélation plus élevée avec le marché indique généralement une prédiction plus fiable
            confidence_score = 0.5 + 0.5 * abs(correlation)
        else:
            # Pour BTC, qui est la référence du marché, attribuer un score élevé
            confidence_score = 0.8
        
        return min(1.0, max(0.0, confidence_score))
    
    def calculate_technical_consensus_confidence(self, indicators_df):
        """
        Calcule le score de confiance basé sur le consensus des indicateurs techniques
        
        Args:
            indicators_df (pandas.DataFrame): DataFrame contenant les indicateurs techniques
        
        Returns:
            float: Score de confiance pour le consensus technique (0-1)
        """
        if indicators_df is None or len(indicators_df) == 0:
            return 0.5
        
        # Obtenir les dernières valeurs des indicateurs
        latest_indicators = indicators_df.iloc[-1]
        
        # Initialiser les signaux
        signals = []
        
        # Vérifier les moyennes mobiles
        if 'MA_7' in latest_indicators and 'MA_20' in latest_indicators:
            if latest_indicators['MA_7'] > latest_indicators['MA_20']:
                signals.append(1)  # Signal haussier
            else:
                signals.append(-1)  # Signal baissier
        
        if 'MA_20' in latest_indicators and 'MA_50' in latest_indicators:
            if latest_indicators['MA_20'] > latest_indicators['MA_50']:
                signals.append(1)
            else:
                signals.append(-1)
        
        # Vérifier le RSI
        if 'RSI_14' in latest_indicators:
            rsi = latest_indicators['RSI_14']
            if rsi > 70:
                signals.append(-1)  # Suracheté
            elif rsi < 30:
                signals.append(1)  # Survendu
            elif rsi > 50:
                signals.append(0.5)  # Légèrement haussier
            else:
                signals.append(-0.5)  # Légèrement baissier
        
        # Vérifier le MACD
        if 'MACD_line' in latest_indicators and 'MACD_signal' in latest_indicators:
            if latest_indicators['MACD_line'] > latest_indicators['MACD_signal']:
                signals.append(1)
            else:
                signals.append(-1)
        
        # Vérifier les bandes de Bollinger
        if 'close' in latest_indicators and 'BB_upper_20' in latest_indicators and 'BB_lower_20' in latest_indicators:
            close = latest_indicators['close']
            upper = latest_indicators['BB_upper_20']
            lower = latest_indicators['BB_lower_20']
            
            if close > upper:
                signals.append(-1)  # Suracheté
            elif close < lower:
                signals.append(1)  # Survendu
            else:
                # Dans la bande, neutre
                signals.append(0)
        
        # Si aucun signal n'est disponible, retourner une valeur neutre
        if not signals:
            return 0.5
        
        # Calculer le consensus (moyenne des signaux)
        consensus = np.mean(signals)
        
        # Convertir le consensus en score de confiance
        # Un consensus fort (proche de 1 ou -1) indique une confiance élevée
        confidence_score = 0.5 + 0.5 * abs(consensus)
        
        return min(1.0, max(0.0, confidence_score))
    
    def calculate_sentiment_stability_confidence(self, sentiment_df, window=7):
        """
        Calcule le score de confiance basé sur la stabilité du sentiment
        
        Args:
            sentiment_df (pandas.DataFrame): DataFrame contenant les données de sentiment
            window (int): Fenêtre pour le calcul de la stabilité du sentiment
        
        Returns:
            float: Score de confiance pour la stabilité du sentiment (0-1)
        """
        if sentiment_df is None or len(sentiment_df) < window:
            return 0.5
        
        # Vérifier si la colonne de sentiment existe
        sentiment_column = None
        for col in sentiment_df.columns:
            if col.lower() in ['sentiment_score', 'sentiment', 'score']:
                sentiment_column = col
                break
        
        if sentiment_column is None:
            return 0.5
        
        # Extraire les scores de sentiment récents
        recent_sentiment = sentiment_df[sentiment_column][-window:]
        
        # Calculer la stabilité du sentiment (inverse de la volatilité)
        sentiment_stability = 1 / (1 + recent_sentiment.std())
        
        # Normaliser le score de confiance
        confidence_score = min(1.0, sentiment_stability)
        
        return confidence_score
    
    def calculate_overall_confidence(self, symbol, prediction_horizon=7):
        """
        Calcule le score de confiance global pour une cryptomonnaie
        
        Args:
            symbol (str): Symbole de la cryptomonnaie
            prediction_horizon (int): Horizon de prédiction en jours
        
        Returns:
            dict: Scores de confiance pour chaque facteur et score global
        """
        print(f"Calcul du score de confiance pour {symbol}...")
        
        # Charger les données
        historical_df = self.load_historical_data(symbol)
        indicators_df = self.load_technical_indicators(symbol)
        sentiment_df = self.load_sentiment_data(symbol)
        
        # Calculer les scores de confiance pour chaque facteur
        confidence_scores = {}
        
        # 1. Volatilité
        confidence_scores['volatility'] = self.calculate_volatility_confidence(historical_df)
        print(f"  Score de confiance pour la volatilité: {confidence_scores['volatility']:.4f}")
        
        # 2. Volume
        confidence_scores['volume'] = self.calculate_volume_confidence(historical_df)
        print(f"  Score de confiance pour le volume: {confidence_scores['volume']:.4f}")
        
        # 3. Force de la tendance
        confidence_scores['trend_strength'] = self.calculate_trend_strength_confidence(indicators_df)
        print(f"  Score de confiance pour la force de la tendance: {confidence_scores['trend_strength']:.4f}")
        
        # 4. Corrélation avec le marché
        confidence_scores['market_correlation'] = self.calculate_market_correlation_confidence(symbol, historical_df)
        print(f"  Score de confiance pour la corrélation avec le marché: {confidence_scores['market_correlation']:.4f}")
        
        # 5. Consensus technique
        confidence_scores['technical_consensus'] = self.calculate_technical_consensus_confidence(indicators_df)
        print(f"  Score de confiance pour le consensus technique: {confidence_scores['technical_consensus']:.4f}")
        
        # 6. Stabilité du sentiment
        confidence_scores['sentiment_stability'] = self.calculate_sentiment_stability_confidence(sentiment_df)
        print(f"  Score de confiance pour la stabilité du sentiment: {confidence_scores['sentiment_stability']:.4f}")
        
        # Calculer le score de confiance global (moyenne pondérée)
        overall_confidence = 0.0
        for factor, score in confidence_scores.items():
            weight = self.confidence_factors[factor]['weight']
            overall_confidence += weight * score
        
        print(f"  Score de confiance global: {overall_confidence:.4f}")
        
        # Déterminer le niveau de confiance
        if overall_confidence >= self.confidence_thresholds['very_high']:
            confidence_level = "Très élevé"
        elif overall_confidence >= self.confidence_thresholds['high']:
            confidence_level = "Élevé"
        elif overall_confidence >= self.confidence_thresholds['medium']:
            confidence_level = "Moyen"
        elif overall_confidence >= self.confidence_thresholds['low']:
            confidence_level = "Faible"
        else:
            confidence_level = "Très faible"
        
        print(f"  Niveau de confiance: {confidence_level}")
        
        # Créer le résultat
        result = {
            'symbol': symbol,
            'prediction_horizon': prediction_horizon,
            'confidence_scores': confidence_scores,
            'overall_confidence': overall_confidence,
            'confidence_level': confidence_level
        }
        
        # Visualiser les résultats
        self.visualize_confidence_scores(symbol, confidence_scores, overall_confidence)
        
        # Générer un rapport
        self.generate_confidence_report(result)
        
        return result
    
    def visualize_confidence_scores(self, symbol, confidence_scores, overall_confidence):
        """
        Visualise les scores de confiance
        
        Args:
            symbol (str): Symbole de la cryptomonnaie
            confidence_scores (dict): Scores de confiance pour chaque facteur
            overall_confidence (float): Score de confiance global
        """
        # Définir le style des graphiques
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Graphique radar des scores de confiance
        categories = list(confidence_scores.keys())
        values = [confidence_scores[category] for category in categories]
        
        # Ajouter le premier élément à la fin pour fermer le polygone
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        # Convertir en radians
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        
        # S'assurer que les angles et les valeurs ont la même longueur
        if len(angles) != len(categories) - 1:
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        
        angles += angles[:1]  # Fermer le graphique
        
        # Vérifier que les dimensions correspondent
        if len(angles) != len(values):
            print(f"Erreur: dimensions incompatibles pour le graphique radar - angles: {len(angles)}, values: {len(values)}")
            # Ajuster les dimensions si nécessaire
            min_len = min(len(angles), len(values))
            angles = angles[:min_len]
            values = values[:min_len]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=symbol)
        ax.fill(angles, values, alpha=0.25)
        
        # Ajouter les étiquettes
        labels = [self.confidence_factors[cat]['description'] if cat in self.confidence_factors else cat for cat in categories[:-1]]
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        
        # Ajouter les niveaux
        ax.set_rlabel_position(0)
        ax.set_rticks([0.25, 0.5, 0.75, 1])
        ax.set_rmax(1)
        
        plt.title(f'{symbol} - Scores de Confiance', size=15, y=1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIDENCE_DIR, 'figures', f"{symbol.replace('-', '_')}_confidence_radar.png"))
        plt.close()
        
        # 2. Graphique à barres des scores de confiance
        plt.figure(figsize=(12, 8))
        
        # Définir les couleurs en fonction du score
        colors = []
        for score in values[:-1]:
            if score >= 0.8:
                colors.append('darkgreen')
            elif score >= 0.65:
                colors.append('green')
            elif score >= 0.5:
                colors.append('orange')
            elif score >= 0.35:
                colors.append('red')
            else:
                colors.append('darkred')
        
        # Créer le graphique à barres
        bars = plt.bar(labels, values[:-1], color=colors)
        
        plt.axhline(y=0.8, color='darkgreen', linestyle='--', alpha=0.5)
        plt.axhline(y=0.65, color='green', linestyle='--', alpha=0.5)
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(y=0.35, color='red', linestyle='--', alpha=0.5)
        
        plt.xlabel('Facteurs de Confiance')
        plt.ylabel('Score de Confiance')
        plt.title(f'{symbol} - Scores de Confiance par Facteur')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIDENCE_DIR, 'figures', f"{symbol.replace('-', '_')}_confidence_bars.png"))
        plt.close()
        
        # 3. Graphique de la jauge de confiance globale
        plt.figure(figsize=(8, 8))
        
        # Créer un graphique de jauge
        gauge_angles = np.linspace(0, 180, 100)
        gauge_values = np.linspace(0, 100, 100)
        
        # Définir les couleurs pour différentes plages de valeurs
        colors = []
        for val in gauge_values:
            if val < 20:
                colors.append('darkred')
            elif val < 35:
                colors.append('red')
            elif val < 50:
                colors.append('orange')
            elif val < 65:
                colors.append('yellowgreen')
            elif val < 80:
                colors.append('green')
            else:
                colors.append('darkgreen')
        
        # Convertir en radians
        gauge_angles = np.deg2rad(gauge_angles)
        
        # Créer le graphique
        ax = plt.subplot(111, polar=True)
        
        # Masquer les axes et les étiquettes
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
        # Définir les limites
        ax.set_ylim(0, 1)
        
        # Tracer les barres de la jauge
        bars = ax.bar(gauge_angles, [1] * len(gauge_angles), width=np.deg2rad(180/len(gauge_angles)), bottom=0, color=colors, alpha=0.8)
        
        # Ajouter une flèche pour indiquer la valeur
        confidence_score = overall_confidence * 100
        score_angle = np.deg2rad(confidence_score * 180 / 100)
        ax.arrow(0, 0, score_angle, 0.8, alpha=0.8, width=0.02, head_width=0.05, head_length=0.1, color='black')
        
        # Ajouter le texte de la valeur
        plt.text(0, 0.5, f"{confidence_score:.1f}", ha='center', va='center', fontsize=30, fontweight='bold')
        plt.text(0, 0.3, f"Score de Confiance", ha='center', va='center', fontsize=14)
        
        # Déterminer le niveau de confiance
        if overall_confidence >= self.confidence_thresholds['very_high']:
            confidence_level = "Très élevé"
        elif overall_confidence >= self.confidence_thresholds['high']:
            confidence_level = "Élevé"
        elif overall_confidence >= self.confidence_thresholds['medium']:
            confidence_level = "Moyen"
        elif overall_confidence >= self.confidence_thresholds['low']:
            confidence_level = "Faible"
        else:
            confidence_level = "Très faible"
        
        plt.text(0, 0.2, f"Niveau: {confidence_level}", ha='center', va='center', fontsize=14)
        
        # Ajouter des étiquettes pour les plages
        plt.text(np.deg2rad(170), 0.85, "Excellent", ha='right', va='center', fontsize=12, color='darkgreen')
        plt.text(np.deg2rad(130), 0.85, "Bon", ha='right', va='center', fontsize=12, color='green')
        plt.text(np.deg2rad(90), 0.85, "Moyen", ha='center', va='center', fontsize=12, color='orange')
        plt.text(np.deg2rad(50), 0.85, "Faible", ha='left', va='center', fontsize=12, color='red')
        plt.text(np.deg2rad(10), 0.85, "Critique", ha='left', va='center', fontsize=12, color='darkred')
        
        plt.title(f'{symbol} - Score de Confiance Global', size=15, y=1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIDENCE_DIR, 'figures', f"{symbol.replace('-', '_')}_confidence_gauge.png"))
        plt.close()
    
    def generate_confidence_report(self, result):
        """
        Génère un rapport de confiance
        
        Args:
            result (dict): Résultats de l'analyse de confiance
        """
        symbol = result['symbol']
        prediction_horizon = result['prediction_horizon']
        confidence_scores = result['confidence_scores']
        overall_confidence = result['overall_confidence']
        confidence_level = result['confidence_level']
        
        report = f"# Rapport de Confiance pour {symbol}\n\n"
        report += f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"Horizon de prédiction: **{prediction_horizon} jours**\n\n"
        
        report += "## Résumé\n\n"
        report += f"Score de confiance global: **{overall_confidence*100:.1f}/100** - Niveau: **{confidence_level}**\n\n"
        
        report += "## Scores de Confiance par Facteur\n\n"
        report += "| Facteur | Score | Description |\n"
        report += "|---------|-------|-------------|\n"
        
        for factor, score in confidence_scores.items():
            description = self.confidence_factors[factor]['description'] if factor in self.confidence_factors else ""
            report += f"| {factor.replace('_', ' ').title()} | {score*100:.1f}/100 | {description} |\n"
        
        report += "\n## Interprétation\n\n"
        
        # Générer une interprétation basée sur le score de confiance global
        if overall_confidence >= self.confidence_thresholds['very_high']:
            report += f"Le niveau de confiance pour les prédictions de {symbol} sur un horizon de {prediction_horizon} jours est **très élevé**. Les facteurs clés contribuant à cette confiance élevée sont:\n\n"
        elif overall_confidence >= self.confidence_thresholds['high']:
            report += f"Le niveau de confiance pour les prédictions de {symbol} sur un horizon de {prediction_horizon} jours est **élevé**. Les facteurs clés contribuant à cette confiance sont:\n\n"
        elif overall_confidence >= self.confidence_thresholds['medium']:
            report += f"Le niveau de confiance pour les prédictions de {symbol} sur un horizon de {prediction_horizon} jours est **moyen**. Les facteurs clés influençant ce niveau de confiance sont:\n\n"
        elif overall_confidence >= self.confidence_thresholds['low']:
            report += f"Le niveau de confiance pour les prédictions de {symbol} sur un horizon de {prediction_horizon} jours est **faible**. Les facteurs clés contribuant à cette faible confiance sont:\n\n"
        else:
            report += f"Le niveau de confiance pour les prédictions de {symbol} sur un horizon de {prediction_horizon} jours est **très faible**. Les facteurs clés contribuant à cette très faible confiance sont:\n\n"
        
        # Identifier les facteurs les plus forts et les plus faibles
        sorted_factors = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Facteurs les plus forts
        report += "**Points forts:**\n\n"
        for factor, score in sorted_factors[:3]:
            if score >= 0.5:
                description = self.confidence_factors[factor]['description'] if factor in self.confidence_factors else factor
                report += f"- **{description}** ({score*100:.1f}/100): "
                
                if factor == 'volatility':
                    if score >= 0.8:
                        report += "La volatilité récente est très faible par rapport à la volatilité historique, ce qui suggère un environnement de marché stable favorable aux prédictions précises.\n"
                    elif score >= 0.65:
                        report += "La volatilité récente est relativement faible, ce qui suggère un environnement de marché assez stable pour les prédictions.\n"
                    else:
                        report += "La volatilité est modérée, ce qui permet des prédictions raisonnablement fiables.\n"
                
                elif factor == 'volume':
                    if score >= 0.8:
                        report += "Le volume de transactions récent est très élevé, ce qui indique une forte liquidité et une meilleure fiabilité des signaux de prix.\n"
                    elif score >= 0.65:
                        report += "Le volume de transactions récent est bon, ce qui indique une liquidité suffisante pour des signaux de prix fiables.\n"
                    else:
                        report += "Le volume de transactions est acceptable, permettant une analyse raisonnable des mouvements de prix.\n"
                
                elif factor == 'trend_strength':
                    if score >= 0.8:
                        report += "La tendance actuelle est très forte et bien établie, ce qui facilite les prédictions directionnelles.\n"
                    elif score >= 0.65:
                        report += "La tendance actuelle est claire et relativement forte, ce qui aide à la fiabilité des prédictions.\n"
                    else:
                        report += "Une tendance modérée est présente, offrant une base pour les prédictions.\n"
                
                elif factor == 'market_correlation':
                    if score >= 0.8:
                        report += "La corrélation avec le marché global est très forte, ce qui permet de s'appuyer sur les tendances générales du marché pour les prédictions.\n"
                    elif score >= 0.65:
                        report += "La corrélation avec le marché global est bonne, offrant un contexte fiable pour les prédictions.\n"
                    else:
                        report += "Une corrélation modérée avec le marché global est observée, fournissant un certain contexte pour les prédictions.\n"
                
                elif factor == 'technical_consensus':
                    if score >= 0.8:
                        report += "Les indicateurs techniques montrent un consensus très fort, ce qui renforce la confiance dans la direction prédite.\n"
                    elif score >= 0.65:
                        report += "Les indicateurs techniques montrent un bon consensus, soutenant la direction prédite.\n"
                    else:
                        report += "Un consensus modéré est observé parmi les indicateurs techniques.\n"
                
                elif factor == 'sentiment_stability':
                    if score >= 0.8:
                        report += "Le sentiment sur les réseaux sociaux est très stable, ce qui suggère une opinion de marché cohérente et fiable.\n"
                    elif score >= 0.65:
                        report += "Le sentiment sur les réseaux sociaux est relativement stable, offrant une base fiable pour l'analyse.\n"
                    else:
                        report += "Le sentiment présente une stabilité modérée, permettant une certaine analyse des opinions du marché.\n"
        
        # Facteurs les plus faibles
        report += "\n**Points faibles:**\n\n"
        for factor, score in sorted_factors[-3:]:
            if score < 0.65:
                description = self.confidence_factors[factor]['description'] if factor in self.confidence_factors else factor
                report += f"- **{description}** ({score*100:.1f}/100): "
                
                if factor == 'volatility':
                    if score < 0.35:
                        report += "La volatilité récente est extrêmement élevée, ce qui rend les prédictions très difficiles et peu fiables.\n"
                    elif score < 0.5:
                        report += "La volatilité récente est élevée, ce qui réduit la fiabilité des prédictions.\n"
                    else:
                        report += "La volatilité est légèrement élevée, ce qui peut affecter la précision des prédictions.\n"
                
                elif factor == 'volume':
                    if score < 0.35:
                        report += "Le volume de transactions récent est très faible, ce qui indique une liquidité insuffisante et des signaux de prix potentiellement trompeurs.\n"
                    elif score < 0.5:
                        report += "Le volume de transactions récent est faible, ce qui peut limiter la fiabilité des signaux de prix.\n"
                    else:
                        report += "Le volume de transactions est modéré, ce qui peut parfois limiter la clarté des signaux de prix.\n"
                
                elif factor == 'trend_strength':
                    if score < 0.35:
                        report += "Aucune tendance claire n'est identifiable, ce qui rend les prédictions directionnelles très difficiles.\n"
                    elif score < 0.5:
                        report += "La tendance actuelle est faible ou incertaine, ce qui réduit la fiabilité des prédictions.\n"
                    else:
                        report += "La tendance n'est pas très prononcée, ce qui peut limiter la précision des prédictions.\n"
                
                elif factor == 'market_correlation':
                    if score < 0.35:
                        report += "La corrélation avec le marché global est très faible, ce qui rend difficile l'utilisation des tendances générales du marché pour les prédictions.\n"
                    elif score < 0.5:
                        report += "La corrélation avec le marché global est faible, limitant la fiabilité du contexte de marché pour les prédictions.\n"
                    else:
                        report += "La corrélation avec le marché global est modérée, ce qui peut parfois limiter la contextualisation des prédictions.\n"
                
                elif factor == 'technical_consensus':
                    if score < 0.35:
                        report += "Les indicateurs techniques montrent des signaux contradictoires, ce qui réduit considérablement la confiance dans la direction prédite.\n"
                    elif score < 0.5:
                        report += "Les indicateurs techniques montrent peu de consensus, ce qui affaiblit la confiance dans la direction prédite.\n"
                    else:
                        report += "Le consensus parmi les indicateurs techniques n'est pas très fort, ce qui peut limiter la fiabilité des signaux.\n"
                
                elif factor == 'sentiment_stability':
                    if score < 0.35:
                        report += "Le sentiment sur les réseaux sociaux est très volatil, ce qui rend difficile l'interprétation des opinions du marché.\n"
                    elif score < 0.5:
                        report += "Le sentiment sur les réseaux sociaux est instable, ce qui réduit la fiabilité de l'analyse des opinions du marché.\n"
                    else:
                        report += "Le sentiment présente une certaine instabilité, ce qui peut parfois compliquer l'analyse des opinions du marché.\n"
        
        report += "\n## Recommandation\n\n"
        
        # Générer une recommandation basée sur le score de confiance global
        if overall_confidence >= self.confidence_thresholds['very_high']:
            report += f"**CONFIANCE TRÈS ÉLEVÉE** - Les prédictions pour {symbol} sur un horizon de {prediction_horizon} jours peuvent être suivies avec un très haut niveau de confiance. Les conditions de marché actuelles sont très favorables à des prédictions précises.\n\n"
            report += f"Recommandation de position: Positions de taille standard à grande, avec des stop-loss standards.\n\n"
        elif overall_confidence >= self.confidence_thresholds['high']:
            report += f"**CONFIANCE ÉLEVÉE** - Les prédictions pour {symbol} sur un horizon de {prediction_horizon} jours peuvent être suivies avec un bon niveau de confiance. Les conditions de marché actuelles sont favorables à des prédictions relativement précises.\n\n"
            report += f"Recommandation de position: Positions de taille standard, avec des stop-loss standards.\n\n"
        elif overall_confidence >= self.confidence_thresholds['medium']:
            report += f"**CONFIANCE MOYENNE** - Les prédictions pour {symbol} sur un horizon de {prediction_horizon} jours devraient être utilisées avec une prudence modérée. Les conditions de marché actuelles permettent des prédictions raisonnablement fiables, mais des écarts sont possibles.\n\n"
            report += f"Recommandation de position: Positions de taille réduite (50-75% de la normale), avec des stop-loss légèrement plus serrés.\n\n"
        elif overall_confidence >= self.confidence_thresholds['low']:
            report += f"**CONFIANCE FAIBLE** - Les prédictions pour {symbol} sur un horizon de {prediction_horizon} jours ne sont pas très fiables dans les conditions actuelles. Il est recommandé de les utiliser avec beaucoup de prudence et de les combiner avec d'autres analyses.\n\n"
            report += f"Recommandation de position: Positions de petite taille (25-50% de la normale), avec des stop-loss serrés, ou attendre de meilleures conditions de marché.\n\n"
        else:
            report += f"**CONFIANCE TRÈS FAIBLE** - Les prédictions pour {symbol} sur un horizon de {prediction_horizon} jours sont très peu fiables dans les conditions actuelles. Il est recommandé de ne pas les utiliser comme base principale pour les décisions d'investissement.\n\n"
            report += f"Recommandation de position: Éviter les positions ou limiter à des positions très petites (max 25% de la normale) avec des stop-loss très serrés. Attendre de meilleures conditions de marché.\n\n"
        
        report += "## Note Méthodologique\n\n"
        report += "Ce rapport de confiance évalue la fiabilité des prédictions en fonction de divers facteurs de marché. Un score de confiance élevé n'est pas une garantie de prédictions correctes, mais indique des conditions de marché plus favorables à des prédictions précises.\n\n"
        report += "Les facteurs de confiance analysés comprennent:\n\n"
        
        for factor, info in self.confidence_factors.items():
            report += f"- **{info['description']}** (poids: {info['weight']*100:.0f}%): Évalue {factor.replace('_', ' ')}\n"
        
        report += "\n*Ce rapport ne constitue pas un conseil en investissement. Investissez de manière responsable et à vos propres risques.*\n"
        
        # Enregistrer le rapport
        report_path = os.path.join(CONFIDENCE_DIR, f"{symbol.replace('-', '_')}_confidence_report.md")
        with open(report_path, 'w') as f:
            f.write(report)

def analyze_confidence_for_all_cryptos():
    """
    Analyse les scores de confiance pour toutes les cryptomonnaies disponibles
    """
    # Obtenir la liste des cryptomonnaies à partir des fichiers CSV dans le dossier historical
    historical_dir = os.path.join(DATA_DIR, 'historical')
    csv_files = [f for f in os.listdir(historical_dir) if f.endswith('.csv')]
    symbols = [f.replace('_', '-').replace('.csv', '') for f in csv_files]
    
    # Limiter à 5 cryptomonnaies pour l'exemple
    symbols = symbols[:5]
    
    print(f"Analyse des scores de confiance pour {len(symbols)} cryptomonnaies...")
    
    # Créer le système d'indicateurs de confiance
    confidence_system = ConfidenceIndicatorSystem()
    
    # Analyser chaque cryptomonnaie
    results = {}
    
    for symbol in symbols:
        result = confidence_system.calculate_overall_confidence(symbol)
        results[symbol] = result
    
    # Créer un résumé des scores de confiance
    summary_data = []
    
    for symbol, result in results.items():
        summary_data.append({
            'symbol': symbol,
            'overall_confidence': result['overall_confidence'],
            'confidence_level': result['confidence_level']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Trier par score de confiance
    summary_df = summary_df.sort_values('overall_confidence', ascending=False)
    
    # Enregistrer le résumé
    summary_df.to_csv(os.path.join(CONFIDENCE_DIR, 'confidence_summary.csv'), index=False)
    
    # Créer un graphique de comparaison des scores de confiance
    plt.figure(figsize=(12, 8))
    
    # Définir les couleurs en fonction du niveau de confiance
    colors = []
    for level in summary_df['confidence_level']:
        if level == "Très élevé":
            colors.append('darkgreen')
        elif level == "Élevé":
            colors.append('green')
        elif level == "Moyen":
            colors.append('orange')
        elif level == "Faible":
            colors.append('red')
        else:
            colors.append('darkred')
    
    # Créer le graphique à barres
    bars = plt.bar(summary_df['symbol'], summary_df['overall_confidence'], color=colors)
    
    plt.axhline(y=0.8, color='darkgreen', linestyle='--', alpha=0.5)
    plt.axhline(y=0.65, color='green', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    plt.axhline(y=0.35, color='red', linestyle='--', alpha=0.5)
    
    plt.xlabel('Cryptomonnaie')
    plt.ylabel('Score de Confiance')
    plt.title('Comparaison des Scores de Confiance')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIDENCE_DIR, 'figures', 'confidence_scores_comparison.png'))
    plt.close()
    
    print("Analyse des scores de confiance terminée.")
    return results

if __name__ == "__main__":
    analyze_confidence_for_all_cryptos()
