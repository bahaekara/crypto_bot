"""
Crypto Alert System
Monitors cryptocurrencies and sends alerts when specific conditions are met
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CryptoAlert:
    def __init__(self, config_file=None):
        """
        Initialize the alert system
        
        Args:
            config_file: Path to configuration file (JSON)
        """
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                "check_interval": 15,  # minutes
                "alerts": {
                    "price_change": {
                        "enabled": True,
                        "threshold": 5.0  # percent
                    },
                    "volume_spike": {
                        "enabled": True,
                        "threshold": 200.0  # percent vs average
                    },
                    "rsi_overbought": {
                        "enabled": True,
                        "threshold": 70.0
                    },
                    "rsi_oversold": {
                        "enabled": True,
                        "threshold": 30.0
                    },
                    "macd_crossover": {
                        "enabled": True
                    },
                    "price_target": {
                        "enabled": True,
                        "targets": {}  # Will be populated from watchlist
                    }
                },
                "notification": {
                    "email": {
                        "enabled": True,
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "sender_email": os.getenv("EMAIL_ADDRESS", ""),
                        "sender_password": os.getenv("EMAIL_PASSWORD", ""),
                        "recipient_email": os.getenv("RECIPIENT_EMAIL", "")
                    },
                    "telegram": {
                        "enabled": False,
                        "token": os.getenv("TELEGRAM_TOKEN", ""),
                        "chat_id": os.getenv("TELEGRAM_CHAT_ID", "")
                    }
                },
                "watchlist": []  # Will be populated from crypto_list.py
            }
        
        # Create alert history directory
        self.alerts_dir = os.path.join(self.base_dir, "alerts", "history")
        os.makedirs(self.alerts_dir, exist_ok=True)
        
        # Load watchlist from crypto_list
        self._load_watchlist()
        
        # Initialize notification systems
        self._init_notifications()
        
        logger.info("Crypto Alert System initialized")
    
    def _load_watchlist(self):
        """
        Load watchlist from crypto_list.py
        """
        try:
            # Add project root to path
            sys.path.append(self.base_dir)
            from data.crypto_list import MAJOR_CRYPTOS, EMERGING_CRYPTOS, DEFI_CRYPTOS
            
            # Combine lists and remove duplicates
            all_cryptos = list(set(MAJOR_CRYPTOS + EMERGING_CRYPTOS + DEFI_CRYPTOS))
            
            # Add to watchlist with default settings
            for symbol in all_cryptos:
                # Skip if already in watchlist
                if any(w['symbol'] == symbol for w in self.config['watchlist']):
                    continue
                
                self.config['watchlist'].append({
                    "symbol": symbol,
                    "price_alerts": {
                        "min": None,  # Will be set dynamically
                        "max": None   # Will be set dynamically
                    },
                    "enabled": True
                })
            
            logger.info(f"Loaded {len(self.config['watchlist'])} cryptocurrencies to watchlist")
        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")
    
    def _init_notifications(self):
        """
        Initialize notification systems
        """
        # Test email configuration if enabled
        if self.config['notification']['email']['enabled']:
            email_config = self.config['notification']['email']
            if not all([email_config['smtp_server'], email_config['sender_email'], 
                       email_config['sender_password'], email_config['recipient_email']]):
                logger.warning("Email notifications enabled but configuration incomplete")
                self.config['notification']['email']['enabled'] = False
        
        # Test Telegram configuration if enabled
        if self.config['notification']['telegram']['enabled']:
            telegram_config = self.config['notification']['telegram']
            if not all([telegram_config['token'], telegram_config['chat_id']]):
                logger.warning("Telegram notifications enabled but configuration incomplete")
                self.config['notification']['telegram']['enabled'] = False
    
    def _get_latest_data(self, symbol, period="1d", interval="15m"):
        """
        Get the latest price data for a cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol
            period: Time period to fetch
            interval: Data interval
            
        Returns:
            DataFrame with price data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, data):
        """
        Calculate technical indicators from price data
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added indicators
        """
        if data is None or data.empty:
            return None
        
        try:
            # Calculate RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Calculate Moving Averages
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            
            # Calculate Bollinger Bands
            data['MA20_std'] = data['Close'].rolling(window=20).std()
            data['Upper_Band'] = data['MA20'] + (data['MA20_std'] * 2)
            data['Lower_Band'] = data['MA20'] - (data['MA20_std'] * 2)
            
            # Calculate daily price change percentage
            data['Price_Change'] = data['Close'].pct_change() * 100
            
            # Calculate volume change
            data['Volume_Change'] = data['Volume'].pct_change() * 100
            
            return data
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return data
    
    def _check_alerts(self, symbol, data):
        """
        Check if any alert conditions are met
        
        Args:
            symbol: Cryptocurrency symbol
            data: DataFrame with price and indicator data
            
        Returns:
            List of triggered alerts
        """
        if data is None or data.empty:
            return []
        
        alerts = []
        current_price = data['Close'].iloc[-1]
        
        # Get the most recent data point
        latest = data.iloc[-1]
        
        # Check price change alert
        if self.config['alerts']['price_change']['enabled']:
            threshold = self.config['alerts']['price_change']['threshold']
            if abs(latest['Price_Change']) >= threshold:
                direction = "up" if latest['Price_Change'] > 0 else "down"
                alerts.append({
                    "type": "price_change",
                    "message": f"{symbol} price changed {direction} by {abs(latest['Price_Change']):.2f}% to {current_price:.2f} USD",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": {
                        "symbol": symbol,
                        "price": current_price,
                        "change": latest['Price_Change'],
                        "direction": direction
                    }
                })
        
        # Check volume spike alert
        if self.config['alerts']['volume_spike']['enabled'] and 'Volume_Change' in latest:
            threshold = self.config['alerts']['volume_spike']['threshold']
            if latest['Volume_Change'] >= threshold:
                alerts.append({
                    "type": "volume_spike",
                    "message": f"{symbol} volume spiked by {latest['Volume_Change']:.2f}% to {latest['Volume']:.0f}",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": {
                        "symbol": symbol,
                        "volume": latest['Volume'],
                        "change": latest['Volume_Change']
                    }
                })
        
        # Check RSI alerts
        if 'RSI' in latest:
            # Overbought
            if self.config['alerts']['rsi_overbought']['enabled']:
                threshold = self.config['alerts']['rsi_overbought']['threshold']
                if latest['RSI'] >= threshold:
                    alerts.append({
                        "type": "rsi_overbought",
                        "message": f"{symbol} is overbought with RSI at {latest['RSI']:.2f}",
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "data": {
                            "symbol": symbol,
                            "rsi": latest['RSI'],
                            "price": current_price
                        }
                    })
            
            # Oversold
            if self.config['alerts']['rsi_oversold']['enabled']:
                threshold = self.config['alerts']['rsi_oversold']['threshold']
                if latest['RSI'] <= threshold:
                    alerts.append({
                        "type": "rsi_oversold",
                        "message": f"{symbol} is oversold with RSI at {latest['RSI']:.2f}",
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "data": {
                            "symbol": symbol,
                            "rsi": latest['RSI'],
                            "price": current_price
                        }
                    })
        
        # Check MACD crossover
        if self.config['alerts']['macd_crossover']['enabled'] and 'MACD' in latest and 'Signal_Line' in latest:
            # Check if MACD crossed above signal line
            if data['MACD'].iloc[-2] < data['Signal_Line'].iloc[-2] and latest['MACD'] > latest['Signal_Line']:
                alerts.append({
                    "type": "macd_crossover_bullish",
                    "message": f"{symbol} MACD crossed above signal line - bullish signal",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": {
                        "symbol": symbol,
                        "macd": latest['MACD'],
                        "signal": latest['Signal_Line'],
                        "price": current_price
                    }
                })
            
            # Check if MACD crossed below signal line
            elif data['MACD'].iloc[-2] > data['Signal_Line'].iloc[-2] and latest['MACD'] < latest['Signal_Line']:
                alerts.append({
                    "type": "macd_crossover_bearish",
                    "message": f"{symbol} MACD crossed below signal line - bearish signal",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": {
                        "symbol": symbol,
                        "macd": latest['MACD'],
                        "signal": latest['Signal_Line'],
                        "price": current_price
                    }
                })
        
        # Check price targets
        if self.config['alerts']['price_target']['enabled']:
            # Find this symbol in the watchlist
            for crypto in self.config['watchlist']:
                if crypto['symbol'] == symbol and crypto['enabled']:
                    # Check min price target
                    if crypto['price_alerts']['min'] is not None and current_price <= crypto['price_alerts']['min']:
                        alerts.append({
                            "type": "price_target_min",
                            "message": f"{symbol} reached minimum price target of {crypto['price_alerts']['min']:.2f} USD",
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "data": {
                                "symbol": symbol,
                                "price": current_price,
                                "target": crypto['price_alerts']['min']
                            }
                        })
                    
                    # Check max price target
                    if crypto['price_alerts']['max'] is not None and current_price >= crypto['price_alerts']['max']:
                        alerts.append({
                            "type": "price_target_max",
                            "message": f"{symbol} reached maximum price target of {crypto['price_alerts']['max']:.2f} USD",
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "data": {
                                "symbol": symbol,
                                "price": current_price,
                                "target": crypto['price_alerts']['max']
                            }
                        })
                    
                    break
        
        return alerts
    
    def _send_notification(self, alert):
        """
        Send notification for an alert
        
        Args:
            alert: Alert data dictionary
        """
        # Send email notification
        if self.config['notification']['email']['enabled']:
            try:
                email_config = self.config['notification']['email']
                
                message = MIMEMultipart()
                message['From'] = email_config['sender_email']
                message['To'] = email_config['recipient_email']
                message['Subject'] = f"Crypto Alert: {alert['type']} for {alert['data']['symbol']}"
                
                # Create email body
                body = f"""
                <html>
                <body>
                    <h2>Crypto Alert</h2>
                    <p><strong>Type:</strong> {alert['type']}</p>
                    <p><strong>Message:</strong> {alert['message']}</p>
                    <p><strong>Time:</strong> {alert['time']}</p>
                    <h3>Details:</h3>
                    <table border="1">
                """
                
                for key, value in alert['data'].items():
                    if isinstance(value, float):
                        value = f"{value:.2f}"
                    body += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
                
                body += """
                    </table>
                    <p>This is an automated message from your Crypto Alert System.</p>
                </body>
                </html>
                """
                
                message.attach(MIMEText(body, 'html'))
                
                # Connect to server and send email
                with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                    server.starttls()
                    server.login(email_config['sender_email'], email_config['sender_password'])
                    server.send_message(message)
                
                logger.info(f"Email alert sent for {alert['data']['symbol']}: {alert['type']}")
            except Exception as e:
                logger.error(f"Error sending email notification: {e}")
        
        # Send Telegram notification
        if self.config['notification']['telegram']['enabled']:
            try:
                telegram_config = self.config['notification']['telegram']
                
                # Format message
                message = f"🚨 *Crypto Alert* 🚨\n\n"
                message += f"*Type:* {alert['type']}\n"
                message += f"*Symbol:* {alert['data']['symbol']}\n"
                message += f"*Message:* {alert['message']}\n"
                message += f"*Time:* {alert['time']}\n\n"
                
                # Add details
                message += "*Details:*\n"
                for key, value in alert['data'].items():
                    if isinstance(value, float):
                        value = f"{value:.2f}"
                    message += f"- {key}: {value}\n"
                
                # Send message
                url = f"https://api.telegram.org/bot{telegram_config['token']}/sendMessage"
                params = {
                    "chat_id": telegram_config['chat_id'],
                    "text": message,
                    "parse_mode": "Markdown"
                }
                
                response = requests.post(url, params=params)
                if response.status_code == 200:
                    logger.info(f"Telegram alert sent for {alert['data']['symbol']}: {alert['type']}")
                else:
                    logger.error(f"Telegram API error: {response.text}")
            except Exception as e:
                logger.error(f"Error sending Telegram notification: {e}")
    
    def _save_alert(self, alert):
        """
        Save alert to history file
        
        Args:
            alert: Alert data dictionary
        """
        try:
            # Create filename based on date
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = os.path.join(self.alerts_dir, f"alerts_{date_str}.json")
            
            # Load existing alerts if file exists
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    alerts = json.load(f)
            else:
                alerts = []
            
            # Add new alert
            alerts.append(alert)
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(alerts, f, indent=2)
            
            logger.info(f"Alert saved to history: {alert['type']} for {alert['data']['symbol']}")
        except Exception as e:
            logger.error(f"Error saving alert to history: {e}")
    
    def update_watchlist(self):
        """
        Update the price targets in the watchlist based on current market prices
        """
        logger.info("Updating watchlist price targets")
        
        for crypto in self.config['watchlist']:
            if not crypto['enabled']:
                continue
            
            # Get current price
            data = self._get_latest_data(crypto['symbol'], period="1d", interval="1d")
            if data is None or data.empty:
                continue
            
            current_price = data['Close'].iloc[-1]
            
            # Set price alerts if not already set
            if crypto['price_alerts']['min'] is None:
                crypto['price_alerts']['min'] = round(current_price * 0.9, 2)  # 10% below current
            
            if crypto['price_alerts']['max'] is None:
                crypto['price_alerts']['max'] = round(current_price * 1.1, 2)  # 10% above current
        
        # Save updated configuration
        self.save_config()
        
        logger.info("Watchlist price targets updated")
    
    def save_config(self):
        """
        Save the current configuration to file
        """
        try:
            config_file = os.path.join(self.base_dir, "alerts", "config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def run(self, single_pass=False):
        """
        Run the alert system
        
        Args:
            single_pass: If True, run once and exit; otherwise run continuously
        """
        logger.info("Starting Crypto Alert System")
        
        try:
            # Update watchlist price targets on first run
            self.update_watchlist()
            
            while True:
                start_time = time.time()
                logger.info("Checking cryptocurrencies for alert conditions")
                
                # Check each cryptocurrency in the watchlist
                for crypto in self.config['watchlist']:
                    if not crypto['enabled']:
                        continue
                    
                    symbol = crypto['symbol']
                    logger.info(f"Checking {symbol}")
                    
                    # Get latest data
                    data = self._get_latest_data(symbol, period="5d", interval="15m")
                    if data is None:
                        continue
                    
                    # Calculate indicators
                    data = self._calculate_indicators(data)
                    if data is None:
                        continue
                    
                    # Check for alerts
                    alerts = self._check_alerts(symbol, data)
                    
                    # Process triggered alerts
                    for alert in alerts:
                        logger.info(f"Alert triggered: {alert['type']} for {symbol}")
                        self._send_notification(alert)
                        self._save_alert(alert)
                
                # Single pass mode for testing
                if single_pass:
                    logger.info("Single pass completed")
                    break
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                interval = self.config['check_interval'] * 60  # convert to seconds
                sleep_time = max(interval - elapsed, 0)
                
                if sleep_time > 0:
                    logger.info(f"Sleeping for {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Alert system stopped by user")
        except Exception as e:
            logger.error(f"Error in alert system: {e}")
            raise

def main():
    """
    Main function to run the alert system
    """
    # Check if config file exists
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file = os.path.join(base_dir, "alerts", "config.json")
    
    # Create alert system
    if os.path.exists(config_file):
        alert_system = CryptoAlert(config_file)
    else:
        alert_system = CryptoAlert()
        # Make sure directory exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        # Save default config
        alert_system.save_config()
    
    # Get command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Crypto Alert System')
    parser.add_argument('--test', action='store_true', help='Run a single test pass')
    args = parser.parse_args()
    
    # Run alert system
    alert_system.run(single_pass=args.test)

if __name__ == "__main__":
    main()