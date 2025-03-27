import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
from tabulate import tabulate
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Strategy:
    """Base class for all trading strategies"""
    
    def __init__(self, name="Generic Strategy"):
        """
        Initialize a trading strategy
        
        Args:
            name: Strategy name
        """
        self.name = name
    
    def generate_signals(self, data):
        """
        Generate trading signals for the data
        
        Args:
            data: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with added signal column (1=buy, -1=sell, 0=hold)
        """
        raise NotImplementedError("Subclass must implement this method")

class MovingAverageCrossStrategy(Strategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, short_window=20, long_window=50):
        """
        Initialize MA crossover strategy
        
        Args:
            short_window: Short moving average window
            long_window: Long moving average window
        """
        super().__init__(f"MA Crossover {short_window}/{long_window}")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        """
        Generate trading signals for MA crossover
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        # Make a copy of the data
        signals = data.copy()
        
        # Create short and long moving averages
        signals[f'SMA{self.short_window}'] = signals['Close'].rolling(window=self.short_window).mean()
        signals[f'SMA{self.long_window}'] = signals['Close'].rolling(window=self.long_window).mean()
        
        # Create signals
        signals['Signal'] = 0
        
        # Generate buy signals
        signals.loc[signals[f'SMA{self.short_window}'] > signals[f'SMA{self.long_window}'], 'Signal'] = 1
        
        # Generate sell signals
        signals.loc[signals[f'SMA{self.short_window}'] < signals[f'SMA{self.long_window}'], 'Signal'] = -1
        
        return signals

class RSIStrategy(Strategy):
    """RSI Overbought/Oversold Strategy"""
    
    def __init__(self, rsi_window=14, oversold=30, overbought=70):
        """
        Initialize RSI strategy
        
        Args:
            rsi_window: RSI calculation window
            oversold: RSI level to consider oversold
            overbought: RSI level to consider overbought
        """
        super().__init__(f"RSI {rsi_window} ({oversold}/{overbought})")
        self.rsi_window = rsi_window
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        # Make a copy of the data
        signals = data.copy()
        
        # Calculate RSI if not already present
        if 'RSI' not in signals.columns:
            delta = signals['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.rsi_window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_window).mean()
            rs = gain / loss
            signals['RSI'] = 100 - (100 / (1 + rs))
        
        # Create signals
        signals['Signal'] = 0
        
        # Buy signal: RSI crosses below oversold and then back above
        signals['Oversold'] = signals['RSI'] < self.oversold
        signals['OversoldExit'] = (signals['RSI'] > self.oversold) & (signals['Oversold'].shift(1))
        signals.loc[signals['OversoldExit'], 'Signal'] = 1
        
        # Sell signal: RSI crosses above overbought and then back below
        signals['Overbought'] = signals['RSI'] > self.overbought
        signals['OverboughtExit'] = (signals['RSI'] < self.overbought) & (signals['Overbought'].shift(1))
        signals.loc[signals['OverboughtExit'], 'Signal'] = -1
        
        # Clean up temporary columns
        signals = signals.drop(['Oversold', 'OversoldExit', 'Overbought', 'OverboughtExit'], axis=1)
        
        return signals

class MACDStrategy(Strategy):
    """MACD Crossover Strategy"""
    
    def __init__(self, fast=12, slow=26, signal=9):
        """
        Initialize MACD strategy
        
        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        """
        super().__init__(f"MACD {fast}/{slow}/{signal}")
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self, data):
        """
        Generate trading signals based on MACD
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        # Make a copy of the data
        signals = data.copy()
        
        # Calculate MACD if not already present
        if 'MACD' not in signals.columns or 'Signal_Line' not in signals.columns:
            # Calculate MACD
            exp1 = signals['Close'].ewm(span=self.fast, adjust=False).mean()
            exp2 = signals['Close'].ewm(span=self.slow, adjust=False).mean()
            signals['MACD'] = exp1 - exp2
            signals['Signal_Line'] = signals['MACD'].ewm(span=self.signal, adjust=False).mean()
            signals['MACD_Histogram'] = signals['MACD'] - signals['Signal_Line']
        
        # Create signals
        signals['Signal'] = 0
        
        # Buy signal: MACD crosses above signal line
        signals['Buy'] = (signals['MACD'] > signals['Signal_Line']) & (signals['MACD'].shift(1) <= signals['Signal_Line'].shift(1))
        signals.loc[signals['Buy'], 'Signal'] = 1
        
        # Sell signal: MACD crosses below signal line
        signals['Sell'] = (signals['MACD'] < signals['Signal_Line']) & (signals['MACD'].shift(1) >= signals['Signal_Line'].shift(1))
        signals.loc[signals['Sell'], 'Signal'] = -1
        
        # Clean up temporary columns
        signals = signals.drop(['Buy', 'Sell'], axis=1)
        
        return signals

class BollingerBandsStrategy(Strategy):
    """Bollinger Bands Strategy"""
    
    def __init__(self, window=20, num_std=2):
        """
        Initialize Bollinger Bands strategy
        
        Args:
            window: Moving average window
            num_std: Number of standard deviations for bands
        """
        super().__init__(f"Bollinger Bands {window}/{num_std}")
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Bollinger Bands
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        # Make a copy of the data
        signals = data.copy()
        
        # Calculate Bollinger Bands if not already present
        if 'Upper_Band' not in signals.columns or 'Lower_Band' not in signals.columns:
            signals['MA'] = signals['Close'].rolling(window=self.window).mean()
            signals['Std'] = signals['Close'].rolling(window=self.window).std()
            signals['Upper_Band'] = signals['MA'] + (signals['Std'] * self.num_std)
            signals['Lower_Band'] = signals['MA'] - (signals['Std'] * self.num_std)
        
        # Create signals
        signals['Signal'] = 0
        
        # Buy signal: Price crosses below lower band and then back above
        signals['Below_Lower'] = signals['Close'] < signals['Lower_Band']
        signals['Lower_Cross_Up'] = (signals['Close'] > signals['Lower_Band']) & (signals['Below_Lower'].shift(1))
        signals.loc[signals['Lower_Cross_Up'], 'Signal'] = 1
        
        # Sell signal: Price crosses above upper band and then back below
        signals['Above_Upper'] = signals['Close'] > signals['Upper_Band']
        signals['Upper_Cross_Down'] = (signals['Close'] < signals['Upper_Band']) & (signals['Above_Upper'].shift(1))
        signals.loc[signals['Upper_Cross_Down'], 'Signal'] = -1
        
        # Clean up temporary columns
        signals = signals.drop(['Below_Lower', 'Lower_Cross_Up', 'Above_Upper', 'Upper_Cross_Down'], axis=1)
        
        return signals

class Backtest:
    """Backtest class to evaluate trading strategies"""
    
    def __init__(self, symbol, data, strategy, initial_capital=10000.0, commission=0.001):
        """
        Initialize backtest
        
        Args:
            symbol: Cryptocurrency symbol
            data: DataFrame with OHLCV data
            strategy: Trading strategy to test
            initial_capital: Starting capital
            commission: Commission rate per trade
        """
        self.symbol = symbol
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        
        # Generate signals
        self.data = strategy.generate_signals(data)
        
        # Make sure index is datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except:
                logger.warning(f"Unable to convert index to datetime for {symbol}")
        
        logger.info(f"Initialized backtest for {symbol} using {strategy.name}")
    
    def run(self):
        """
        Run the backtest
        
        Returns:
            DataFrame with portfolio performance
        """
        # Create portfolio dataframe
        portfolio = self.data[['Close']].copy()
        
        # Initialize holdings and cash columns
        portfolio['Signal'] = self.data['Signal']
        portfolio['Holdings'] = 0.0
        portfolio['Cash'] = self.initial_capital
        portfolio['Position'] = 0
        portfolio['Trade'] = 0
        
        # Process signals and build portfolio
        for i in range(1, len(portfolio)):
            # Default is yesterday's position
            portfolio.loc[portfolio.index[i], 'Position'] = portfolio.loc[portfolio.index[i-1], 'Position']
            
            # Process buy signal
            if portfolio.loc[portfolio.index[i], 'Signal'] == 1 and portfolio.loc[portfolio.index[i-1], 'Position'] == 0:
                # Calculate how many coins we can buy
                price = portfolio.loc[portfolio.index[i], 'Close']
                available_cash = portfolio.loc[portfolio.index[i-1], 'Cash']
                trade_cost = price * (1 + self.commission)
                units = available_cash / trade_cost
                
                # Update position and cash
                portfolio.loc[portfolio.index[i], 'Position'] = units
                portfolio.loc[portfolio.index[i], 'Cash'] = available_cash - (units * trade_cost)
                portfolio.loc[portfolio.index[i], 'Trade'] = 1  # Buy
            
            # Process sell signal
            elif portfolio.loc[portfolio.index[i], 'Signal'] == -1 and portfolio.loc[portfolio.index[i-1], 'Position'] > 0:
                # Calculate cash from selling
                price = portfolio.loc[portfolio.index[i], 'Close']
                units = portfolio.loc[portfolio.index[i-1], 'Position']
                sale_value = units * price * (1 - self.commission)
                
                # Update position and cash
                portfolio.loc[portfolio.index[i], 'Position'] = 0
                portfolio.loc[portfolio.index[i], 'Cash'] = portfolio.loc[portfolio.index[i-1], 'Cash'] + sale_value
                portfolio.loc[portfolio.index[i], 'Trade'] = -1  # Sell
            
            # No trade
            else:
                portfolio.loc[portfolio.index[i], 'Cash'] = portfolio.loc[portfolio.index[i-1], 'Cash']
            
            # Calculate holdings value
            portfolio.loc[portfolio.index[i], 'Holdings'] = portfolio.loc[portfolio.index[i], 'Position'] * portfolio.loc[portfolio.index[i], 'Close']
        
        # Calculate total portfolio value and returns
        portfolio['Total'] = portfolio['Holdings'] + portfolio['Cash']
        portfolio['Returns'] = portfolio['Total'].pct_change()
        portfolio['CumulativeReturns'] = (1 + portfolio['Returns']).cumprod()
        
        # Calculate buy & hold returns for comparison
        price_initial = portfolio['Close'].iloc[0]
        price_final = portfolio['Close'].iloc[-1]
        buyhold_return = (price_final / price_initial) - 1
        
        logger.info(f"Backtest completed for {self.symbol} using {self.strategy.name}")
        
        return portfolio
    
    def get_performance_metrics(self, portfolio):
        """
        Calculate performance metrics for the backtest
        
        Args:
            portfolio: DataFrame with portfolio performance
            
        Returns:
            Dictionary of performance metrics
        """
        # Extract returns
        returns = portfolio['Returns'].dropna()
        
        # Calculate metrics
        total_return = (portfolio['Total'].iloc[-1] / self.initial_capital) - 1
        annual_return = ((1 + total_return) ** (252 / len(returns))) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        max_drawdown = (portfolio['Total'] / portfolio['Total'].cummax() - 1).min()
        
        # Count trades
        buys = (portfolio['Trade'] == 1).sum()
        sells = (portfolio['Trade'] == -1).sum()
        
        # Calculate win rate and profit factor
        trade_returns = []
        in_position = False
        entry_price = 0
        
        for i in range(len(portfolio)):
            if portfolio['Trade'].iloc[i] == 1:  # Buy
                in_position = True
                entry_price = portfolio['Close'].iloc[i]
            elif portfolio['Trade'].iloc[i] == -1 and in_position:  # Sell
                exit_price = portfolio['Close'].iloc[i]
                trade_return = (exit_price / entry_price) - 1
                trade_returns.append(trade_return)
                in_position = False
        
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0
        
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r <= 0]
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if sum(losing_trades) != 0 else float('inf')
        
        # Calculate buy & hold return for comparison
        price_initial = portfolio['Close'].iloc[0]
        price_final = portfolio['Close'].iloc[-1]
        buyhold_return = (price_final / price_initial) - 1
        
        # Create metrics dictionary
        metrics = {
            'Symbol': self.symbol,
            'Strategy': self.strategy.name,
            'Start Date': portfolio.index[0].strftime('%Y-%m-%d'),
            'End Date': portfolio.index[-1].strftime('%Y-%m-%d'),
            'Duration (Days)': (portfolio.index[-1] - portfolio.index[0]).days,
            'Initial Capital': self.initial_capital,
            'Final Capital': portfolio['Total'].iloc[-1],
            'Total Return (%)': total_return * 100,
            'Buy & Hold Return (%)': buyhold_return * 100,
            'Outperformance (%)': (total_return - buyhold_return) * 100,
            'Annual Return (%)': annual_return * 100,
            'Volatility (%)': volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Number of Trades': buys,
            'Win Rate (%)': win_rate * 100 if trade_returns else 0,
            'Profit Factor': profit_factor,
            'Commission Rate (%)': self.commission * 100
        }
        
        return metrics
    
    def plot_performance(self, portfolio, save_path=None):
        """
        Plot backtest performance
        
        Args:
            portfolio: DataFrame with portfolio performance
            save_path: Path to save the plot image
            
        Returns:
            Figure object
        """
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot 1: Price and position
        ax1 = axs[0]
        ax1.plot(portfolio.index, portfolio['Close'], label='Price', color='blue', alpha=0.5)
        
        # Plot buy and sell signals
        buys = portfolio[portfolio['Trade'] == 1]
        sells = portfolio[portfolio['Trade'] == -1]
        
        ax1.scatter(buys.index, buys['Close'], color='green', marker='^', s=100, label='Buy Signal')
        ax1.scatter(sells.index, sells['Close'], color='red', marker='v', s=100, label='Sell Signal')
        
        # Add moving averages if they exist
        for col in portfolio.columns:
            if col.startswith('SMA') or col == 'MA':
                ax1.plot(portfolio.index, portfolio[col], label=col, alpha=0.7)
        
        ax1.set_title(f'{self.symbol} - {self.strategy.name} Backtest')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Portfolio value vs Buy & Hold
        ax2 = axs[1]
        ax2.plot(portfolio.index, portfolio['Total'], label='Strategy', color='green')
        
        # Calculate buy & hold portfolio
        buyhold = pd.Series(index=portfolio.index)
        units = self.initial_capital / portfolio['Close'].iloc[0]
        buyhold = units * portfolio['Close']
        ax2.plot(portfolio.index, buyhold, label='Buy & Hold', color='blue', alpha=0.7)
        
        ax2.set_title('Portfolio Value')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Drawdown
        ax3 = axs[2]
        drawdown = (portfolio['Total'] / portfolio['Total'].cummax() - 1) * 100
        ax3.fill_between(portfolio.index, drawdown, 0, color='red', alpha=0.3)
        ax3.set_title('Drawdown (%)')
        ax3.set_ylabel('Drawdown %')
        ax3.set_ylim(drawdown.min() * 1.1, 1)
        ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        return fig
    
    def generate_trade_list(self, portfolio):
        """
        Generate a list of all trades
        
        Args:
            portfolio: DataFrame with portfolio performance
            
        Returns:
            DataFrame with trade details
        """
        trades = []
        in_position = False
        entry_date = None
        entry_price = 0
        entry_units = 0
        
        for i in range(len(portfolio)):
            date = portfolio.index[i]
            
            # Buy signal
            if portfolio['Trade'].iloc[i] == 1:
                entry_date = date
                entry_price = portfolio['Close'].iloc[i]
                entry_units = portfolio['Position'].iloc[i]
                in_position = True
            
            # Sell signal
            elif portfolio['Trade'].iloc[i] == -1 and in_position:
                exit_date = date
                exit_price = portfolio['Close'].iloc[i]
                
                trade_return = (exit_price / entry_price) - 1
                profit = entry_units * (exit_price - entry_price)
                duration = (exit_date - entry_date).days
                
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': exit_date,
                    'Duration (Days)': duration,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Units': entry_units,
                    'Return (%)': trade_return * 100,
                    'Profit': profit,
                    'Result': 'Win' if trade_return > 0 else 'Loss'
                })
                
                in_position = False
        
        # Handle open position at the end
        if in_position:
            exit_date = portfolio.index[-1]
            exit_price = portfolio['Close'].iloc[-1]
            
            trade_return = (exit_price / entry_price) - 1
            profit = entry_units * (exit_price - entry_price)
            duration = (exit_date - entry_date).days
            
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Duration (Days)': duration,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Units': entry_units,
                'Return (%)': trade_return * 100,
                'Profit': profit,
                'Result': 'Open Position'
            })
        
        return pd.DataFrame(trades)
    
    def generate_report(self, portfolio, output_dir=None):
        """
        Generate a comprehensive backtest report
        
        Args:
            portfolio: DataFrame with portfolio performance
            output_dir: Directory to save the report
            
        Returns:
            Dictionary with report data
        """
        # Create output directory
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'analysis', 'backtesting')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report data
        metrics = self.get_performance_metrics(portfolio)
        trades_df = self.generate_trade_list(portfolio)
        
        # Format the report data
        report_data = {
            'symbol': self.symbol,
            'strategy': self.strategy.name,
            'metrics': metrics,
            'trades': trades_df.to_dict('records')
        }
        
        # Save metrics to JSON
        metrics_file = os.path.join(output_dir, f"{self.symbol}_{self.strategy.name.replace('/', '_')}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save trades to CSV
        trades_file = os.path.join(output_dir, f"{self.symbol}_{self.strategy.name.replace('/', '_')}_trades.csv")
        trades_df.to_csv(trades_file, index=False)
        
        # Generate performance plot
        plot_file = os.path.join(output_dir, f"{self.symbol}_{self.strategy.name.replace('/', '_')}_performance.png")
        self.plot_performance(portfolio, save_path=plot_file)
        
        # Generate markdown report
        report_file = os.path.join(output_dir, f"{self.symbol}_{self.strategy.name.replace('/', '_')}_report.md")
        
        with open(report_file, 'w') as f:
            f.write(f"# Backtest Report: {self.symbol} - {self.strategy.name}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Symbol:** {self.symbol}\n")
            f.write(f"- **Strategy:** {self.strategy.name}\n")
            f.write(f"- **Period:** {metrics['Start Date']} to {metrics['End Date']} ({metrics['Duration (Days)']} days)\n")
            f.write(f"- **Initial Capital:** ${metrics['Initial Capital']:,.2f}\n")
            f.write(f"- **Final Capital:** ${metrics['Final Capital']:,.2f}\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            metrics_to_show = [
                'Total Return (%)', 'Buy & Hold Return (%)', 'Outperformance (%)', 
                'Annual Return (%)', 'Volatility (%)', 'Sharpe Ratio', 
                'Max Drawdown (%)', 'Number of Trades', 'Win Rate (%)', 'Profit Factor'
            ]
            
            for metric in metrics_to_show:
                value = metrics[metric]
                if isinstance(value, float):
                    f.write(f"| {metric} | {value:.2f} |\n")
                else:
                    f.write(f"| {metric} | {value} |\n")
            
            f.write("\n## Trading Statistics\n\n")
            
            if len(trades_df) > 0:
                # Calculate stats
                win_trades = trades_df[trades_df['Return (%)'] > 0]
                loss_trades = trades_df[trades_df['Return (%)'] <= 0]
                
                avg_win = win_trades['Return (%)'].mean() if len(win_trades) > 0 else 0
                avg_loss = loss_trades['Return (%)'].mean() if len(loss_trades) > 0 else 0
                avg_duration = trades_df['Duration (Days)'].mean()
                
                f.write(f"- **Total Trades:** {len(trades_df)}\n")
                f.write(f"- **Winning Trades:** {len(win_trades)} ({len(win_trades)/len(trades_df)*100:.1f}%)\n")
                f.write(f"- **Losing Trades:** {len(loss_trades)} ({len(loss_trades)/len(trades_df)*100:.1f}%)\n")
                f.write(f"- **Average Winning Trade:** {avg_win:.2f}%\n")
                f.write(f"- **Average Losing Trade:** {avg_loss:.2f}%\n")
                f.write(f"- **Average Trade Duration:** {avg_duration:.1f} days\n")
                
                f.write("\n## Recent Trades\n\n")
                f.write("| Entry Date | Exit Date | Entry Price | Exit Price | Return (%) | Result |\n")
                f.write("|------------|-----------|-------------|------------|------------|--------|\n")
                
                # Show last 10 trades
                for _, trade in trades_df.tail(10).iterrows():
                    f.write(f"| {trade['Entry Date'].strftime('%Y-%m-%d')} | {trade['Exit Date'].strftime('%Y-%m-%d')} | ")
                    f.write(f"${trade['Entry Price']:.2f} | ${trade['Exit Price']:.2f} | ")
                    f.write(f"{trade['Return (%)']:.2f}% | {trade['Result']} |\n")
            else:
                f.write("No trades were executed during the backtest period.\n")
            
            f.write("\n## Visualization\n\n")
            f.write(f"![Performance Chart]({os.path.basename(plot_file)})\n\n")
            
            f.write("## Conclusion\n\n")
            
            # Generate conclusion
            outperformance = metrics['Outperformance (%)']
            if outperformance > 5:
                f.write(f"The {self.strategy.name} strategy significantly outperformed the buy & hold approach by {outperformance:.2f}%. ")
                f.write("This strategy showed strong performance and could be considered for real trading after further optimization.\n\n")
            elif outperformance > 0:
                f.write(f"The {self.strategy.name} strategy slightly outperformed the buy & hold approach by {outperformance:.2f}%. ")
                f.write("The strategy shows promise but may need further optimization for better results.\n\n")
            else:
                f.write(f"The {self.strategy.name} strategy underperformed the buy & hold approach by {abs(outperformance):.2f}%. ")
                f.write("This strategy would need significant improvement or may not be suitable for this cryptocurrency in the current market conditions.\n\n")
            
            f.write("*This report was automatically generated by the Crypto Backtest System.*\n")
        
        logger.info(f"Backtest report generated: {report_file}")
        
        return report_data

def load_historical_data(symbol, data_dir=None):
    """
    Load historical data for a cryptocurrency
    
    Args:
        symbol: Cryptocurrency symbol
        data_dir: Directory containing historical data files
        
    Returns:
        DataFrame with historical data
    """
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data', 'historical')
    
    # Try different filename formats
    file_formats = [
        f"{symbol}.csv",
        f"{symbol.replace('-', '_')}.csv",
        f"{symbol}.parquet",
        f"{symbol.replace('-', '_')}.parquet"
    ]
    
    for file_format in file_formats:
        file_path = os.path.join(data_dir, file_format)
        if os.path.exists(file_path):
            logger.info(f"Loading data from {file_path}")
            
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                continue
            
            # Convert Date column to datetime if needed
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
            
            # Check if required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns in {file_path}: {missing_cols}")
                continue
            
            return data
    
    logger.error(f"No valid data file found for {symbol} in {data_dir}")
    return None

def run_backtest(symbol, strategy, data_dir=None, output_dir=None, initial_capital=10000.0, commission=0.001):
    """
    Run a backtest for a specific symbol and strategy
    
    Args:
        symbol: Cryptocurrency symbol
        strategy: Strategy object to test
        data_dir: Directory containing historical data
        output_dir: Directory to save results
        initial_capital: Initial capital for the backtest
        commission: Commission rate per trade
        
    Returns:
        Dictionary with backtest results
    """
    # Load historical data
    data = load_historical_data(symbol, data_dir)
    if data is None:
        logger.error(f"Failed to load data for {symbol}")
        return None
    
    # Initialize and run backtest
    backtest = Backtest(symbol, data, strategy, initial_capital, commission)
    portfolio = backtest.run()
    
    # Generate report
    report = backtest.generate_report(portfolio, output_dir)
    
    return report

def run_strategy_comparison(symbol, strategies, data_dir=None, output_dir=None, initial_capital=10000.0, commission=0.001):
    """
    Compare multiple strategies on the same cryptocurrency
    
    Args:
        symbol: Cryptocurrency symbol
        strategies: List of Strategy objects to test
        data_dir: Directory containing historical data
        output_dir: Directory to save results
        initial_capital: Initial capital for the backtest
        commission: Commission rate per trade
        
    Returns:
        DataFrame comparing strategy performance
    """
    results = []
    
    for strategy in strategies:
        logger.info(f"Running {strategy.name} on {symbol}")
        report = run_backtest(symbol, strategy, data_dir, output_dir, initial_capital, commission)
        
        if report:
            results.append(report['metrics'])
    
    if not results:
        logger.error(f"No successful backtest results for {symbol}")
        return None
    
    # Create comparison dataframe
    comparison = pd.DataFrame(results)
    
    # Save comparison to CSV
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'analysis', 'backtesting')
    
    comparison_file = os.path.join(output_dir, f"{symbol}_strategy_comparison.csv")
    comparison.to_csv(comparison_file, index=False)
    
    # Generate comparison chart
    metrics_to_plot = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
    
    # Create a figure with a subplot for each metric
    fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 10))
    
    for i, metric in enumerate(metrics_to_plot):
        axs[i].bar(comparison['Strategy'], comparison[metric])
        axs[i].set_title(metric)
        axs[i].set_ylabel(metric)
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_chart = os.path.join(output_dir, f"{symbol}_strategy_comparison.png")
    plt.savefig(comparison_chart, dpi=300, bbox_inches='tight')
    
    # Generate comparison markdown report
    report_file = os.path.join(output_dir, f"{symbol}_strategy_comparison_report.md")
    
    with open(report_file, 'w') as f:
        f.write(f"# Strategy Comparison Report: {symbol}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This report compares {len(strategies)} trading strategies for {symbol}.\n\n")
        
        f.write("## Performance Comparison\n\n")
        f.write("| Strategy | Total Return (%) | Annual Return (%) | Sharpe Ratio | Max Drawdown (%) | Win Rate (%) | Profit Factor |\n")
        f.write("|----------|------------------|-------------------|--------------|-----------------|--------------|---------------|\n")
        
        for _, row in comparison.iterrows():
            f.write(f"| {row['Strategy']} | {row['Total Return (%)']:.2f} | {row['Annual Return (%)']:.2f} | ")
            f.write(f"{row['Sharpe Ratio']:.2f} | {row['Max Drawdown (%)']:.2f} | {row['Win Rate (%)']:.2f} | {row['Profit Factor']:.2f} |\n")
        
        f.write("\n## Visualization\n\n")
        f.write(f"![Strategy Comparison]({os.path.basename(comparison_chart)})\n\n")
        
        # Find best strategy by return
        best_return = comparison.loc[comparison['Total Return (%)'].idxmax()]
        best_sharpe = comparison.loc[comparison['Sharpe Ratio'].idxmax()]
        
        f.write("## Conclusions\n\n")
        f.write(f"- **Best Return Strategy:** {best_return['Strategy']} with {best_return['Total Return (%)']:.2f}% return\n")
        f.write(f"- **Best Risk-Adjusted Strategy:** {best_sharpe['Strategy']} with Sharpe Ratio of {best_sharpe['Sharpe Ratio']:.2f}\n\n")
        
        if best_return['Total Return (%)'] > 0 and best_return['Sharpe Ratio'] > 1:
            f.write(f"The {best_return['Strategy']} strategy shows promising results and could be considered for further optimization or live trading.\n")
        else:
            f.write("None of the tested strategies showed exceptional performance. Consider further optimization or alternative strategies.\n")
    
    logger.info(f"Strategy comparison report generated: {report_file}")
    
    return comparison

def run_multi_asset_backtest(symbols, strategy, data_dir=None, output_dir=None, initial_capital=10000.0, commission=0.001):
    """
    Run the same strategy across multiple cryptocurrencies
    
    Args:
        symbols: List of cryptocurrency symbols
        strategy: Strategy object to test
        data_dir: Directory containing historical data
        output_dir: Directory to save results
        initial_capital: Initial capital for the backtest
        commission: Commission rate per trade
        
    Returns:
        DataFrame comparing results across cryptocurrencies
    """
    results = []
    
    for symbol in symbols:
        logger.info(f"Running {strategy.name} on {symbol}")
        report = run_backtest(symbol, strategy, data_dir, output_dir, initial_capital, commission)
        
        if report:
            results.append(report['metrics'])
    
    if not results:
        logger.error("No successful backtest results")
        return None
    
    # Create comparison dataframe
    comparison = pd.DataFrame(results)
    
    # Save comparison to CSV
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'analysis', 'backtesting')
    
    comparison_file = os.path.join(output_dir, f"{strategy.name.replace('/', '_')}_multi_asset_comparison.csv")
    comparison.to_csv(comparison_file, index=False)
    
    # Generate comparison chart
    metrics_to_plot = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
    
    # Create a figure with a subplot for each metric
    fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 12))
    
    for i, metric in enumerate(metrics_to_plot):
        axs[i].bar(comparison['Symbol'], comparison[metric])
        axs[i].set_title(metric)
        axs[i].set_ylabel(metric)
        axs[i].grid(True, alpha=0.3)
        
        # Rotate x labels if many symbols
        if len(symbols) > 5:
            plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    comparison_chart = os.path.join(output_dir, f"{strategy.name.replace('/', '_')}_multi_asset_comparison.png")
    plt.savefig(comparison_chart, dpi=300, bbox_inches='tight')
    
    # Generate comparison markdown report
    report_file = os.path.join(output_dir, f"{strategy.name.replace('/', '_')}_multi_asset_comparison_report.md")
    
    with open(report_file, 'w') as f:
        f.write(f"# Multi-Asset Strategy Report: {strategy.name}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This report evaluates the {strategy.name} strategy across {len(symbols)} cryptocurrencies.\n\n")
        
        f.write("## Performance Comparison\n\n")
        f.write("| Symbol | Total Return (%) | Annual Return (%) | Sharpe Ratio | Max Drawdown (%) | Win Rate (%) | Profit Factor |\n")
        f.write("|--------|------------------|-------------------|--------------|-----------------|--------------|---------------|\n")
        
        for _, row in comparison.iterrows():
            f.write(f"| {row['Symbol']} | {row['Total Return (%)']:.2f} | {row['Annual Return (%)']:.2f} | ")
            f.write(f"{row['Sharpe Ratio']:.2f} | {row['Max Drawdown (%)']:.2f} | {row['Win Rate (%)']:.2f} | {row['Profit Factor']:.2f} |\n")
        
        f.write("\n## Visualization\n\n")
        f.write(f"![Multi-Asset Comparison]({os.path.basename(comparison_chart)})\n\n")
        
        # Find best asset by return
        best_return = comparison.loc[comparison['Total Return (%)'].idxmax()]
        best_sharpe = comparison.loc[comparison['Sharpe Ratio'].idxmax()]
        
        f.write("## Conclusions\n\n")
        f.write(f"- **Best Performing Asset:** {best_return['Symbol']} with {best_return['Total Return (%)']:.2f}% return\n")
        f.write(f"- **Best Risk-Adjusted Asset:** {best_sharpe['Symbol']} with Sharpe Ratio of {best_sharpe['Sharpe Ratio']:.2f}\n\n")
        
        # Calculate average metrics
        avg_return = comparison['Total Return (%)'].mean()
        avg_sharpe = comparison['Sharpe Ratio'].mean()
        avg_winrate = comparison['Win Rate (%)'].mean()
        profitable_assets = (comparison['Total Return (%)'] > 0).sum()
        
        f.write(f"- **Average Return:** {avg_return:.2f}%\n")
        f.write(f"- **Average Sharpe Ratio:** {avg_sharpe:.2f}\n")
        f.write(f"- **Average Win Rate:** {avg_winrate:.2f}%\n")
        f.write(f"- **Profitable Assets:** {profitable_assets} out of {len(symbols)} ({profitable_assets/len(symbols)*100:.1f}%)\n\n")
        
        if avg_return > 0 and avg_sharpe > 1:
            f.write(f"The {strategy.name} strategy performs well across most tested cryptocurrencies and could be considered for a multi-asset portfolio strategy.\n")
        elif profitable_assets > len(symbols)/2:
            f.write(f"The {strategy.name} strategy shows mixed results. Consider applying it selectively to the top performing assets like {best_return['Symbol']}.\n")
        else:
            f.write(f"The {strategy.name} strategy doesn't perform consistently across the tested cryptocurrencies. Consider refining the strategy or limiting its application.\n")
    
    logger.info(f"Multi-asset comparison report generated: {report_file}")
    
    return comparison

def main():
    """
    Main function to run backtests
    """
    import argparse
    parser = argparse.ArgumentParser(description='Crypto Backtesting System')
    parser.add_argument('--symbol', type=str, help='Cryptocurrency symbol to backtest')
    parser.add_argument('--strategy', type=str, choices=['ma', 'rsi', 'macd', 'bb', 'all'], default='all',
                        help='Strategy to backtest (ma=Moving Average, rsi=RSI, macd=MACD, bb=Bollinger Bands, all=All)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple strategies')
    parser.add_argument('--multi', action='store_true', help='Test one strategy across multiple assets')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    args = parser.parse_args()
    
    # Set base directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'historical')
    output_dir = os.path.join(base_dir, 'analysis', 'backtesting')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load list of cryptocurrencies if needed
    all_symbols = []
    if args.multi or not args.symbol:
        try:
            # Add project root to path
            sys.path.append(base_dir)
            from data.crypto_list import MAJOR_CRYPTOS
            all_symbols = [c for c in MAJOR_CRYPTOS]
            logger.info(f"Loaded {len(all_symbols)} cryptocurrencies from crypto_list.py")
        except Exception as e:
            logger.error(f"Error loading crypto list: {e}")
            # Default list of major cryptos
            all_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD']
    
    # Define strategies
    strategies = []
    if args.strategy == 'ma' or args.strategy == 'all':
        strategies.append(MovingAverageCrossStrategy(short_window=20, long_window=50))
    if args.strategy == 'rsi' or args.strategy == 'all':
        strategies.append(RSIStrategy(rsi_window=14, oversold=30, overbought=70))
    if args.strategy == 'macd' or args.strategy == 'all':
        strategies.append(MACDStrategy(fast=12, slow=26, signal=9))
    if args.strategy == 'bb' or args.strategy == 'all':
        strategies.append(BollingerBandsStrategy(window=20, num_std=2))
    
    # Run appropriate backtest
    if args.compare:
        # Compare multiple strategies on one symbol
        symbol = args.symbol if args.symbol else 'BTC-USD'
        logger.info(f"Comparing {len(strategies)} strategies on {symbol}")
        comparison = run_strategy_comparison(symbol, strategies, data_dir, output_dir, args.capital, args.commission)
        
        if comparison is not None:
            print("\nStrategy Comparison Results:")
            print(tabulate(comparison[['Strategy', 'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']], 
                          headers='keys', tablefmt='grid', floatfmt='.2f'))
    
    elif args.multi:
        # Test one strategy across multiple assets
        if len(strategies) > 0:
            strategy = strategies[0]
            symbols = all_symbols if not args.symbol else [args.symbol]
            logger.info(f"Testing {strategy.name} across {len(symbols)} cryptocurrencies")
            comparison = run_multi_asset_backtest(symbols, strategy, data_dir, output_dir, args.capital, args.commission)
            
            if comparison is not None:
                print("\nMulti-Asset Results:")
                print(tabulate(comparison[['Symbol', 'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']], 
                             headers='keys', tablefmt='grid', floatfmt='.2f'))
    
    else:
        # Run single backtest
        symbol = args.symbol if args.symbol else 'BTC-USD'
        
        for strategy in strategies:
            logger.info(f"Running backtest for {symbol} with {strategy.name}")
            report = run_backtest(symbol, strategy, data_dir, output_dir, args.capital, args.commission)
            
            if report:
                print(f"\nBacktest Results for {symbol} using {strategy.name}:")
                metrics = report['metrics']
                print(f"Total Return: {metrics['Total Return (%)']:.2f}%")
                print(f"Buy & Hold Return: {metrics['Buy & Hold Return (%)']:.2f}%")
                print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
                print(f"Max Drawdown: {metrics['Max Drawdown (%)']:.2f}%")
                print(f"Win Rate: {metrics['Win Rate (%)']:.2f}%")
                print(f"Report saved to {output_dir}")

if __name__ == "__main__":
    main()