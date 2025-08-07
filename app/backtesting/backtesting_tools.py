import numpy as np
import pandas as pd
import json
import asyncio
import aiofiles
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import orjson


class TechnicalIndicators:
    """Calculate various technical indicators for backtesting"""
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def ema(prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return {'upper': upper, 'middle': sma, 'lower': lower}
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return {'%K': k_percent, '%D': d_percent}


class BacktestingEngine:
    """Advanced backtesting engine with multiple strategies"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions = []
        self.trades = []
        self.portfolio_value = []
        self.cash = initial_capital
        self.shares = 0
        self.ti = TechnicalIndicators()
    
    def _find_closest_date(self, df: pd.DataFrame, target_date: pd.Timestamp, direction: str = 'forward') -> Optional[pd.Timestamp]:
        """
        Find the closest available date in the dataframe to the target date.
        
        Args:
            df: DataFrame with datetime index
            target_date: Target date to find
            direction: 'forward' to find next available date, 'backward' for previous
        
        Returns:
            Closest available date or None if no suitable date found
        """
        if df.empty:
            return None
            
        if target_date in df.index:
            return target_date
        
        if direction == 'forward':
            # Find the first date greater than or equal to target
            future_dates = df.index[df.index >= target_date]
            if len(future_dates) > 0:
                return future_dates[0]
            else:
                # If no future dates, return the last available date
                print(f"Warning: No dates available after {target_date.strftime('%Y-%m-%d')}. Using last available date.")
                return df.index[-1]
        else:  # backward
            # Find the last date less than or equal to target
            past_dates = df.index[df.index <= target_date]
            if len(past_dates) > 0:
                return past_dates[-1]
            else:
                # If no past dates, return the first available date
                print(f"Warning: No dates available before {target_date.strftime('%Y-%m-%d')}. Using first available date.")
                return df.index[0]
    
    async def load_historical_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load historical price data for backtesting with robust date handling"""
        # Try different possible file paths
        file_paths = [
            f"../json/historical-price/adj/{ticker}.json",
        ]
        
        for file_path in file_paths:
            try:
                async with aiofiles.open(file_path, 'rb') as f:
                    content = await f.read()
                    data = orjson.loads(content)
                
                if not data or len(data) == 0:
                    continue
                
                df = pd.DataFrame(data)
                
                # Handle different column name formats
                column_mapping = {
                    'adjOpen': 'open',
                    'adjHigh': 'high', 
                    'adjLow': 'low',
                    'adjClose': 'close'
                }
                
                # Rename adjusted columns if they exist
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns:
                        df[new_col] = df[old_col]
                
                # Validate required columns exist
                required_cols = ['date', 'close']  # Only require date and close as minimum
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    continue
                
                # Fill missing OHLV columns with close price if needed
                if 'open' not in df.columns:
                    df['open'] = df['close']
                if 'high' not in df.columns:
                    df['high'] = df['close']
                if 'low' not in df.columns:
                    df['low'] = df['close']
                if 'volume' not in df.columns:
                    df['volume'] = 0
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                
                # Remove rows with missing price data
                df = df.dropna(subset=['close'])
                
                if df.empty:
                    continue
                
                # Enhanced date range filtering with closest date selection
                actual_start = None
                actual_end = None
                
                if start_date:
                    start_date_pd = pd.to_datetime(start_date)
                    actual_start = self._find_closest_date(df, start_date_pd, 'forward')
                    if actual_start:
                        df = df[df.index >= actual_start]
                        if actual_start != start_date_pd:
                            print(f"Start date adjusted from {start_date} to {actual_start.strftime('%Y-%m-%d')}")
                
                if end_date:
                    end_date_pd = pd.to_datetime(end_date)
                    actual_end = self._find_closest_date(df, end_date_pd, 'backward')
                    if actual_end:
                        df = df[df.index <= actual_end]
                        if actual_end != end_date_pd:
                            print(f"End date adjusted from {end_date} to {actual_end.strftime('%Y-%m-%d')}")
                
                if df.empty:
                    print(f"Warning: No data available in the adjusted date range for {ticker}")
                    return pd.DataFrame()
                
                # Add metadata about date adjustments
                df.attrs['requested_start'] = start_date
                df.attrs['requested_end'] = end_date
                df.attrs['actual_start'] = actual_start.strftime('%Y-%m-%d') if actual_start else None
                df.attrs['actual_end'] = actual_end.strftime('%Y-%m-%d') if actual_end else None
                    
                print(f"Successfully loaded {len(df)} data points for {ticker}")
                if actual_start or actual_end:
                    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                
                return df
                
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error loading data from {file_path}: {str(e)}")
                continue
        
        print(f"Error: No historical data file found for {ticker}")
        return pd.DataFrame()
    
    def rsi_strategy(self, df: pd.DataFrame, rsi_buy: float = 30, rsi_sell: float = 70, rsi_window: int = 14) -> Dict:
        """RSI-based trading strategy"""
        if df.empty:
            return self._empty_backtest_result()
        
        df = df.copy()
        df['rsi'] = self.ti.rsi(df['close'], window=rsi_window)
        
        # Reset portfolio for this backtest
        self._reset_portfolio()
        
        signals = []
        portfolio_values = []
        
        for i, (date, row) in enumerate(df.iterrows()):
            if pd.isna(row['rsi']) or i < rsi_window:
                portfolio_values.append(self.cash + (self.shares * row['close']))
                continue
                
            current_value = self.cash + (self.shares * row['close'])
            
            # Buy signal: RSI below buy threshold and we don't have shares
            if row['rsi'] < rsi_buy and self.shares == 0:
                shares_to_buy = int((self.cash * 0.95) / row['close'])  # Use 95% of cash
                if shares_to_buy > 0:
                    cost = shares_to_buy * row['close'] * (1 + self.commission)
                    if cost <= self.cash:
                        self.cash -= cost
                        self.shares += shares_to_buy
                        signals.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'action': 'BUY',
                            'price': row['close'],
                            'shares': shares_to_buy,
                            'rsi': row['rsi'],
                            'portfolio_value': current_value
                        })
            
            # Sell signal: RSI above sell threshold and we have shares
            elif row['rsi'] > rsi_sell and self.shares > 0:
                proceeds = self.shares * row['close'] * (1 - self.commission)
                self.cash += proceeds
                signals.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': 'SELL',
                    'price': row['close'],
                    'shares': self.shares,
                    'rsi': row['rsi'],
                    'portfolio_value': current_value
                })
                self.shares = 0
            
            portfolio_values.append(self.cash + (self.shares * row['close']))
        
        # Calculate performance metrics
        df['portfolio_value'] = portfolio_values
        return self._calculate_performance_metrics(df, signals, 'RSI Strategy')
    
    def moving_average_crossover_strategy(self, df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> Dict:
        """Moving Average Crossover Strategy"""
        if df.empty:
            return self._empty_backtest_result()
            
        df = df.copy()
        df['sma_short'] = self.ti.sma(df['close'], short_window)
        df['sma_long'] = self.ti.sma(df['close'], long_window)
        
        self._reset_portfolio()
        
        signals = []
        portfolio_values = []
        
        for i, (date, row) in enumerate(df.iterrows()):
            if pd.isna(row['sma_short']) or pd.isna(row['sma_long']) or i < long_window:
                portfolio_values.append(self.cash + (self.shares * row['close']))
                continue
                
            current_value = self.cash + (self.shares * row['close'])
            prev_row = df.iloc[i-1] if i > 0 else row
            
            # Buy signal: short MA crosses above long MA
            if (row['sma_short'] > row['sma_long'] and 
                prev_row['sma_short'] <= prev_row['sma_long'] and 
                self.shares == 0):
                
                shares_to_buy = int((self.cash * 0.95) / row['close'])
                if shares_to_buy > 0:
                    cost = shares_to_buy * row['close'] * (1 + self.commission)
                    if cost <= self.cash:
                        self.cash -= cost
                        self.shares += shares_to_buy
                        signals.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'action': 'BUY',
                            'price': row['close'],
                            'shares': shares_to_buy,
                            'sma_short': row['sma_short'],
                            'sma_long': row['sma_long'],
                            'portfolio_value': current_value
                        })
            
            # Sell signal: short MA crosses below long MA
            elif (row['sma_short'] < row['sma_long'] and 
                  prev_row['sma_short'] >= prev_row['sma_long'] and 
                  self.shares > 0):
                
                proceeds = self.shares * row['close'] * (1 - self.commission)
                self.cash += proceeds
                signals.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': 'SELL',
                    'price': row['close'],
                    'shares': self.shares,
                    'sma_short': row['sma_short'],
                    'sma_long': row['sma_long'],
                    'portfolio_value': current_value
                })
                self.shares = 0
            
            portfolio_values.append(self.cash + (self.shares * row['close']))
        
        df['portfolio_value'] = portfolio_values
        return self._calculate_performance_metrics(df, signals, f'MA Crossover ({short_window}/{long_window})')
    
    def bollinger_bands_strategy(self, df: pd.DataFrame, window: int = 20, num_std: float = 2) -> Dict:
        """Bollinger Bands mean reversion strategy"""
        if df.empty:
            return self._empty_backtest_result()
            
        df = df.copy()
        bb = self.ti.bollinger_bands(df['close'], window, num_std)
        df['bb_upper'] = bb['upper']
        df['bb_middle'] = bb['middle']
        df['bb_lower'] = bb['lower']
        
        self._reset_portfolio()
        
        signals = []
        portfolio_values = []
        
        for i, (date, row) in enumerate(df.iterrows()):
            if pd.isna(row['bb_upper']) or i < window:
                portfolio_values.append(self.cash + (self.shares * row['close']))
                continue
                
            current_value = self.cash + (self.shares * row['close'])
            
            # Buy signal: price touches lower band
            if row['close'] <= row['bb_lower'] and self.shares == 0:
                shares_to_buy = int((self.cash * 0.95) / row['close'])
                if shares_to_buy > 0:
                    cost = shares_to_buy * row['close'] * (1 + self.commission)
                    if cost <= self.cash:
                        self.cash -= cost
                        self.shares += shares_to_buy
                        signals.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'action': 'BUY',
                            'price': row['close'],
                            'shares': shares_to_buy,
                            'bb_position': 'Lower Band',
                            'portfolio_value': current_value
                        })
            
            # Sell signal: price touches upper band
            elif row['close'] >= row['bb_upper'] and self.shares > 0:
                proceeds = self.shares * row['close'] * (1 - self.commission)
                self.cash += proceeds
                signals.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': 'SELL',
                    'price': row['close'],
                    'shares': self.shares,
                    'bb_position': 'Upper Band',
                    'portfolio_value': current_value
                })
                self.shares = 0
            
            portfolio_values.append(self.cash + (self.shares * row['close']))
        
        df['portfolio_value'] = portfolio_values
        return self._calculate_performance_metrics(df, signals, f'Bollinger Bands ({window}, {num_std})')
    
    def macd_strategy(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD crossover strategy"""
        if df.empty:
            return self._empty_backtest_result()
            
        df = df.copy()
        macd_data = self.ti.macd(df['close'], fast, slow, signal)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        self._reset_portfolio()
        
        signals = []
        portfolio_values = []
        
        for i, (date, row) in enumerate(df.iterrows()):
            if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or i < slow + signal:
                portfolio_values.append(self.cash + (self.shares * row['close']))
                continue
                
            current_value = self.cash + (self.shares * row['close'])
            prev_row = df.iloc[i-1] if i > 0 else row
            
            # Buy signal: MACD crosses above signal line
            if (row['macd'] > row['macd_signal'] and 
                prev_row['macd'] <= prev_row['macd_signal'] and 
                self.shares == 0):
                
                shares_to_buy = int((self.cash * 0.95) / row['close'])
                if shares_to_buy > 0:
                    cost = shares_to_buy * row['close'] * (1 + self.commission)
                    if cost <= self.cash:
                        self.cash -= cost
                        self.shares += shares_to_buy
                        signals.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'action': 'BUY',
                            'price': row['close'],
                            'shares': shares_to_buy,
                            'macd': row['macd'],
                            'macd_signal': row['macd_signal'],
                            'portfolio_value': current_value
                        })
            
            # Sell signal: MACD crosses below signal line
            elif (row['macd'] < row['macd_signal'] and 
                  prev_row['macd'] >= prev_row['macd_signal'] and 
                  self.shares > 0):
                
                proceeds = self.shares * row['close'] * (1 - self.commission)
                self.cash += proceeds
                signals.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': 'SELL',
                    'price': row['close'],
                    'shares': self.shares,
                    'macd': row['macd'],
                    'macd_signal': row['macd_signal'],
                    'portfolio_value': current_value
                })
                self.shares = 0
            
            portfolio_values.append(self.cash + (self.shares * row['close']))
        
        df['portfolio_value'] = portfolio_values
        return self._calculate_performance_metrics(df, signals, f'MACD ({fast}, {slow}, {signal})')
    
    def _reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.shares = 0
    
    def _empty_backtest_result(self) -> Dict:
        """Return empty backtest result when no data available"""
        return {
            'strategy_name': 'No Data Available',
            'total_return': 0,
            'annual_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'final_portfolio_value': self.initial_capital,
            'signals': [],
            'summary': 'No historical data available for backtesting'
        }
    
    def buy_and_hold_strategy(self, df: pd.DataFrame) -> Dict:
        """Simple buy and hold strategy - buy at start, hold until end"""
        if df.empty:
            return self._empty_backtest_result()
            
        df = df.copy()
        self._reset_portfolio()
        
        signals = []
        portfolio_values = []
        
        # Buy on the first day
        first_price = df['close'].iloc[0]
        shares_to_buy = int((self.initial_capital * 0.95) / first_price)  # Use 95% of capital
        
        if shares_to_buy > 0:
            cost = shares_to_buy * first_price * (1 + self.commission)
            if cost <= self.cash:
                self.cash -= cost
                self.shares = shares_to_buy
                signals.append({
                    'date': df.index[0].strftime('%Y-%m-%d'),
                    'action': 'BUY',
                    'price': first_price,
                    'shares': shares_to_buy,
                    'portfolio_value': self.initial_capital
                })
        
        # Calculate portfolio value for each day (no more trades)
        for i, (date, row) in enumerate(df.iterrows()):
            current_portfolio_value = self.cash + (self.shares * row['close'])
            portfolio_values.append(current_portfolio_value)
        
        # Sell on the last day for final value calculation
        if self.shares > 0:
            last_price = df['close'].iloc[-1]
            proceeds = self.shares * last_price * (1 - self.commission)
            signals.append({
                'date': df.index[-1].strftime('%Y-%m-%d'),
                'action': 'SELL',
                'price': last_price,
                'shares': self.shares,
                'portfolio_value': portfolio_values[-1]
            })
        
        df['portfolio_value'] = portfolio_values
        return self._calculate_performance_metrics(df, signals, 'Buy and Hold')

    def _calculate_performance_metrics(self, df: pd.DataFrame, signals: List[Dict], strategy_name: str) -> Dict:
        """Calculate comprehensive performance metrics"""
        if df.empty or 'portfolio_value' not in df.columns:
            return self._empty_backtest_result()
        
        portfolio_values = df['portfolio_value'].values
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annual_return = (portfolio_values[-1] / self.initial_capital) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        
        # Trade analysis
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        sell_signals = [s for s in signals if s['action'] == 'SELL']
        
        total_trades = len(buy_signals)
        winning_trades = 0
        total_profit = 0
        total_loss = 0
        
        for i in range(min(len(buy_signals), len(sell_signals))):
            profit = (sell_signals[i]['price'] - buy_signals[i]['price']) * buy_signals[i]['shares']
            if profit > 0:
                winning_trades += 1
                total_profit += profit
            else:
                total_loss += abs(profit)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        
        # Buy and hold comparison
        buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] if len(df) > 0 else 0
        
        # Include date adjustment info if available
        date_info = {}
        if hasattr(df, 'attrs'):
            if df.attrs.get('requested_start') != df.attrs.get('actual_start'):
                date_info['start_date_adjusted'] = True
                date_info['requested_start'] = df.attrs.get('requested_start')
                date_info['actual_start'] = df.attrs.get('actual_start')
            if df.attrs.get('requested_end') != df.attrs.get('actual_end'):
                date_info['end_date_adjusted'] = True
                date_info['requested_end'] = df.attrs.get('requested_end')
                date_info['actual_end'] = df.attrs.get('actual_end')
        
        result = {
            'strategy_name': strategy_name,
            'period': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
            'total_return': round(total_return * 100, 2),
            'annual_return': round(annual_return * 100, 2),
            'buy_hold_return': round(buy_hold_return * 100, 2),
            'excess_return': round((total_return - buy_hold_return) * 100, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'volatility': round(volatility * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'total_trades': total_trades,
            'win_rate': round(win_rate * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'initial_capital': self.initial_capital,
            'final_portfolio_value': round(portfolio_values[-1], 2),
            'signals': signals[:50],  # Limit signals for response size
            'total_signals': len(signals),
            'summary': self._generate_summary(strategy_name, total_return, buy_hold_return, win_rate, total_trades)
        }
        
        if date_info:
            result['date_adjustments'] = date_info
        
        return result
    
    def _calculate_max_drawdown(self, portfolio_values: np.array) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return abs(np.min(drawdown))
    
    def _generate_summary(self, strategy_name: str, total_return: float, buy_hold_return: float, win_rate: float, total_trades: int) -> str:
        """Generate human-readable summary"""
        performance = "outperformed" if total_return > buy_hold_return else "underperformed"
        return (f"{strategy_name} generated a {total_return*100:.2f}% total return with {total_trades} trades "
                f"and a {win_rate*100:.1f}% win rate. The strategy {performance} buy-and-hold by "
                f"{abs((total_return - buy_hold_return)*100):.2f} percentage points.")


async def run_comprehensive_backtest(ticker: str, start_date: str = None, end_date: str = None, 
                                   initial_capital: float = 100000) -> Dict:
    """Run multiple backtesting strategies and compare results"""
    engine = BacktestingEngine(initial_capital=initial_capital)
    df = await engine.load_historical_data(ticker, start_date, end_date)
    
    if df.empty:
        return {
            'ticker': ticker,
            'error': f'No historical data available for {ticker}',
            'strategies': []
        }
    
    strategies = [
        ('Buy and Hold', engine.buy_and_hold_strategy(df.copy())),
        ('RSI (30/70)', engine.rsi_strategy(df.copy(), 30, 70)),
        ('MA Crossover (20/50)', engine.moving_average_crossover_strategy(df.copy(), 20, 50)),
        ('MA Crossover (50/200)', engine.moving_average_crossover_strategy(df.copy(), 50, 200)),
        ('Bollinger Bands', engine.bollinger_bands_strategy(df.copy())),
        ('MACD', engine.macd_strategy(df.copy())),
    ]
    
    results = []
    for name, result in strategies:
        if result:
            results.append(result)
    
    # Sort by total return
    results.sort(key=lambda x: x.get('total_return', 0), reverse=True)
    
    # Add date adjustment info to main result
    date_adjustments = {}
    if hasattr(df, 'attrs'):
        if df.attrs.get('requested_start'):
            date_adjustments['requested_start'] = df.attrs.get('requested_start')
            date_adjustments['actual_start'] = df.attrs.get('actual_start')
        if df.attrs.get('requested_end'):
            date_adjustments['requested_end'] = df.attrs.get('requested_end')
            date_adjustments['actual_end'] = df.attrs.get('actual_end')
    
    response = {
        'ticker': ticker.upper(),
        'period': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
        'data_points': len(df),
        'initial_capital': initial_capital,
        'strategies': results,
        'best_strategy': results[0] if results else None,
        'summary': f"Backtested {len(results)} strategies for {ticker.upper()} over {len(df)} trading days."
    }
    
    if date_adjustments:
        response['date_adjustments'] = date_adjustments
    
    return response


async def run_single_strategy_backtest(ticker: str, strategy_name: str = "buy_and_hold", 
                                     start_date: str = None, end_date: str = None, 
                                     initial_capital: float = 100000, **strategy_params) -> Dict:
    """Run a single backtesting strategy with detailed results for AI agent"""
    engine = BacktestingEngine(initial_capital=initial_capital)
    df = await engine.load_historical_data(ticker, start_date, end_date)
    
    if df.empty:
        return {
            'ticker': ticker.upper(),
            'strategy': strategy_name,
            'error': f'No historical data available for {ticker}',
            'success': False
        }
    
    # Strategy mapping
    strategy_functions = {
        'buy_and_hold': engine.buy_and_hold_strategy,
        'rsi': lambda df_copy: engine.rsi_strategy(df_copy, 
                                                  strategy_params.get('rsi_buy', 30),
                                                  strategy_params.get('rsi_sell', 70),
                                                  strategy_params.get('rsi_window', 14)),
        'ma_crossover': lambda df_copy: engine.moving_average_crossover_strategy(df_copy,
                                                                                strategy_params.get('short_window', 20),
                                                                                strategy_params.get('long_window', 50)),
        'bollinger': lambda df_copy: engine.bollinger_bands_strategy(df_copy,
                                                                    strategy_params.get('window', 20),
                                                                    strategy_params.get('num_std', 2)),
        'macd': lambda df_copy: engine.macd_strategy(df_copy,
                                                    strategy_params.get('fast', 12),
                                                    strategy_params.get('slow', 26),
                                                    strategy_params.get('signal', 9))
    }
    
    if strategy_name not in strategy_functions:
        return {
            'ticker': ticker.upper(),
            'strategy': strategy_name,
            'error': f'Unknown strategy: {strategy_name}. Available: {list(strategy_functions.keys())}',
            'success': False
        }
    
    try:
        result = strategy_functions[strategy_name](df.copy())
        result.update({
            'ticker': ticker.upper(),
            'strategy': strategy_name,
            'success': True,
            'data_points': len(df),
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d')
        })
        
        # Add date adjustment info if available
        if hasattr(df, 'attrs'):
            date_adjustments = {}
            if df.attrs.get('requested_start'):
                date_adjustments['requested_start'] = df.attrs.get('requested_start')
                date_adjustments['actual_start'] = df.attrs.get('actual_start')
            if df.attrs.get('requested_end'):
                date_adjustments['requested_end'] = df.attrs.get('requested_end')
                date_adjustments['actual_end'] = df.attrs.get('actual_end')
            if date_adjustments:
                result['date_adjustments'] = date_adjustments
        
        return result
    except Exception as e:
        return {
            'ticker': ticker.upper(),
            'strategy': strategy_name,
            'error': f'Backtesting failed: {str(e)}',
            'success': False
        }


