import numpy as np
import pandas as pd
import asyncio
import aiofiles
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import orjson
from strategy_engine import StrategyRegistry, BaseStrategy
from portfolio_manager import PortfolioManager


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
    """Advanced backtesting engine with modular strategy support"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.ti = TechnicalIndicators()
        self.strategy_registry = StrategyRegistry()
        
        # Legacy support - to be removed
        self.positions = []
        self.trades = []
        self.portfolio_value = []
        self.cash = initial_capital
        self.shares = 0
    
    async def run(self, ticker: str, strategy_name: str = "buy_and_hold", 
                  start_date: str = None, end_date: str = None, 
                  comprehensive: bool = False, **strategy_params) -> Dict:
        """
        Unified method to run backtesting with clean interface
        
        Args:
            ticker: Stock ticker symbol
            strategy_name: Strategy to run or 'all' for comprehensive test
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            comprehensive: If True, run all strategies for comparison
            **strategy_params: Additional parameters for the strategy
        
        Returns:
            Dictionary with backtesting results
        """
        if comprehensive or strategy_name == "all":
            return await self._run_comprehensive(ticker, start_date, end_date)
        else:
            return await self._run_single_strategy(ticker, strategy_name, start_date, end_date, **strategy_params)
    
    async def _run_comprehensive(self, ticker: str, start_date: str = None, end_date: str = None) -> Dict:
        """Run comprehensive backtest with all strategies"""
        df = await self.load_historical_data(ticker, start_date, end_date)
        
        if df.empty:
            return {
                'ticker': ticker,
                'error': f'No historical data available for {ticker}',
                'strategies': []
            }
        
        # Calculate SPY benchmark once for all strategies
        spy_start = start_date or df.index[0].strftime('%Y-%m-%d')
        spy_end = end_date or df.index[-1].strftime('%Y-%m-%d')
        spy_benchmark = await self._calculate_spy_benchmark_with_history(spy_start, spy_end)
        
        # Calculate stock buy-and-hold benchmark
        stock_buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] if len(df) > 0 else 0
        
        # Prepare stock buy-and-hold plot data
        stock_normalized = (df['close'] / df['close'].iloc[0]) * self.initial_capital
        stock_buy_hold_plot = [
            {
                'date': date.strftime('%Y-%m-%d'),
                'value': float(value),
                'return_pct': ((value - self.initial_capital) / self.initial_capital) * 100
            }
            for date, value in stock_normalized.items()
        ]
        
        # Run all strategies async 
        strategy_tasks = [
            ('Buy and Hold', self.buy_and_hold_strategy(df.copy(), start_date, end_date, ticker)),
            ('RSI (30/70)', self.rsi_strategy(df.copy(), 30, 70, 14, start_date, end_date, ticker)),
            ('MA Crossover (20/50)', self.moving_average_crossover_strategy(df.copy(), 20, 50, start_date, end_date, ticker)),
            ('MA Crossover (50/200)', self.moving_average_crossover_strategy(df.copy(), 50, 200, start_date, end_date, ticker)),
            ('Bollinger Bands', self.bollinger_bands_strategy(df.copy(), 20, 2, start_date, end_date, ticker)),
            ('MACD', self.macd_strategy(df.copy(), 12, 26, 9, start_date, end_date, ticker)),
        ]
        
        results = []
        for name, strategy_coro in strategy_tasks:
            result = await strategy_coro
            if result:
                results.append(result)
        
        results.sort(key=lambda x: x.get('total_return', 0), reverse=True)
        
        date_adjustments = {}
        if hasattr(df, 'attrs'):
            if df.attrs.get('requested_start'):
                date_adjustments['requested_start'] = df.attrs.get('requested_start')
                date_adjustments['actual_start'] = df.attrs.get('actual_start')
            if df.attrs.get('requested_end'):
                date_adjustments['requested_end'] = df.attrs.get('requested_end')
                date_adjustments['actual_end'] = df.attrs.get('actual_end')
        
        # Prepare SPY plot data
        spy_plot = []
        if 'spy_history' in spy_benchmark and spy_benchmark['spy_history']:
            spy_plot = [
                {
                    'date': item['date'],
                    'value': item['value'],
                    'return_pct': ((item['value'] - self.initial_capital) / self.initial_capital) * 100
                }
                for item in spy_benchmark['spy_history']
            ]
        
        response = {
            'ticker': ticker.upper(),
            'period': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
            'data_points': len(df),
            'initial_capital': self.initial_capital,
            'strategies': results,
            'best_strategy': results[0] if results else None,
            'summary': f"Backtested {len(results)} strategies for {ticker.upper()} over {len(df)} trading days.",
            # Benchmark data for comparison
            'benchmarks': {
                'stock_buy_hold': {
                    'return': round(stock_buy_hold_return * 100, 2),
                    'plot_data': stock_buy_hold_plot
                },
                'spy': {
                    'return': round(spy_benchmark.get('spy_return', 0) * 100, 2),
                    'annual_return': round(spy_benchmark.get('spy_annual_return', 0) * 100, 2),
                    'plot_data': spy_plot
                }
            }
        }
        
        if date_adjustments:
            response['date_adjustments'] = date_adjustments
        
        return response
    
    async def _run_single_strategy(self, ticker: str, strategy_name: str, 
                                   start_date: str = None, end_date: str = None, 
                                   **strategy_params) -> Dict:
        """Run single strategy backtest"""
        df = await self.load_historical_data(ticker, start_date, end_date)
        
        if df.empty:
            return {
                'ticker': ticker.upper(),
                'strategy': strategy_name,
                'error': f'No historical data available for {ticker}',
                'success': False
            }
        
        strategy_functions = {
            'buy_and_hold': lambda df_copy: self.buy_and_hold_strategy(df_copy, start_date, end_date, ticker),
            'rsi': lambda df_copy: self.rsi_strategy(df_copy, 
                                                      strategy_params.get('rsi_buy', 30),
                                                      strategy_params.get('rsi_sell', 70),
                                                      strategy_params.get('rsi_window', 14),
                                                      start_date, end_date, ticker),
            'ma_crossover': lambda df_copy: self.moving_average_crossover_strategy(df_copy,
                                                                                    strategy_params.get('short_window', 20),
                                                                                    strategy_params.get('long_window', 50),
                                                                                    start_date, end_date, ticker),
            'bollinger': lambda df_copy: self.bollinger_bands_strategy(df_copy,
                                                                        strategy_params.get('window', 20),
                                                                        strategy_params.get('num_std', 2),
                                                                        start_date, end_date, ticker),
            'macd': lambda df_copy: self.macd_strategy(df_copy,
                                                        strategy_params.get('fast', 12),
                                                        strategy_params.get('slow', 26),
                                                        strategy_params.get('signal', 9),
                                                        start_date, end_date, ticker)
        }
        
        if strategy_name not in strategy_functions:
            return {
                'ticker': ticker.upper(),
                'strategy': strategy_name,
                'error': f'Unknown strategy: {strategy_name}. Available: {list(strategy_functions.keys())}',
                'success': False
            }
        
        try:
            result = await strategy_functions[strategy_name](df.copy())
            result.update({
                'ticker': ticker.upper(),
                'strategy': strategy_name,
                'success': True,
                'data_points': len(df),
                'start_date': df.index[0].strftime('%Y-%m-%d'),
                'end_date': df.index[-1].strftime('%Y-%m-%d')
            })
            
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
    
    async def run_strategy_backtest(self, data: pd.DataFrame, strategy: BaseStrategy, start_date: str = None, end_date: str = None, ticker: str = None) -> Dict[str, Any]:
        """Run backtest using a strategy instance"""
        if data.empty:
            return self._empty_backtest_result()
        
        # Reset strategy state
        strategy.reset()
        
        # Prepare data with indicators
        prepared_data = strategy.prepare_data(data)
        
        # Initialize portfolio manager
        portfolio = PortfolioManager(
            initial_capital=self.initial_capital,
            commission_rate=self.commission
        )
        
        # Generate signals
        signals = strategy.generate_signals(prepared_data)
        
        # Create a dict to map dates to signals for efficient lookup
        signal_dict = {}
        for signal in signals:
            signal_dict[signal.date] = signal
        
        # Execute trades and track portfolio value throughout the period
        executed_trades = []
        for date, row in prepared_data.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            
            # Check if there's a signal for this date
            if date_str in signal_dict:
                trade = portfolio.execute_signal(signal_dict[date_str], row['close'])
                if trade:
                    executed_trades.append(trade)
            
            # Update portfolio history for every trading day
            portfolio.update_portfolio_history(date_str, row['close'])
        
        # Calculate performance metrics
        performance = portfolio.calculate_performance_metrics()
        
        # Calculate buy-and-hold benchmark for the tested stock
        buy_hold_return = (prepared_data['close'].iloc[-1] - prepared_data['close'].iloc[0]) / prepared_data['close'].iloc[0] if len(prepared_data) > 0 else 0
        
        # Calculate SPY benchmark for the same period
        spy_start = start_date or prepared_data.index[0].strftime('%Y-%m-%d')
        spy_end = end_date or prepared_data.index[-1].strftime('%Y-%m-%d')
        spy_benchmark = await self._calculate_spy_benchmark_with_history(spy_start, spy_end)
        
        # Convert trades to legacy format for compatibility
        legacy_signals = [trade.to_dict() for trade in executed_trades]
        
        # Create trade history with ticker information
        trade_history = []
        for trade in executed_trades:
            trade_entry = {
                'date': trade.date,
                'ticker': ticker if ticker else 'N/A',
                'action': trade.action,
                'shares': trade.shares,
                'price': round(trade.price, 2),
                'commission': round(trade.commission, 2),
                'gross_amount': round(trade.gross_amount, 2),
                'net_amount': round(trade.net_amount, 2),
                'portfolio_value': round(trade.metadata.get('portfolio_value', 0), 2) if trade.metadata else 0
            }
            # Add any additional metadata from the trade
            if trade.metadata:
                for key, value in trade.metadata.items():
                    if key not in trade_entry:
                        trade_entry[key] = value
            trade_history.append(trade_entry)
        
        # Prepare plotting data
        plot_data = self._prepare_plot_data(portfolio.portfolio_history, prepared_data, spy_benchmark)
        
        result = {
            'strategy_name': strategy.name,
            'period': f"{prepared_data.index[0].strftime('%Y-%m-%d')} to {prepared_data.index[-1].strftime('%Y-%m-%d')}",
            'total_return': round(performance.get('total_return', 0) * 100, 2),
            'annual_return': round(performance.get('annual_return', 0) * 100, 2),
            'buy_hold_return': round(buy_hold_return * 100, 2),
            'excess_return': round((performance.get('total_return', 0) - buy_hold_return) * 100, 2),
            'max_drawdown': round(performance.get('max_drawdown', 0) * 100, 2),
            'volatility': round(performance.get('volatility', 0) * 100, 2),
            'sharpe_ratio': round(performance.get('sharpe_ratio', 0), 2),
            'total_trades': performance.get('total_trades_completed', 0),
            'win_rate': round(performance.get('win_rate', 0) * 100, 2),
            'profit_factor': round(performance.get('profit_factor', 0), 2),
            'initial_capital': self.initial_capital,
            'final_portfolio_value': round(performance.get('final_portfolio_value', self.initial_capital), 2),
            'signals': legacy_signals[:50],
            'total_signals': len(legacy_signals),
            'trade_history': trade_history,  # New field with complete trade history
            'summary': self._generate_summary(strategy.name, performance.get('total_return', 0), buy_hold_return, performance.get('win_rate', 0), performance.get('total_trades_completed', 0)),
            # SPY Benchmark
            'spy_benchmark': {
                'spy_return': round(spy_benchmark.get('spy_return', 0) * 100, 2),
                'spy_annual_return': round(spy_benchmark.get('spy_annual_return', 0) * 100, 2),
                'spy_period': spy_benchmark.get('spy_period', 'N/A'),
                'spy_data_points': spy_benchmark.get('spy_data_points', 0),
                'vs_spy': round((performance.get('total_return', 0) - spy_benchmark.get('spy_return', 0)) * 100, 2)
            },
            # Plotting data for visualization
            'plot_data': plot_data
        }
        
        # Add error information if SPY benchmark failed
        if 'error' in spy_benchmark:
            result['spy_benchmark']['error'] = spy_benchmark['error']
        
        return result
    
    # Legacy strategy methods for backward compatibility
    async def rsi_strategy(self, df: pd.DataFrame, rsi_buy: float = 30, rsi_sell: float = 70, rsi_window: int = 14, start_date: str = None, end_date: str = None, ticker: str = None) -> Dict:
        """RSI-based trading strategy (legacy method)"""
        from strategy_engine import RSIStrategy
        
        parameters = {
            'rsi_buy_threshold': rsi_buy,
            'rsi_sell_threshold': rsi_sell,
            'rsi_window': rsi_window
        }
        
        strategy = RSIStrategy(parameters)
        return await self.run_strategy_backtest(df, strategy, start_date, end_date, ticker)
    
    async def moving_average_crossover_strategy(self, df: pd.DataFrame, short_window: int = 20, long_window: int = 50, start_date: str = None, end_date: str = None, ticker: str = None) -> Dict:
        """Moving Average Crossover Strategy (legacy method)"""
        from strategy_engine import MovingAverageCrossoverStrategy
        
        parameters = {
            'short_window': short_window,
            'long_window': long_window
        }
        
        strategy = MovingAverageCrossoverStrategy(parameters)
        return await self.run_strategy_backtest(df, strategy, start_date, end_date, ticker)
    
    async def bollinger_bands_strategy(self, df: pd.DataFrame, window: int = 20, num_std: float = 2, start_date: str = None, end_date: str = None, ticker: str = None) -> Dict:
        """Bollinger Bands mean reversion strategy (legacy method)"""
        from strategy_engine import BollingerBandsStrategy
        
        parameters = {
            'window': window,
            'num_std': num_std
        }
        
        strategy = BollingerBandsStrategy(parameters)
        return await self.run_strategy_backtest(df, strategy, start_date, end_date, ticker)
    
    async def macd_strategy(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, start_date: str = None, end_date: str = None, ticker: str = None) -> Dict:
        """MACD crossover strategy (legacy method)"""
        from strategy_engine import MACDStrategy
        
        parameters = {
            'fast': fast,
            'slow': slow,
            'signal': signal
        }
        
        strategy = MACDStrategy(parameters)
        return await self.run_strategy_backtest(df, strategy, start_date, end_date, ticker)
    
    async def buy_and_hold_strategy(self, df: pd.DataFrame, start_date: str = None, end_date: str = None, ticker: str = None) -> Dict:
        """Simple buy and hold strategy (legacy method)"""
        from strategy_engine import BuyAndHoldStrategy
        
        strategy = BuyAndHoldStrategy()
        return await self.run_strategy_backtest(df, strategy, start_date, end_date, ticker)
    
    def _reset_portfolio(self):
        """Reset portfolio to initial state (legacy method)"""
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
    
    def _calculate_performance_metrics(self, df: pd.DataFrame, signals: List[Dict], strategy_name: str) -> Dict:
        """Calculate comprehensive performance metrics (legacy method)"""
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
    
    async def _calculate_spy_benchmark(self, start_date: str = None, end_date: str = None) -> Dict[str, float]:
        """Calculate SPY buy-and-hold benchmark for the same period"""
        try:
            spy_data = await self.load_historical_data("SPY", start_date, end_date)

            if spy_data.empty:
                return {
                    'spy_return': 0.0,
                    'spy_annual_return': 0.0,
                    'error': 'SPY data not available'
                }
            
            # Calculate SPY buy-and-hold return
            spy_return = (spy_data['close'].iloc[-1] - spy_data['close'].iloc[0]) / spy_data['close'].iloc[0]
            
            # Calculate annualized return
            days = len(spy_data)
            spy_annual_return = (spy_data['close'].iloc[-1] / spy_data['close'].iloc[0]) ** (252 / days) - 1 if days > 0 else 0
            
            return {
                'spy_return': spy_return,
                'spy_annual_return': spy_annual_return,
                'spy_period': f"{spy_data.index[0].strftime('%Y-%m-%d')} to {spy_data.index[-1].strftime('%Y-%m-%d')}",
                'spy_data_points': len(spy_data)
            }
        except Exception as e:
            return {
                'spy_return': 0.0,
                'spy_annual_return': 0.0,
                'error': f'Failed to load SPY benchmark: {str(e)}'
            }
    
    async def _calculate_spy_benchmark_with_history(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Calculate SPY buy-and-hold benchmark with full price history for plotting"""
        try:
            spy_data = await self.load_historical_data("SPY", start_date, end_date)

            if spy_data.empty:
                return {
                    'spy_return': 0.0,
                    'spy_annual_return': 0.0,
                    'spy_history': [],
                    'error': 'SPY data not available'
                }
            
            # Calculate SPY buy-and-hold return
            spy_return = (spy_data['close'].iloc[-1] - spy_data['close'].iloc[0]) / spy_data['close'].iloc[0]
            
            # Calculate annualized return
            days = len(spy_data)
            spy_annual_return = (spy_data['close'].iloc[-1] / spy_data['close'].iloc[0]) ** (252 / days) - 1 if days > 0 else 0
            
            # Calculate normalized SPY prices (starting from initial capital)
            spy_normalized = (spy_data['close'] / spy_data['close'].iloc[0]) * self.initial_capital
            
            # Prepare SPY history for plotting
            spy_history = [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'value': float(value)
                }
                for date, value in spy_normalized.items()
            ]
            
            return {
                'spy_return': spy_return,
                'spy_annual_return': spy_annual_return,
                'spy_period': f"{spy_data.index[0].strftime('%Y-%m-%d')} to {spy_data.index[-1].strftime('%Y-%m-%d')}",
                'spy_data_points': len(spy_data),
                'spy_history': spy_history
            }
        except Exception as e:
            return {
                'spy_return': 0.0,
                'spy_annual_return': 0.0,
                'spy_history': [],
                'error': f'Failed to load SPY benchmark: {str(e)}'
            }
    
    def _prepare_plot_data(self, portfolio_history: List[Dict], stock_data: pd.DataFrame, spy_benchmark: Dict) -> Dict[str, Any]:
        """Prepare data for plotting portfolio performance vs benchmarks"""
        
        # Strategy portfolio values over time
        strategy_values = [
            {
                'date': item['date'],
                'value': item['portfolio_value'],
                'return_pct': ((item['portfolio_value'] - self.initial_capital) / self.initial_capital) * 100
            }
            for item in portfolio_history
        ]
        
        # Stock buy-and-hold benchmark (normalized to initial capital)
        stock_normalized = (stock_data['close'] / stock_data['close'].iloc[0]) * self.initial_capital
        stock_buy_hold = [
            {
                'date': date.strftime('%Y-%m-%d'),
                'value': float(value),
                'return_pct': ((value - self.initial_capital) / self.initial_capital) * 100
            }
            for date, value in stock_normalized.items()
        ]
        
        # SPY benchmark data
        spy_values = []
        if 'spy_history' in spy_benchmark and spy_benchmark['spy_history']:
            spy_values = [
                {
                    'date': item['date'],
                    'value': item['value'],
                    'return_pct': ((item['value'] - self.initial_capital) / self.initial_capital) * 100
                }
                for item in spy_benchmark['spy_history']
            ]
        
        return {
            'strategy': strategy_values,
            'stock_buy_hold': stock_buy_hold,
            'spy_benchmark': spy_values,
            'initial_capital': self.initial_capital
        }


