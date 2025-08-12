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
    
    async def run(self, ticker=None, tickers: List[str] = None, strategy_name: str = "buy_and_hold", 
                  start_date: str = None, end_date: str = None, 
                  comprehensive: bool = False, **strategy_params) -> Dict:
        """
        Unified method to run backtesting with clean interface
        
        Args:
            tickers: List of ticker symbols for multi-ticker backtesting
            strategy_name: Strategy to run or 'all' for comprehensive test
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            comprehensive: If True, run all strategies for comparison
            **strategy_params: Additional parameters for the strategy
        
        Returns:
            Dictionary with backtesting results
        """
        ticker_list = tickers
        
        return await self._run_strategy(ticker_list, strategy_name, start_date, end_date, **strategy_params)
    
    
    async def _run_strategy(self, ticker_list: List[str], strategy_name: str, 
                                   start_date: str = None, end_date: str = None, 
                                   **strategy_params) -> Dict:
        """Run single strategy backtest - supports both single and multi-ticker"""
        
        if len(ticker_list) == 1:
            # Single ticker mode (legacy)
            ticker = ticker_list[0]
            df = await self.load_historical_data(ticker, start_date, end_date)
            
            if df.empty:
                return {
                    'ticker': ticker.upper(),
                    'strategy': strategy_name,
                    'error': f'No historical data available for {ticker}',
                    'success': False
                }
        else:
            # Multi-ticker mode
            data_dict = await self.load_historical_data_multi_ticker(ticker_list, start_date, end_date)
            
            if not data_dict:
                return {
                    'tickers': [t.upper() for t in ticker_list],
                    'strategy': strategy_name,
                    'error': f'No historical data available for any of the tickers: {ticker_list}',
                    'success': False
                }
        
        if len(ticker_list) == 1:
            # Single ticker execution (legacy path)
            ticker = ticker_list[0]
            
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
        
        else:
            # Multi-ticker execution 
            try:
                # Import strategy classes dynamically based on strategy name
                strategy_map = {
                    'buy_and_hold': 'BuyAndHoldStrategy',
                    'rsi': 'RSIStrategy',
                    'ma_crossover': 'MovingAverageCrossoverStrategy',
                    'bollinger': 'BollingerBandsStrategy',
                    'macd': 'MACDStrategy'
                }
                
                if strategy_name not in strategy_map:
                    return {
                        'tickers': [t.upper() for t in ticker_list],
                        'strategy': strategy_name,
                        'error': f'Unknown strategy: {strategy_name}. Available: {list(strategy_map.keys())}',
                        'success': False
                    }
                
                # Import and create strategy instance
                from strategy_engine import BuyAndHoldStrategy, RSIStrategy, MovingAverageCrossoverStrategy, BollingerBandsStrategy, MACDStrategy
                
                strategy_class = {
                    'buy_and_hold': BuyAndHoldStrategy,
                    'rsi': RSIStrategy,
                    'ma_crossover': MovingAverageCrossoverStrategy,
                    'bollinger': BollingerBandsStrategy,
                    'macd': MACDStrategy
                }[strategy_name]
                
                # Create parameters based on strategy
                parameters = {}
                if strategy_name == 'rsi':
                    parameters = {
                        'rsi_buy_threshold': strategy_params.get('rsi_buy', 30),
                        'rsi_sell_threshold': strategy_params.get('rsi_sell', 70),
                        'rsi_window': strategy_params.get('rsi_window', 14)
                    }
                elif strategy_name == 'ma_crossover':
                    parameters = {
                        'short_window': strategy_params.get('short_window', 20),
                        'long_window': strategy_params.get('long_window', 50)
                    }
                elif strategy_name == 'bollinger':
                    parameters = {
                        'window': strategy_params.get('window', 20),
                        'num_std': strategy_params.get('num_std', 2)
                    }
                elif strategy_name == 'macd':
                    parameters = {
                        'fast': strategy_params.get('fast', 12),
                        'slow': strategy_params.get('slow', 26),
                        'signal': strategy_params.get('signal', 9)
                    }
                
                strategy = strategy_class(parameters) if parameters else strategy_class()
                
                # Run multi-ticker backtest
                result = await self.run_strategy_backtest_multi_ticker(data_dict, strategy, start_date, end_date)
                result.update({
                    'strategy': strategy_name,
                    'success': True,
                    'data_points': sum(len(df) for df in data_dict.values())
                })
                
                return result
                
            except Exception as e:
                return {
                    'tickers': [t.upper() for t in ticker_list],
                    'strategy': strategy_name,
                    'error': f'Multi-ticker backtesting failed: {str(e)}',
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
    
    async def load_historical_data_multi_ticker(self, tickers: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Load historical data for multiple tickers"""
        data_dict = {}
        
        for ticker in tickers:
            try:
                df = await self.load_historical_data(ticker, start_date, end_date)
                if not df.empty:
                    data_dict[ticker] = df
                else:
                    print(f"Warning: No data available for {ticker}")
            except Exception as e:
                print(f"Error loading data for {ticker}: {str(e)}")
        
        return data_dict
    
    async def run_strategy_backtest_multi_ticker(self, data_dict: Dict[str, pd.DataFrame], strategy: BaseStrategy, 
                                                start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Run backtest using a strategy instance with multiple tickers"""
        if not data_dict:
            return self._empty_backtest_result()
        
        tickers = list(data_dict.keys())
        
        # Reset strategy state
        strategy.reset()
        
        # Initialize portfolio manager for multi-ticker
        portfolio = PortfolioManager(
            initial_capital=self.initial_capital,
            commission_rate=self.commission,
            tickers=tickers
        )
        
        # Prepare data with indicators for all tickers
        prepared_data_dict = {}
        for ticker, data in data_dict.items():
            prepared_data_dict[ticker] = strategy.prepare_data(data)
        
        # Find common date range across all tickers
        common_dates = None
        for ticker, df in prepared_data_dict.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        if not common_dates:
            return self._empty_backtest_result()
        
        common_dates = sorted(list(common_dates))
        
        # Generate signals for all tickers
        signals_dict = {}
        for ticker, df in prepared_data_dict.items():
            signals = strategy.generate_signals(df)
            signal_dict = {signal.date: signal for signal in signals}
            signals_dict[ticker] = signal_dict
        
        # Execute trades and track portfolio value throughout the period
        executed_trades = []
        for date in common_dates:
            date_str = date.strftime('%Y-%m-%d')
            current_prices = {}
            
            # Get current prices for all tickers on this date
            for ticker in tickers:
                if date in prepared_data_dict[ticker].index:
                    current_prices[ticker] = prepared_data_dict[ticker].loc[date, 'close']
            
            # Check for signals and execute trades
            for ticker in tickers:
                if date_str in signals_dict.get(ticker, {}):
                    signal = signals_dict[ticker][date_str]
                    trade = portfolio.execute_signal(signal, current_prices.get(ticker), ticker)
                    if trade:
                        executed_trades.append(trade)
            
            # Update portfolio history for every trading day
            portfolio.update_portfolio_history(date_str, current_prices)
        
        # Calculate SPY benchmark for the same period (needed for beta/alpha calculations)
        spy_start = start_date or min(df.index[0] for df in prepared_data_dict.values()).strftime('%Y-%m-%d')
        spy_end = end_date or max(df.index[-1] for df in prepared_data_dict.values()).strftime('%Y-%m-%d')
        spy_benchmark = await self._calculate_spy_benchmark_with_history(spy_start, spy_end)
        
        # Extract SPY returns for beta/alpha calculation
        spy_returns = None
        if 'spy_data' in spy_benchmark:
            spy_prices = spy_benchmark['spy_data']['close']
            spy_returns = spy_prices.pct_change().dropna()
        
        # Calculate performance metrics with SPY benchmark returns
        performance = portfolio.calculate_performance_metrics(spy_returns)
        
        # Calculate multi-ticker buy-and-hold benchmark
        multi_buy_hold_return = await self._calculate_multi_ticker_buy_hold(data_dict, start_date, end_date)
        
        # Create trade history with ticker information
        trade_history = []
        for trade in executed_trades:
            trade_entry = {
                'date': trade.date,
                'ticker': trade.metadata.get('ticker', 'N/A'),
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
        
        # Prepare plotting data for multi-ticker
        plot_data = self._prepare_multi_ticker_plot_data(portfolio.portfolio_history, data_dict, spy_benchmark)
        
        result = {
            'strategy_name': strategy.name,
            'tickers': tickers,
            'period': f"{min(df.index[0] for df in prepared_data_dict.values()).strftime('%Y-%m-%d')} to {max(df.index[-1] for df in prepared_data_dict.values()).strftime('%Y-%m-%d')}",
            'total_return': round(performance.get('total_return', 0) * 100, 2),
            'annual_return': round(performance.get('annual_return', 0) * 100, 2),
            'multi_ticker_buy_hold_return': round(multi_buy_hold_return * 100, 2),
            'excess_return': round((performance.get('total_return', 0) - multi_buy_hold_return) * 100, 2),
            'max_drawdown': round(performance.get('max_drawdown', 0) * 100, 2),
            'volatility': round(performance.get('volatility', 0) * 100, 2),
            'sharpe_ratio': round(performance.get('sharpe_ratio', 0), 2),
            'sortino_ratio': round(performance.get('sortino_ratio', 0), 3),
            'beta': round(performance.get('beta', 0), 3),
            'alpha': round(performance.get('alpha', 0), 3),
            'total_trades': performance.get('total_trades_completed', 0),
            'win_rate': round(performance.get('win_rate', 0) * 100, 2),
            'profit_factor': round(performance.get('profit_factor', 0), 2),
            'initial_capital': self.initial_capital,
            'final_portfolio_value': round(performance.get('final_portfolio_value', self.initial_capital), 2),
            'trade_history': trade_history,
            'summary': self._generate_multi_ticker_summary(strategy.name, performance.get('total_return', 0), multi_buy_hold_return, 
                                                          performance.get('win_rate', 0), performance.get('total_trades_completed', 0), tickers),
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
    
    async def run_strategy_backtest(self, data: pd.DataFrame, strategy: BaseStrategy, start_date: str = None, end_date: str = None, ticker: str = None) -> Dict[str, Any]:
        """Run backtest using a strategy instance"""
        if data.empty:
            return self._empty_backtest_result()
        
        # Reset strategy state
        strategy.reset()
        
        # Prepare data with indicators
        prepared_data = strategy.prepare_data(data)
        
        # Initialize portfolio manager (single ticker mode)
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
            
            # Update portfolio history for every trading day (single ticker mode)
            portfolio.update_portfolio_history(date_str, price=row['close'])
        
        # Calculate SPY benchmark for the same period (needed for beta/alpha calculations)
        spy_start = start_date or prepared_data.index[0].strftime('%Y-%m-%d')
        spy_end = end_date or prepared_data.index[-1].strftime('%Y-%m-%d')
        spy_benchmark = await self._calculate_spy_benchmark_with_history(spy_start, spy_end)
        
        # Extract SPY returns for beta/alpha calculation
        spy_returns = None
        if 'spy_data' in spy_benchmark:
            spy_prices = spy_benchmark['spy_data']['close']
            spy_returns = spy_prices.pct_change().dropna()
        
        # Calculate performance metrics with SPY benchmark returns
        performance = portfolio.calculate_performance_metrics(spy_returns)
        
        # Calculate buy-and-hold benchmark for the tested stock
        buy_hold_return = (prepared_data['close'].iloc[-1] - prepared_data['close'].iloc[0]) / prepared_data['close'].iloc[0] if len(prepared_data) > 0 else 0
        
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
            'sortino_ratio': round(performance.get('sortino_ratio', 0), 3),
            'beta': round(performance.get('beta', 0), 3),
            'alpha': round(performance.get('alpha', 0), 3),
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
                'spy_history': spy_history,
                'spy_data': spy_data  # Include raw SPY data for beta/alpha calculations
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
    
    async def _calculate_multi_ticker_buy_hold(self, data_dict: Dict[str, pd.DataFrame], start_date: str = None, end_date: str = None) -> float:
        """Calculate multi-ticker buy-and-hold benchmark with equal position sizing"""
        if not data_dict:
            return 0.0
        
        try:
            total_return = 0.0
            num_tickers = len(data_dict)
            capital_per_ticker = self.initial_capital / num_tickers
            
            for ticker, df in data_dict.items():
                if len(df) > 0:
                    # Calculate return for this ticker with equal capital allocation
                    ticker_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                    # Weight by equal capital allocation
                    total_return += ticker_return / num_tickers
            
            return total_return
            
        except Exception as e:
            print(f"Error calculating multi-ticker buy-and-hold: {str(e)}")
            return 0.0
    
    def _prepare_multi_ticker_plot_data(self, portfolio_history: List[Dict], data_dict: Dict[str, pd.DataFrame], spy_benchmark: Dict) -> Dict[str, Any]:
        """Prepare plot data for multi-ticker backtesting"""
        
        # Strategy portfolio values over time
        strategy_values = [
            {
                'date': item['date'],
                'value': item['portfolio_value'],
                'return_pct': ((item['portfolio_value'] - self.initial_capital) / self.initial_capital) * 100
            }
            for item in portfolio_history
        ]
        
        # Multi-ticker buy-and-hold benchmark (equally weighted)
        multi_ticker_buy_hold = []
        if data_dict:
            # Find common dates across all tickers
            common_dates = None
            for ticker, df in data_dict.items():
                if common_dates is None:
                    common_dates = set(df.index)
                else:
                    common_dates = common_dates.intersection(set(df.index))
            
            if common_dates:
                common_dates = sorted(list(common_dates))
                capital_per_ticker = self.initial_capital / len(data_dict)
                
                for date in common_dates:
                    date_str = date.strftime('%Y-%m-%d')
                    total_value = 0
                    
                    # Calculate portfolio value for this date across all tickers
                    for ticker, df in data_dict.items():
                        if date in df.index:
                            # Equal weight allocation
                            initial_price = df['close'].iloc[0]
                            current_price = df.loc[date, 'close']
                            ticker_value = capital_per_ticker * (current_price / initial_price)
                            total_value += ticker_value
                    
                    multi_ticker_buy_hold.append({
                        'date': date_str,
                        'value': float(total_value),
                        'return_pct': ((total_value - self.initial_capital) / self.initial_capital) * 100
                    })
        
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
            'multi_ticker_buy_hold': multi_ticker_buy_hold,
            'spy_benchmark': spy_values,
            'initial_capital': self.initial_capital,
            'tickers': list(data_dict.keys()) if data_dict else []
        }
    
    def _generate_multi_ticker_summary(self, strategy_name: str, total_return: float, buy_hold_return: float, 
                                      win_rate: float, total_trades: int, tickers: List[str]) -> str:
        """Generate human-readable summary for multi-ticker backtest"""
        performance = "outperformed" if total_return > buy_hold_return else "underperformed"
        ticker_str = ", ".join(tickers[:3]) + (f" and {len(tickers)-3} others" if len(tickers) > 3 else "")
        
        return (f"{strategy_name} on {ticker_str} generated a {total_return*100:.2f}% total return with {total_trades} trades "
                f"and a {win_rate*100:.1f}% win rate. The strategy {performance} equal-weight buy-and-hold by "
                f"{abs((total_return - buy_hold_return)*100):.2f} percentage points.")


