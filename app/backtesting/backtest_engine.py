import pandas as pd
from typing import Dict, List, Optional, Any
from backtesting.strategy_engine import BaseStrategy, CustomStrategy
from backtesting.portfolio_manager import PortfolioManager
from backtesting.data_manager import DataManager
import sqlite3

def load_symbol_list():
    stock_symbols = []
    etf_symbols = []
    index_symbols = []

    db_configs = [
        ("stocks.db", "SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'", stock_symbols),
        ("etf.db",    "SELECT DISTINCT symbol FROM etfs", etf_symbols),
        ("index.db",  "SELECT DISTINCT symbol FROM indices", index_symbols),
    ]

    for db_file, query, target_list in db_configs:
        try:
            con = sqlite3.connect(db_file)
            cur = con.cursor()
            cur.execute(query)
            target_list.extend([r[0] for r in cur.fetchall()])
            con.close()
        except Exception:
            continue

    return stock_symbols, etf_symbols, index_symbols

stock_symbols, etf_symbols, index_symbols = load_symbol_list()

class BacktestingEngine:
    """Custom backtesting engine with rule-based strategy support"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.data_manager = DataManager()
    
    async def run(self, tickers: List[str], buy_conditions: List[Dict[str, Any]], 
                  sell_conditions: List[Dict[str, Any]], start_date: str = None, 
                  end_date: str = None) -> Dict:
        """
        Unified method to run custom rule-based backtesting
        
        Args:
            tickers: List of ticker symbols to backtest
            buy_conditions: List of buy condition dictionaries with logical connectors
            sell_conditions: List of sell condition dictionaries with logical connectors
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
        
        Returns:
            Dictionary with backtesting results
        """
        # Validate that conditions are provided
        if not buy_conditions:
            return {
                'tickers': [t.upper() for t in tickers] if len(tickers) > 1 else tickers[0].upper(),
                'strategy': 'custom',
                'error': 'buy_conditions are required and cannot be empty',
                'success': False
            }
        
        if not sell_conditions:
            return {
                'tickers': [t.upper() for t in tickers] if len(tickers) > 1 else tickers[0].upper(),
                'strategy': 'custom',
                'error': 'sell_conditions are required and cannot be empty',
                'success': False
            }
        
        try:
            # Create strategy with conditions
            parameters = {
                'buy_conditions': buy_conditions,
                'sell_conditions': sell_conditions
            }
            strategy = CustomStrategy(parameters)
            
            # Load data
            if len(tickers) == 1:
                # Single ticker mode
                ticker = tickers[0]
                data = await self.data_manager.load_historical_data(ticker, start_date, end_date)
                
                if data.empty:
                    return {
                        'ticker': ticker.upper(),
                        'strategy': 'custom',
                        'error': f'No historical data available for {ticker}',
                        'success': False
                    }
                
                # Run single ticker backtest
                result = await self.run_strategy_backtest(data, strategy, start_date, end_date, ticker)
                result.update({
                    'ticker': ticker.upper(),
                    'strategy': 'custom',
                    'success': True,
                    'data_points': len(data)
                })
                
                return result
                
            else:
                # Multi-ticker mode
                data_dict = await self.data_manager.load_multiple_tickers(tickers, start_date, end_date)
                
                if not data_dict:
                    return {
                        'tickers': [t.upper() for t in tickers],
                        'strategy': 'custom',
                        'error': f'No historical data available for any of the tickers: {tickers}',
                        'success': False
                    }
                
                # Run multi-ticker backtest
                result = await self.run_strategy_backtest_multi_ticker(data_dict, strategy, start_date, end_date)
                result.update({
                    'strategy': 'custom',
                    'success': True,
                    'data_points': sum(len(df) for df in data_dict.values())
                })
                
                return result
                
        except Exception as e:
            return {
                'tickers': [t.upper() for t in tickers] if len(tickers) > 1 else tickers[0].upper(),
                'strategy': 'custom',
                'error': f'Custom backtesting failed: {str(e)}',
                'success': False
            }
    

    async def run_strategy_backtest_multi_ticker(self, data_dict: Dict[str, pd.DataFrame], strategy: BaseStrategy, 
                                                start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Run backtest using a strategy instance with multiple tickers"""
        if not data_dict:
            return self._empty_backtest_result()
        
        tickers = list(data_dict.keys())
        
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
        
        # Sort dates
        common_dates = sorted(list(common_dates))
        
        # Initialize portfolio manager for multi-ticker trading
        portfolio = PortfolioManager(
            initial_capital=self.initial_capital,
            commission_rate=self.commission,
            position_size_pct=0.95,
            tickers=tickers
        )
        
        # Generate signals for all tickers
        signals_dict = {}
        for ticker, df in prepared_data_dict.items():
            signals = strategy.generate_signals(df)
            signal_dict = {signal.date: signal for signal in signals}
            signals_dict[ticker] = signal_dict
        
        executed_trades = []
        
        # Execute trades for each common trading day
        for date in common_dates:
            date_str = date.strftime('%Y-%m-%d')
            
            # Get current prices for all tickers on this date
            current_prices = {}
            for ticker in tickers:
                if date in prepared_data_dict[ticker].index:
                    current_prices[ticker] = prepared_data_dict[ticker].loc[date, 'close']
            
            # Check for signals and execute trades
            for ticker in tickers:
                if ticker in signals_dict and date_str in signals_dict[ticker]:
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
        portfolio_history_map = {h['date']: h for h in portfolio.portfolio_history}

        for trade in executed_trades:
            ticker = trade.metadata.get("ticker")
            trade_entry = {
                'date': trade.date,
                'action': trade.action,
                'shares': trade.shares,
                'price': trade.price,
                'commission': trade.commission,
                'gross_amount': trade.gross_amount,
                'net_amount': trade.net_amount,
                'symbol': ticker
            }

            # Asset type
            if ticker:
                if ticker in stock_symbols:
                    trade_entry['assetType'] = 'stocks'
                elif ticker in etf_symbols:
                    trade_entry['assetType'] = 'etf'
                elif ticker in index_symbols:
                    trade_entry['assetType'] = 'index'
                else:
                    trade_entry['assetType'] = ""

            # Add portfolio value & return_pct at trade time
            if trade.date in portfolio_history_map:
                hist = portfolio_history_map[trade.date]
                trade_entry['portfolio_value'] = hist['portfolio_value']
                trade_entry['return_pct'] = round(((hist['portfolio_value'] - self.initial_capital) / self.initial_capital) * 100, 2)

            trade_history.append(trade_entry)

        
        # Prepare plotting data for multi-ticker
        plot_data = self._prepare_multi_ticker_plot_data(portfolio.portfolio_history, data_dict, spy_benchmark)
        
        result = {
            'strategy_name': strategy.name,
            'tickers': tickers,
            'period': f"{min(df.index[0] for df in prepared_data_dict.values()).strftime('%b %d, %Y')} to {max(df.index[-1] for df in prepared_data_dict.values()).strftime('%b %d, %Y')}",
            'total_return': round(performance.get('total_return', 0) * 100, 2),
            'annual_return': round(performance.get('annual_return', 0) * 100, 2),
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
            # SPY Benchmark
            'spy_benchmark': {
                'spy_return': round(spy_benchmark.get('spy_return', 0) * 100, 2),
                'spy_annual_return': round(spy_benchmark.get('spy_annual_return', 0) * 100, 2),
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
            commission_rate=self.commission,
            position_size_pct=0.95,
            tickers=[ticker] if ticker else None
        )
        
        # Generate signals
        signals = strategy.generate_signals(prepared_data)
        
        # Create a dict to map dates to signals for efficient lookup
        signal_dict = {signal.date: signal for signal in signals}
        
        # Execute trades and track portfolio value throughout the period
        executed_trades = []
        
        # Execute trades and update portfolio history during the process
        for i, (date, row) in enumerate(prepared_data.iterrows()):
            date_str = date.strftime('%Y-%m-%d')
            current_prices = {ticker: row['close']}
            
            # Check if there's a signal for this date and execute it
            if date_str in signal_dict:
                trade = portfolio.execute_signal(signal_dict[date_str], row['close'], ticker)
                if trade:
                    executed_trades.append(trade)
            
            # Update portfolio history AFTER any trades on this date
            portfolio.update_portfolio_history(date_str, current_prices)
        
        
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

        
        # Create trade history with ticker information
        trade_history = []
        portfolio_history_map = {h['date']: h for h in portfolio.portfolio_history}

        for trade in executed_trades:
            trade_entry = {
                'date': trade.date,
                'action': trade.action,
                'shares': trade.shares,
                'price': trade.price,
                'commission': trade.commission,
                'gross_amount': trade.gross_amount,
                'net_amount': trade.net_amount
            }

            # Asset type
            if ticker:
                trade_entry['symbol'] = ticker
                if ticker in stock_symbols:
                    trade_entry['assetType'] = 'stocks'
                elif ticker in etf_symbols:
                    trade_entry['assetType'] = 'etf'
                elif ticker in index_symbols:
                    trade_entry['assetType'] = 'index'
                else:
                    trade_entry['assetType'] = ""

            # Add portfolio value & return_pct at trade time
            if trade.date in portfolio_history_map:
                hist = portfolio_history_map[trade.date]
                trade_entry['portfolio_value'] = hist['portfolio_value']
                trade_entry['return_pct'] = round(((hist['portfolio_value'] - self.initial_capital) / self.initial_capital) * 100, 2)

            trade_history.append(trade_entry)


        # Prepare plot data
        plot_data = self._prepare_plot_data(portfolio.portfolio_history, prepared_data, spy_benchmark)
        
        result = {
            'strategy_name': strategy.name,
            'period': f"{prepared_data.index[0].strftime('%b %d, %Y')} to {prepared_data.index[-1].strftime('%b %d, %Y')}",
            'total_return': round(performance.get('total_return', 0) * 100, 2),
            'annual_return': round(performance.get('annual_return', 0) * 100, 2),
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
            # SPY Benchmark
            'spy_benchmark': {
                'spy_return': round(spy_benchmark.get('spy_return', 0) * 100, 2),
                'spy_annual_return': round(spy_benchmark.get('spy_annual_return', 0) * 100, 2),
            },
            # Plotting data for visualization
            'plot_data': plot_data
        }
        
        # Add error information if SPY benchmark failed
        if 'error' in spy_benchmark:
            result['spy_benchmark']['error'] = spy_benchmark['error']
        
        return result

    def _empty_backtest_result(self) -> Dict[str, Any]:
        """Return empty backtest result structure"""
        return {
            'strategy_name': 'Unknown',
            'total_return': 0,
            'annual_return': 0,
            'max_drawdown': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'total_trades': 0,
            'final_portfolio_value': self.initial_capital,
            'error': 'No data available for backtesting'
        }

    async def _calculate_spy_benchmark(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Calculate SPY buy-and-hold benchmark return"""
        try:
            spy_data = await self.data_manager.load_historical_data("SPY", start_date, end_date)
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
            spy_data = await self.data_manager.load_historical_data("SPY", start_date, end_date)
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
        
        # Strategy portfolio values over time (normalized to percentage returns)
        initial_value = portfolio_history[0]['portfolio_value'] if portfolio_history else self.initial_capital
        strategy_values = [
            {
                'date': item['date'],
                'value': item['portfolio_value'],
                'return_pct': round(((item['portfolio_value'] - initial_value) / initial_value) * 100, 2)
            }
            for item in portfolio_history
        ]
        
        # Stock buy-and-hold performance (normalized to same initial capital)
        stock_initial = stock_data['close'].iloc[0]
        stock_buy_hold = [
            {
                'date': date.strftime('%Y-%m-%d'),
                'value': (price / stock_initial) * self.initial_capital,
                'return_pct': round(((price / stock_initial - 1) * 100), 2)
            }
            for date, price in stock_data['close'].items()
        ]
        
        # SPY benchmark performance (from spy_benchmark['spy_history'])
        spy_buy_hold = [
            {
                'date': item['date'],
                'value': item['value'],
                'return_pct': round(((item['value'] - self.initial_capital) / self.initial_capital) * 100, 2)
            }
            for item in spy_benchmark.get('spy_history', [])
        ]
        
        return {
            'strategy': strategy_values,
            'stock_buy_hold': stock_buy_hold,
            'spy_benchmark': spy_buy_hold
        }

    async def _calculate_multi_ticker_buy_hold(self, data_dict: Dict[str, pd.DataFrame], start_date: str = None, end_date: str = None) -> float:
        """Calculate equal-weight buy-and-hold return for multiple tickers"""
        try:
            if not data_dict:
                return 0.0
            
            num_tickers = len(data_dict)
            capital_per_ticker = self.initial_capital / num_tickers
            total_return = 0.0
            
            for ticker, df in data_dict.items():
                if len(df) > 0:
                    # Calculate individual ticker return
                    ticker_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                    # Add weighted return (equal weight)
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
                'return_pct': round(((item['portfolio_value'] - self.initial_capital) / self.initial_capital) * 100, 2)
            }
            for item in portfolio_history
        ]
        
        # Multi-ticker equal-weight buy-and-hold simulation
        # Find common date range
        common_dates = None
        for ticker, df in data_dict.items():
            ticker_dates = set(df.index.strftime('%Y-%m-%d'))
            if common_dates is None:
                common_dates = ticker_dates
            else:
                common_dates = common_dates.intersection(ticker_dates)
        
        if common_dates:
            common_dates = sorted(list(common_dates))
            capital_per_ticker = self.initial_capital / len(data_dict)
            
            multi_buy_hold = []
            for date_str in common_dates:
                date_dt = pd.to_datetime(date_str)
                total_value = 0.0
                
                for ticker, df in data_dict.items():
                    if date_dt in df.index:
                        # Calculate value for each ticker (equal weight)
                        initial_price = df['close'].iloc[0]
                        current_price = df.loc[date_dt, 'close']
                        ticker_value = (current_price / initial_price) * capital_per_ticker
                        total_value += ticker_value
                
                multi_buy_hold.append({
                    'date': date_str,
                    'value': total_value,
                    'return_pct': round(((total_value - self.initial_capital) / self.initial_capital) * 100, 2)
                })
        else:
            multi_buy_hold = []
        
        # SPY benchmark (same as single ticker)
        spy_buy_hold = [
            {
                'date': item['date'],
                'value': item['value'],
                'return_pct': round(((item['value'] - self.initial_capital) / self.initial_capital) * 100, 2)
            }
            for item in spy_benchmark.get('spy_history', [])
        ]
        
        return {
            'strategy': strategy_values,
            'multi_ticker_buy_hold': multi_buy_hold,
            'spy_benchmark': spy_buy_hold,
            'tickers': list(data_dict.keys())
        }

    def _generate_summary(self, strategy_name: str, total_return: float, benchmark_return: float, win_rate: float, total_trades: int) -> str:
        """Generate a text summary of the backtest results"""
        return f"{strategy_name} generated {total_return*100:.1f}% return vs {benchmark_return*100:.1f}% buy-and-hold, with {win_rate*100:.1f}% win rate over {total_trades} trades"

    def _generate_multi_ticker_summary(self, strategy_name: str, total_return: float, benchmark_return: float, win_rate: float, total_trades: int, tickers: List[str]) -> str:
        """Generate a text summary for multi-ticker backtest results"""
        return f"{strategy_name} on {len(tickers)} tickers generated {total_return*100:.1f}% return vs {benchmark_return*100:.1f}% equal-weight buy-and-hold, with {win_rate*100:.1f}% win rate over {total_trades} trades"