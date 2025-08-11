from .backtest_engine import BacktestingEngine, TechnicalIndicators, BacktestRunner
from .backtesting_tools import run_single_strategy_backtest, run_comprehensive_backtest

# Make main classes easily accessible
__all__ = [
    'BacktestingEngine',
    'BacktestRunner', 
    'TechnicalIndicators', 
    'run_single_strategy_backtest',
    'run_comprehensive_backtest',
    'run'
]

# Default engine instance for quick access
default_engine = BacktestingEngine()

# Convenience functions that use the default engine
async def run(ticker: str, strategy_name: str = "buy_and_hold", 
              start_date: str = None, end_date: str = None, 
              comprehensive: bool = False, initial_capital: float = 100000, 
              **strategy_params):

    engine = BacktestingEngine(initial_capital=initial_capital)
    return await engine.run(ticker, strategy_name, start_date, end_date, comprehensive, **strategy_params)