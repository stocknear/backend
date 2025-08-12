from .backtest_engine import BacktestingEngine
from .technical_indicators import TechnicalIndicators

# Make main classes easily accessible
__all__ = [
    'BacktestingEngine',
    'TechnicalIndicators', 
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