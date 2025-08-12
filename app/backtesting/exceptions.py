"""
Custom exceptions for the backtesting system.
Provides specific error types for better error handling and debugging.
"""


class BacktestingError(Exception):
    """Base exception for all backtesting-related errors"""
    pass


class DataLoadingError(BacktestingError):
    """Raised when data loading fails"""
    def __init__(self, ticker: str, message: str):
        self.ticker = ticker
        super().__init__(f"Failed to load data for {ticker}: {message}")


class StrategyError(BacktestingError):
    """Raised when strategy initialization or execution fails"""
    pass


class InvalidConditionError(StrategyError):
    """Raised when buy/sell conditions are invalid"""
    pass


class IndicatorCalculationError(BacktestingError):
    """Raised when technical indicator calculation fails"""
    def __init__(self, indicator: str, message: str):
        self.indicator = indicator
        super().__init__(f"Failed to calculate {indicator}: {message}")


class InsufficientDataError(BacktestingError):
    """Raised when there's insufficient data for analysis"""
    def __init__(self, required: int, available: int):
        self.required = required
        self.available = available
        super().__init__(f"Insufficient data: required {required}, available {available}")