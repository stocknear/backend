"""
Shared technical indicators module for backtesting system.
Centralizes all technical indicator calculations to avoid duplication.
"""

import pandas as pd
import numpy as np
from typing import Dict


class TechnicalIndicators:
    """Centralized technical indicator calculations"""
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index) with safety checks
        
        Args:
            prices: Price series
            window: Period for RSI calculation (default: 14)
            
        Returns:
            RSI values as pandas Series
        """
        if len(prices) < window + 1:
            return pd.Series([50.0] * len(prices), index=prices.index)
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            # Replace inf and -inf with NaN
            rsi = rsi.replace([np.inf, -np.inf], np.nan)
        
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
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' series
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            window: Period for moving average (default: 20)
            num_std: Number of standard deviations (default: 2)
            
        Returns:
            Dictionary with 'upper', 'middle', and 'lower' series
        """
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower
        }