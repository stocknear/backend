from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from enum import Enum


class SignalType(Enum):
    """Enumeration for trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingSignal:
    """Represents a trading signal with metadata"""
    
    def __init__(self, signal_type: SignalType, price: float, shares: int, 
                 date: str, metadata: Dict[str, Any] = None):
        self.signal_type = signal_type
        self.price = price
        self.shares = shares
        self.date = date
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary format"""
        return {
            'date': self.date,
            'action': self.signal_type.value,
            'price': self.price,
            'shares': self.shares,
            **self.metadata
        }


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.signals = []
        self.current_position = 0
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals based on the provided data
        
        Args:
            data: DataFrame with OHLCV data and any computed indicators
            
        Returns:
            List of TradingSignal objects
        """
        pass
    
    @abstractmethod
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with required indicators for the strategy
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with computed indicators
        """
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy metadata"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'description': self.__doc__ or 'No description available'
        }
    
    def reset(self):
        """Reset strategy state for new backtest"""
        self.signals = []
        self.current_position = 0


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy-and-hold strategy that buys at start and holds until end"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__("Buy and Hold", parameters)
        self.bought = False
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []
        
        if data.empty:
            return signals
        
        # Buy on first day
        if not self.bought:
            first_date = data.index[0].strftime('%Y-%m-%d')
            first_price = data['close'].iloc[0]
            
            signals.append(TradingSignal(
                SignalType.BUY,
                first_price,
                shares=1,  # Will be calculated by portfolio manager
                date=first_date,
                metadata={'portfolio_value': 0}
            ))
            self.bought = True
        
        # Sell on last day (for final calculation)
        last_date = data.index[-1].strftime('%Y-%m-%d')
        last_price = data['close'].iloc[-1]
        
        signals.append(TradingSignal(
            SignalType.SELL,
            last_price,
            shares=0,  # Will be set by portfolio manager
            date=last_date,
            metadata={'portfolio_value': 0}
        ))
        
        return signals
    
    def reset(self):
        super().reset()
        self.bought = False


class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'rsi_window': 14,
            'rsi_buy_threshold': 30,
            'rsi_sell_threshold': 70
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("RSI Strategy", default_params)
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        from backtest_engine import TechnicalIndicators
        
        data = data.copy()
        ti = TechnicalIndicators()
        data['rsi'] = ti.rsi(data['close'], window=self.parameters['rsi_window'])
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            if pd.isna(row['rsi']) or i < self.parameters['rsi_window']:
                continue
            
            date_str = date.strftime('%Y-%m-%d')
            
            # Buy signal: RSI below buy threshold and no current position
            if (row['rsi'] < self.parameters['rsi_buy_threshold'] and 
                self.current_position == 0):
                
                signals.append(TradingSignal(
                    SignalType.BUY,
                    row['close'],
                    shares=1,  # Will be calculated by portfolio manager
                    date=date_str,
                    metadata={
                        'rsi': row['rsi'],
                        'portfolio_value': 0
                    }
                ))
                self.current_position = 1
            
            # Sell signal: RSI above sell threshold and have position
            elif (row['rsi'] > self.parameters['rsi_sell_threshold'] and 
                  self.current_position > 0):
                
                signals.append(TradingSignal(
                    SignalType.SELL,
                    row['close'],
                    shares=self.current_position,
                    date=date_str,
                    metadata={
                        'rsi': row['rsi'],
                        'portfolio_value': 0
                    }
                ))
                self.current_position = 0
        
        return signals


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving average crossover strategy"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'short_window': 20,
            'long_window': 50
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(f"MA Crossover ({default_params['short_window']}/{default_params['long_window']})", default_params)
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        from backtest_engine import TechnicalIndicators
        
        data = data.copy()
        ti = TechnicalIndicators()
        data['sma_short'] = ti.sma(data['close'], self.parameters['short_window'])
        data['sma_long'] = ti.sma(data['close'], self.parameters['long_window'])
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            if (pd.isna(row['sma_short']) or pd.isna(row['sma_long']) or 
                i < self.parameters['long_window'] or i == 0):
                continue
            
            date_str = date.strftime('%Y-%m-%d')
            prev_row = data.iloc[i-1]
            
            # Buy signal: short MA crosses above long MA
            if (row['sma_short'] > row['sma_long'] and 
                prev_row['sma_short'] <= prev_row['sma_long'] and 
                self.current_position == 0):
                
                signals.append(TradingSignal(
                    SignalType.BUY,
                    row['close'],
                    shares=1,
                    date=date_str,
                    metadata={
                        'sma_short': row['sma_short'],
                        'sma_long': row['sma_long'],
                        'portfolio_value': 0
                    }
                ))
                self.current_position = 1
            
            # Sell signal: short MA crosses below long MA
            elif (row['sma_short'] < row['sma_long'] and 
                  prev_row['sma_short'] >= prev_row['sma_long'] and 
                  self.current_position > 0):
                
                signals.append(TradingSignal(
                    SignalType.SELL,
                    row['close'],
                    shares=self.current_position,
                    date=date_str,
                    metadata={
                        'sma_short': row['sma_short'],
                        'sma_long': row['sma_long'],
                        'portfolio_value': 0
                    }
                ))
                self.current_position = 0
        
        return signals


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean reversion strategy"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'window': 20,
            'num_std': 2
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(f"Bollinger Bands ({default_params['window']}, {default_params['num_std']})", default_params)
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        from backtest_engine import TechnicalIndicators
        
        data = data.copy()
        ti = TechnicalIndicators()
        bb = ti.bollinger_bands(data['close'], self.parameters['window'], self.parameters['num_std'])
        data['bb_upper'] = bb['upper']
        data['bb_middle'] = bb['middle']
        data['bb_lower'] = bb['lower']
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            if pd.isna(row['bb_upper']) or i < self.parameters['window']:
                continue
            
            date_str = date.strftime('%Y-%m-%d')
            
            # Buy signal: price touches lower band
            if row['close'] <= row['bb_lower'] and self.current_position == 0:
                signals.append(TradingSignal(
                    SignalType.BUY,
                    row['close'],
                    shares=1,
                    date=date_str,
                    metadata={
                        'bb_position': 'Lower Band',
                        'portfolio_value': 0
                    }
                ))
                self.current_position = 1
            
            # Sell signal: price touches upper band
            elif row['close'] >= row['bb_upper'] and self.current_position > 0:
                signals.append(TradingSignal(
                    SignalType.SELL,
                    row['close'],
                    shares=self.current_position,
                    date=date_str,
                    metadata={
                        'bb_position': 'Upper Band',
                        'portfolio_value': 0
                    }
                ))
                self.current_position = 0
        
        return signals


class MACDStrategy(BaseStrategy):
    """MACD crossover strategy"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'fast': 12,
            'slow': 26,
            'signal': 9
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(f"MACD ({default_params['fast']}, {default_params['slow']}, {default_params['signal']})", default_params)
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        from backtest_engine import TechnicalIndicators
        
        data = data.copy()
        ti = TechnicalIndicators()
        macd_data = ti.macd(data['close'], self.parameters['fast'], 
                           self.parameters['slow'], self.parameters['signal'])
        data['macd'] = macd_data['macd']
        data['macd_signal'] = macd_data['signal']
        data['macd_histogram'] = macd_data['histogram']
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            if (pd.isna(row['macd']) or pd.isna(row['macd_signal']) or 
                i < self.parameters['slow'] + self.parameters['signal'] or i == 0):
                continue
            
            date_str = date.strftime('%Y-%m-%d')
            prev_row = data.iloc[i-1]
            
            # Buy signal: MACD crosses above signal line
            if (row['macd'] > row['macd_signal'] and 
                prev_row['macd'] <= prev_row['macd_signal'] and 
                self.current_position == 0):
                
                signals.append(TradingSignal(
                    SignalType.BUY,
                    row['close'],
                    shares=1,
                    date=date_str,
                    metadata={
                        'macd': row['macd'],
                        'macd_signal': row['macd_signal'],
                        'portfolio_value': 0
                    }
                ))
                self.current_position = 1
            
            # Sell signal: MACD crosses below signal line
            elif (row['macd'] < row['macd_signal'] and 
                  prev_row['macd'] >= prev_row['macd_signal'] and 
                  self.current_position > 0):
                
                signals.append(TradingSignal(
                    SignalType.SELL,
                    row['close'],
                    shares=self.current_position,
                    date=date_str,
                    metadata={
                        'macd': row['macd'],
                        'macd_signal': row['macd_signal'],
                        'portfolio_value': 0
                    }
                ))
                self.current_position = 0
        
        return signals


class StrategyRegistry:
    """Registry for managing available strategies"""
    
    def __init__(self):
        self._strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register built-in strategies"""
        self.register_strategy('buy_and_hold', BuyAndHoldStrategy)
        self.register_strategy('rsi', RSIStrategy)
        self.register_strategy('ma_crossover', MovingAverageCrossoverStrategy)
        self.register_strategy('bollinger', BollingerBandsStrategy)
        self.register_strategy('macd', MACDStrategy)
    
    def register_strategy(self, name: str, strategy_class: type):
        """Register a new strategy class"""
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"Strategy class must inherit from BaseStrategy")
        self._strategies[name] = strategy_class
    
    def create_strategy(self, name: str, parameters: Dict[str, Any] = None) -> BaseStrategy:
        """Create a strategy instance by name"""
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(self._strategies.keys())}")
        
        return self._strategies[name](parameters)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self._strategies.keys())
    
    def get_strategy_info(self, name: str) -> Dict[str, Any]:
        """Get information about a strategy"""
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        
        dummy_instance = self._strategies[name]()
        return dummy_instance.get_strategy_info()