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


class AdvancedStrategy(BaseStrategy):
    """Advanced strategy that uses flexible rule-based conditions with logical connectors"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__("Advanced Rules", parameters)
        
        # Extract buy and sell conditions from parameters
        self.buy_conditions = parameters.get('buy_conditions', []) if parameters else []
        self.sell_conditions = parameters.get('sell_conditions', []) if parameters else []
        
        # Validate that both conditions are provided
        if not self.buy_conditions:
            raise ValueError("buy_conditions are required for AdvancedStrategy")
        if not self.sell_conditions:
            raise ValueError("sell_conditions are required for AdvancedStrategy")
        
        # Initialize rule engine
        from advanced_rule_engine import AdvancedRuleEngine
        self.rule_engine = AdvancedRuleEngine()
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by calculating all required indicators"""
        # The rule engine will calculate all indicators, so we return the raw data
        return data.copy()
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals using advanced rules"""
        signals = []
        
        if data.empty:
            return signals
        
        # Generate signals using rule engine
        signals_df = self.rule_engine.generate_signals(data, self.buy_conditions, self.sell_conditions)
        
        # Convert to TradingSignal objects
        for date, row in signals_df.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            
            if row['buy']:
                signals.append(TradingSignal(
                    signal_type=SignalType.BUY,
                    price=row['price'],
                    shares=0,  # Portfolio manager will calculate shares
                    date=date_str,
                    metadata={
                        'reason': 'Advanced rule conditions met',
                        'conditions': 'buy_conditions',
                        'rule_type': 'advanced'
                    }
                ))
            elif row['sell']:
                signals.append(TradingSignal(
                    signal_type=SignalType.SELL,
                    price=row['price'],
                    shares=0,  # Portfolio manager will calculate shares
                    date=date_str,
                    metadata={
                        'reason': 'Advanced rule conditions met',
                        'conditions': 'sell_conditions',
                        'rule_type': 'advanced'
                    }
                ))
        
        return signals
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy metadata including conditions"""
        info = super().get_strategy_info()
        info.update({
            'description': 'Advanced rule-based strategy with flexible conditions and logical connectors',
            'buy_conditions': self.buy_conditions,
            'sell_conditions': self.sell_conditions,
            'rule_engine': 'AdvancedRuleEngine'
        })
        return info


class StrategyRegistry:
    """Registry for managing available strategies"""
    
    def __init__(self):
        self._strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register built-in strategies"""
        self.register_strategy('advanced', AdvancedStrategy)
    
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
        
        # Create a dummy instance with minimal parameters to get info
        try:
            dummy_params = {
                'buy_conditions': [{'name': 'rsi', 'value': 30, 'operator': 'below'}],
                'sell_conditions': [{'name': 'rsi', 'value': 70, 'operator': 'above'}]
            }
            dummy_instance = self._strategies[name](dummy_params)
            return dummy_instance.get_strategy_info()
        except Exception as e:
            return {
                'name': name,
                'description': f'Strategy info unavailable: {str(e)}',
                'parameters': {}
            }