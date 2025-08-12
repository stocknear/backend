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


class CustomStrategy(BaseStrategy):
    """Custom strategy that uses flexible rule-based conditions with logical connectors"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__("Custom Rules", parameters)
        
        # Extract buy and sell conditions from parameters
        self.buy_conditions = parameters.get('buy_conditions', []) if parameters else []
        self.sell_conditions = parameters.get('sell_conditions', []) if parameters else []
        
        # Validate that both conditions are provided
        if not self.buy_conditions:
            raise ValueError("buy_conditions are required for CustomStrategy")
        if not self.sell_conditions:
            raise ValueError("sell_conditions are required for CustomStrategy")
        
        # Initialize rule engine
        from advanced_rule_engine import CustomRuleEngine
        self.rule_engine = CustomRuleEngine()
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by calculating all required indicators"""
        # The rule engine will calculate all indicators, so we return the raw data
        return data.copy()
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals using custom rules"""
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
                        'reason': 'Custom rule conditions met',
                        'conditions': 'buy_conditions',
                        'rule_type': 'custom'
                    }
                ))
            elif row['sell']:
                signals.append(TradingSignal(
                    signal_type=SignalType.SELL,
                    price=row['price'],
                    shares=0,  # Portfolio manager will calculate shares
                    date=date_str,
                    metadata={
                        'reason': 'Custom rule conditions met',
                        'conditions': 'sell_conditions',
                        'rule_type': 'custom'
                    }
                ))
        
        return signals
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy metadata including conditions"""
        info = super().get_strategy_info()
        info.update({
            'description': 'Custom rule-based strategy with flexible conditions and logical connectors',
            'buy_conditions': self.buy_conditions,
            'sell_conditions': self.sell_conditions,
            'rule_engine': 'CustomRuleEngine'
        })
        return info