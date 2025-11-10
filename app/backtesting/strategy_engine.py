from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from enum import Enum
from .advanced_rule_engine import CustomRuleEngine

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
        
        # Risk management parameters
        self.stop_loss_pct = parameters.get('stop_loss', None) if parameters else None
        self.profit_taker_pct = parameters.get('profit_taker', None) if parameters else None
        
        # Track entry prices for risk management (ticker -> entry_price)
        self.entry_prices = {}
        
        # Validate that both conditions are provided
        if not self.buy_conditions:
            raise ValueError("buy_conditions are required for CustomStrategy")
        if not self.sell_conditions:
            raise ValueError("sell_conditions are required for CustomStrategy")
        
        
        self.rule_engine = CustomRuleEngine()
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by calculating all required indicators"""
        # The rule engine will calculate all indicators, so we return the raw data
        return data.copy()
    
    def generate_signals(self, data: pd.DataFrame, ticker: str = None) -> List[TradingSignal]:
        """Generate trading signals using custom rules with risk management"""
        signals = []
        
        if data.empty:
            return signals
        
        # Generate signals using rule engine
        signals_df = self.rule_engine.generate_signals(data, self.buy_conditions, self.sell_conditions)
        
        # Track position state for risk management
        in_position = False
        entry_price = None
        
        # Convert to TradingSignal objects with risk management
        for date, row in signals_df.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            current_price = row['price']
            
            # Check for risk management exits if in position
            if in_position and entry_price is not None:
                exit_signal = None
                exit_reason = None
                
                # Check stop loss
                if self.stop_loss_pct is not None:
                    stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100)
                    if current_price <= stop_loss_price:
                        exit_signal = SignalType.SELL
                        exit_reason = f'Stop loss triggered at {self.stop_loss_pct}% loss'
                
                # Check profit taker
                if self.profit_taker_pct is not None and exit_signal is None:
                    profit_taker_price = entry_price * (1 + self.profit_taker_pct / 100)
                    if current_price >= profit_taker_price:
                        exit_signal = SignalType.SELL
                        exit_reason = f'Profit taker triggered at {self.profit_taker_pct}% gain'
                
                # Create exit signal if triggered
                if exit_signal is not None:
                    signals.append(TradingSignal(
                        signal_type=exit_signal,
                        price=current_price,
                        shares=0,
                        date=date_str,
                        metadata={
                            'reason': exit_reason,
                            'conditions': 'risk_management',
                            'rule_type': 'risk_management',
                            'entry_price': entry_price
                        }
                    ))
                    in_position = False
                    entry_price = None
                    continue  # Skip regular signal processing for this row
            
            # Process regular buy/sell signals
            if row['buy'] and not in_position:
                signals.append(TradingSignal(
                    signal_type=SignalType.BUY,
                    price=current_price,
                    shares=0,  # Portfolio manager will calculate shares
                    date=date_str,
                    metadata={
                        'reason': 'Custom rule conditions met',
                        'conditions': 'buy_conditions',
                        'rule_type': 'custom'
                    }
                ))
                in_position = True
                entry_price = current_price
                
            elif row['sell'] and in_position:
                signals.append(TradingSignal(
                    signal_type=SignalType.SELL,
                    price=current_price,
                    shares=0,  # Portfolio manager will calculate shares
                    date=date_str,
                    metadata={
                        'reason': 'Custom rule conditions met',
                        'conditions': 'sell_conditions',
                        'rule_type': 'custom'
                    }
                ))
                in_position = False
                entry_price = None
        
        return signals
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy metadata including conditions"""
        info = super().get_strategy_info()
        info.update({
            'description': 'Custom rule-based strategy with flexible conditions and logical connectors',
            'buy_conditions': self.buy_conditions,
            'sell_conditions': self.sell_conditions,
            'stop_loss': self.stop_loss_pct,
            'profit_taker': self.profit_taker_pct,
            'rule_engine': 'CustomRuleEngine'
        })
        return info