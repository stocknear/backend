import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union
from dataclasses import dataclass
from enum import Enum


class OperatorType(Enum):
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    ABOVE = "above"
    BELOW = "below"
    EQUALS = "equals"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"


class ConnectorType(Enum):
    AND = "AND"
    OR = "OR"


@dataclass
class RuleCondition:
    """Represents a single condition in a trading rule"""
    name: str  # Technical indicator name (e.g., "rsi", "ma_20", "price")
    value: Union[str, float, int]  # Value to compare against ("price" or numeric value)
    operator: OperatorType  # Comparison operator
    connector: ConnectorType = None  # Logical connector to next condition


class AdvancedRuleEngine:
    """Engine for evaluating complex trading rules with multiple conditions and logical connectors"""
    
    def __init__(self):
        self.indicators = {}
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all technical indicators for the given data"""
        indicators = {}
        
        # Price
        indicators['price'] = data['close']
        
        # RSI - only if we have enough data
        if len(data) >= 15:  # Need at least 15 points for RSI with window=14
            indicators['rsi'] = self._calculate_rsi(data['close'])
        else:
            indicators['rsi'] = pd.Series([50.0] * len(data), index=data.index)  # Default RSI
        
        # Moving Averages
        indicators['ma_5'] = data['close'].rolling(window=5).mean()
        indicators['ma_10'] = data['close'].rolling(window=10).mean()
        indicators['ma_20'] = data['close'].rolling(window=20).mean()
        indicators['ma_50'] = data['close'].rolling(window=50).mean()
        indicators['ma_100'] = data['close'].rolling(window=100).mean()
        indicators['ma_200'] = data['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        indicators['ema_5'] = data['close'].ewm(span=5).mean()
        indicators['ema_10'] = data['close'].ewm(span=10).mean()
        indicators['ema_20'] = data['close'].ewm(span=20).mean()
        indicators['ema_50'] = data['close'].ewm(span=50).mean()
        
        # MACD
        macd_data = self._calculate_macd(data['close'])
        indicators['macd'] = macd_data['macd']
        indicators['macd_signal'] = macd_data['signal']
        indicators['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(data['close'])
        indicators['bb_upper'] = bb_data['upper']
        indicators['bb_middle'] = bb_data['middle']
        indicators['bb_lower'] = bb_data['lower']
        
        # Volume indicators
        indicators['volume'] = data['volume']
        indicators['volume_ma'] = data['volume'].rolling(window=20).mean()
        
        self.indicators = indicators
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI with safety checks"""
        if len(prices) < window + 1:
            # Not enough data for RSI calculation
            return pd.Series([np.nan] * len(prices), index=prices.index)
            
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
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD"""
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
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower
        }
    
    def parse_conditions(self, conditions: List[Dict[str, Any]]) -> List[RuleCondition]:
        """Parse condition dictionaries into RuleCondition objects"""
        parsed_conditions = []
        
        for condition in conditions:
            operator = OperatorType(condition['operator'])
            connector = ConnectorType(condition['connector']) if condition.get('connector') else None
            
            parsed_condition = RuleCondition(
                name=condition['name'],
                value=condition['value'],
                operator=operator,
                connector=connector
            )
            parsed_conditions.append(parsed_condition)
        
        return parsed_conditions
    
    def evaluate_condition(self, condition: RuleCondition, index: int) -> bool:
        """Evaluate a single condition at a specific index"""
        if condition.name not in self.indicators:
            raise ValueError(f"Unknown indicator: {condition.name}")
        
        indicator_value = self.indicators[condition.name].iloc[index]
        
        # Handle NaN values
        if pd.isna(indicator_value):
            return False
        
        # Get comparison value
        if isinstance(condition.value, str):
            if condition.value == "price":
                compare_value = self.indicators['price'].iloc[index]
            elif condition.value in self.indicators:
                compare_value = self.indicators[condition.value].iloc[index]
            else:
                raise ValueError(f"Unknown value reference: {condition.value}")
        else:
            compare_value = condition.value
        
        # Handle NaN comparison values
        if pd.isna(compare_value):
            return False
        
        # Apply operator
        if condition.operator == OperatorType.GREATER_THAN:
            return indicator_value > compare_value
        elif condition.operator == OperatorType.LESS_THAN:
            return indicator_value < compare_value
        elif condition.operator == OperatorType.ABOVE:
            return indicator_value > compare_value
        elif condition.operator == OperatorType.BELOW:
            return indicator_value < compare_value
        elif condition.operator == OperatorType.EQUALS:
            return abs(indicator_value - compare_value) < 1e-10
        elif condition.operator == OperatorType.GREATER_EQUAL:
            return indicator_value >= compare_value
        elif condition.operator == OperatorType.LESS_EQUAL:
            return indicator_value <= compare_value
        else:
            raise ValueError(f"Unknown operator: {condition.operator}")
    
    def evaluate_rules(self, conditions: List[RuleCondition], index: int) -> bool:
        """Evaluate a list of conditions with logical connectors"""
        if not conditions:
            return False
        
        # Start with the first condition
        result = self.evaluate_condition(conditions[0], index)
        
        # Process remaining conditions with connectors
        for i in range(len(conditions) - 1):
            current_condition = conditions[i]
            next_condition = conditions[i + 1]
            
            next_result = self.evaluate_condition(next_condition, index)
            
            if current_condition.connector == ConnectorType.AND:
                result = result and next_result
            elif current_condition.connector == ConnectorType.OR:
                result = result or next_result
            else:
                # If no connector specified, default to AND
                result = result and next_result
        
        return result
    
    def generate_signals(self, data: pd.DataFrame, buy_conditions: List[Dict[str, Any]], 
                        sell_conditions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate buy/sell signals based on advanced rules"""
        
        # Calculate all indicators
        self.calculate_indicators(data)
        
        # Parse conditions
        parsed_buy_conditions = self.parse_conditions(buy_conditions)
        parsed_sell_conditions = self.parse_conditions(sell_conditions)
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['buy'] = False
        signals['sell'] = False
        signals['price'] = data['close']
        
        for i in range(len(data)):
            try:
                # Evaluate buy conditions
                if self.evaluate_rules(parsed_buy_conditions, i):
                    signals.iloc[i, signals.columns.get_loc('buy')] = True
                
                # Evaluate sell conditions
                if self.evaluate_rules(parsed_sell_conditions, i):
                    signals.iloc[i, signals.columns.get_loc('sell')] = True
                    
            except (IndexError, KeyError):
                # Handle edge cases where indicators may not be available
                continue
        
        return signals


class AdvancedRuleStrategy:
    """Custom strategy that uses the advanced rule engine"""
    
    def __init__(self, name: str, buy_conditions: List[Dict[str, Any]], 
                 sell_conditions: List[Dict[str, Any]]):
        self.name = name
        self.buy_conditions = buy_conditions
        self.sell_conditions = sell_conditions
        self.rule_engine = AdvancedRuleEngine()
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals using advanced rules"""
        
        # Generate signals using rule engine
        signals_df = self.rule_engine.generate_signals(data, self.buy_conditions, self.sell_conditions)
        
        # Convert to signal format compatible with existing backtesting engine
        signals = []
        
        for date, row in signals_df.iterrows():
            if row['buy']:
                signals.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'signal': 'BUY',
                    'price': row['price'],
                    'reason': 'Advanced rule conditions met'
                })
            elif row['sell']:
                signals.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'signal': 'SELL',
                    'price': row['price'],
                    'reason': 'Advanced rule conditions met'
                })
        
        return signals