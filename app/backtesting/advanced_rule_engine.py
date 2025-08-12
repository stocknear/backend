import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union
from dataclasses import dataclass
from enum import Enum
from technical_indicators import TechnicalIndicators


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
        self.ti = TechnicalIndicators()
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all technical indicators for the given data"""
        indicators = {}
        
        # Price
        indicators['price'] = data['close']
        
        # RSI
        indicators['rsi'] = self.ti.rsi(data['close'])
        
        # Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            indicators[f'ma_{window}'] = self.ti.sma(data['close'], window)
        
        # Exponential Moving Averages
        for window in [5, 10, 20, 50]:
            indicators[f'ema_{window}'] = self.ti.ema(data['close'], window)
        
        # MACD
        macd_data = self.ti.macd(data['close'])
        indicators['macd'] = macd_data['macd']
        indicators['macd_signal'] = macd_data['signal']
        indicators['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self.ti.bollinger_bands(data['close'])
        indicators['bb_upper'] = bb_data['upper']
        indicators['bb_middle'] = bb_data['middle']
        indicators['bb_lower'] = bb_data['lower']
        
        # Volume indicators
        indicators['volume'] = data['volume']
        indicators['volume_ma'] = data['volume'].rolling(window=20).mean()
        
        self.indicators = indicators
        return indicators
    
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
        
        # Generate signals using vectorized operations where possible
        signals = pd.DataFrame(index=data.index)
        signals['buy'] = False
        signals['sell'] = False
        signals['price'] = data['close']
        
        # Optimize: Use vectorized evaluation for simple conditions when possible
        # For complex conditions with logical connectors, fall back to row-by-row
        try:
            if self._can_vectorize_conditions(parsed_buy_conditions):
                buy_mask = self._evaluate_conditions_vectorized(parsed_buy_conditions)
                signals['buy'] = buy_mask
            else:
                for i in range(len(data)):
                    if self.evaluate_rules(parsed_buy_conditions, i):
                        signals.iloc[i, signals.columns.get_loc('buy')] = True
            
            if self._can_vectorize_conditions(parsed_sell_conditions):
                sell_mask = self._evaluate_conditions_vectorized(parsed_sell_conditions)
                signals['sell'] = sell_mask
            else:
                for i in range(len(data)):
                    if self.evaluate_rules(parsed_sell_conditions, i):
                        signals.iloc[i, signals.columns.get_loc('sell')] = True
        except Exception as e:
            # Fallback to safe row-by-row processing
            for i in range(len(data)):
                try:
                    if self.evaluate_rules(parsed_buy_conditions, i):
                        signals.iloc[i, signals.columns.get_loc('buy')] = True
                    if self.evaluate_rules(parsed_sell_conditions, i):
                        signals.iloc[i, signals.columns.get_loc('sell')] = True
                except (IndexError, KeyError):
                    continue
        
        return signals
    
    def _can_vectorize_conditions(self, conditions: List[RuleCondition]) -> bool:
        """Check if conditions can be evaluated using vectorized operations"""
        # Simple heuristic: if only one condition and no complex value references
        if len(conditions) == 1:
            condition = conditions[0]
            return isinstance(condition.value, (int, float)) and condition.name in self.indicators
        return False
    
    def _evaluate_conditions_vectorized(self, conditions: List[RuleCondition]) -> pd.Series:
        """Evaluate simple conditions using vectorized pandas operations"""
        if len(conditions) != 1:
            raise ValueError("Vectorized evaluation only supports single conditions")
        
        condition = conditions[0]
        indicator_values = self.indicators[condition.name]
        
        if condition.operator == OperatorType.GREATER_THAN or condition.operator == OperatorType.ABOVE:
            return indicator_values > condition.value
        elif condition.operator == OperatorType.LESS_THAN or condition.operator == OperatorType.BELOW:
            return indicator_values < condition.value
        elif condition.operator == OperatorType.EQUALS:
            return np.abs(indicator_values - condition.value) < 1e-10
        elif condition.operator == OperatorType.GREATER_EQUAL:
            return indicator_values >= condition.value
        elif condition.operator == OperatorType.LESS_EQUAL:
            return indicator_values <= condition.value
        else:
            raise ValueError(f"Unsupported operator for vectorized evaluation: {condition.operator}")