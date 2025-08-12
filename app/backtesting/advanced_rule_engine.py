import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union
from dataclasses import dataclass
from enum import Enum
from backtesting.technical_indicators import TechnicalIndicators


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


class CustomRuleEngine:
    """Engine for evaluating complex trading rules with multiple conditions and logical connectors"""
    
    def __init__(self):
        self.indicators = {}
        self.ti = TechnicalIndicators()
        
    def calculate_indicators(self, data: pd.DataFrame, required_indicators: set) -> Dict[str, pd.Series]:
        """Calculate only the technical indicators that are required by the conditions"""
        indicators = {}
        
        # Always include price as it's fundamental
        indicators['price'] = data['close']
        
        # Calculate only required indicators
        if 'rsi' in required_indicators:
            indicators['rsi'] = self.ti.rsi(data['close'])
        
        # Moving Averages - only calculate needed windows
        ma_windows = [int(ind.split('_')[1]) for ind in required_indicators if ind.startswith('sma_')]
        for window in ma_windows:
            indicators[f'sma_{window}'] = self.ti.sma(data['close'], window)
        
        # Exponential Moving Averages - only calculate needed windows  
        ema_windows = [int(ind.split('_')[1]) for ind in required_indicators if ind.startswith('ema_')]
        for window in ema_windows:
            indicators[f'ema_{window}'] = self.ti.ema(data['close'], window)
        
        # MACD components - only if any MACD indicator is needed
        macd_indicators = {'macd', 'macd_signal', 'macd_histogram'}
        if macd_indicators.intersection(required_indicators):
            macd_data = self.ti.macd(data['close'])
            if 'macd' in required_indicators:
                indicators['macd'] = macd_data['macd']
            if 'macd_signal' in required_indicators:
                indicators['macd_signal'] = macd_data['signal']
            if 'macd_histogram' in required_indicators:
                indicators['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands - only if any BB indicator is needed
        bb_indicators = {'bb_upper', 'bb_middle', 'bb_lower'}
        if bb_indicators.intersection(required_indicators):
            bb_data = self.ti.bollinger_bands(data['close'])
            if 'bb_upper' in required_indicators:
                indicators['bb_upper'] = bb_data['upper']
            if 'bb_middle' in required_indicators:
                indicators['bb_middle'] = bb_data['middle']
            if 'bb_lower' in required_indicators:
                indicators['bb_lower'] = bb_data['lower']
        
        # Volume indicators - only if needed
        if 'volume' in required_indicators:
            indicators['volume'] = data['volume']
        if 'volume_ma' in required_indicators:
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
        if condition.operator == OperatorType.ABOVE:
            return indicator_value > compare_value
        elif condition.operator == OperatorType.BELOW:
            return indicator_value < compare_value
        elif condition.operator == OperatorType.EQUALS:
            return abs(indicator_value - compare_value) < 1e-10
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
    
    def _extract_required_indicators(self, conditions: List[Dict[str, Any]]) -> set:
        """Extract all indicator names that are needed based on the conditions"""
        required_indicators = set()
        
        for condition in conditions:
            # Add the main indicator
            indicator_name = condition.get('name', '')
            required_indicators.add(indicator_name)
            
            # Check if the value references another indicator
            value = condition.get('value', '')
            if isinstance(value, str) and value != 'price':
                # This might be a reference to another indicator
                required_indicators.add(value)
        
        # Always include price as it's fundamental
        required_indicators.add('price')
        
        return required_indicators
    
    def generate_signals(self, data: pd.DataFrame, buy_conditions: List[Dict[str, Any]], 
                        sell_conditions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate buy/sell signals based on custom rules"""
        
        # Extract required indicators from both buy and sell conditions
        buy_required = self._extract_required_indicators(buy_conditions)
        sell_required = self._extract_required_indicators(sell_conditions)
        all_required = buy_required.union(sell_required)
        
        # Calculate only the required indicators
        self.calculate_indicators(data, all_required)
        
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