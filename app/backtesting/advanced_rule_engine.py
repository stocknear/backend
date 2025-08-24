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
        
        # RSI
        if 'rsi' in required_indicators:
            indicators['rsi'] = self.ti.rsi(data['close'])
        
        # SMA - multiple windows
        ma_windows = [int(ind.split('_')[1]) for ind in required_indicators if ind.startswith('sma_')]
        for window in ma_windows:
            indicators[f'sma_{window}'] = self.ti.sma(data['close'], window)
        
        # EMA - multiple windows
        ema_windows = [int(ind.split('_')[1]) for ind in required_indicators if ind.startswith('ema_')]
        for window in ema_windows:
            indicators[f'ema_{window}'] = self.ti.ema(data['close'], window)
        
        # MACD
        macd_indicators = {'macd', 'macd_signal', 'macd_histogram'}
        if macd_indicators.intersection(required_indicators):
            macd_data = self.ti.macd(data['close'])
            if 'macd' in required_indicators:
                indicators['macd'] = macd_data['macd']
            if 'macd_signal' in required_indicators:
                indicators['macd_signal'] = macd_data['signal']
            if 'macd_histogram' in required_indicators:
                indicators['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_indicators = {'bb_upper', 'bb_middle', 'bb_lower'}
        if bb_indicators.intersection(required_indicators):
            bb_data = self.ti.bollinger_bands(data['close'])
            if 'bb_upper' in required_indicators:
                indicators['bb_upper'] = bb_data['upper']
            if 'bb_middle' in required_indicators:
                indicators['bb_middle'] = bb_data['middle']
            if 'bb_lower' in required_indicators:
                indicators['bb_lower'] = bb_data['lower']
        
        # ATR base indicator (still needed for multiplier bands)
        if 'atr' in required_indicators:
            indicators['atr'] = self.ti.atr(data['high'], data['low'], data['close'])
        
        # ATR Multiplier Bands for breakout signals
        # Extract multiplier values from indicator names like 'atr_upper_1.5' or 'atr_lower_2'
        atr_upper_indicators = [ind for ind in required_indicators if ind.startswith('atr_upper_')]
        atr_lower_indicators = [ind for ind in required_indicators if ind.startswith('atr_lower_')]
        
        if atr_upper_indicators or atr_lower_indicators:
            atr = self.ti.atr(data['high'], data['low'], data['close'])
            prev_close = data['close'].shift(1)
            
            # Upper ATR bands (for buy signals)
            for indicator in atr_upper_indicators:
                try:
                    multiplier = float(indicator.split('_')[2])
                    indicators[indicator] = prev_close + (atr * multiplier)
                except (IndexError, ValueError):
                    pass
            
            # Lower ATR bands (for sell signals)
            for indicator in atr_lower_indicators:
                try:
                    multiplier = float(indicator.split('_')[2])
                    indicators[indicator] = prev_close - (atr * multiplier)
                except (IndexError, ValueError):
                    pass
        
        # ADX
        if 'adx' in required_indicators:
            indicators['adx'] = self.ti.adx(data['high'], data['low'], data['close'])
        
        # Stochastic Oscillator
        stochastic_indicators = {'stoch_k', 'stoch_d', 'stoch_k_oversold', 'stoch_k_overbought', 
                                'stoch_d_oversold', 'stoch_d_overbought', 'stoch_crossover'}
        if stochastic_indicators.intersection(required_indicators):
            stoch_data = self.ti.stochastic_oscillator(data['high'], data['low'], data['close'])
            if 'stoch_k' in required_indicators:
                indicators['stoch_k'] = stoch_data['k_percent']
            if 'stoch_d' in required_indicators:
                indicators['stoch_d'] = stoch_data['d_percent']
            
            # Oversold/Overbought levels for %K
            if 'stoch_k_oversold' in required_indicators:
                indicators['stoch_k_oversold'] = 20  # Static level for comparison
            if 'stoch_k_overbought' in required_indicators:
                indicators['stoch_k_overbought'] = 80  # Static level for comparison
                
            # Oversold/Overbought levels for %D
            if 'stoch_d_oversold' in required_indicators:
                indicators['stoch_d_oversold'] = 20  # Static level for comparison
            if 'stoch_d_overbought' in required_indicators:
                indicators['stoch_d_overbought'] = 80  # Static level for comparison
            
            # Crossover indicator (positive when %K > %D, negative when %K < %D)
            if 'stoch_crossover' in required_indicators:
                indicators['stoch_crossover'] = stoch_data['k_percent'] - stoch_data['d_percent']
        
        # CCI
        if 'cci' in required_indicators:
            indicators['cci'] = self.ti.cci(data['high'], data['low'], data['close'])
        
        # OBV
        if 'obv' in required_indicators:
            indicators['obv'] = self.ti.obv(data['close'], data['volume'])
            
        # OBV with moving averages (e.g., obv_sma_10, obv_sma_20)
        obv_sma_indicators = [ind for ind in required_indicators if ind.startswith('obv_sma_')]
        if obv_sma_indicators:
            if 'obv' not in indicators:
                indicators['obv'] = self.ti.obv(data['close'], data['volume'])
            
            for indicator in obv_sma_indicators:
                try:
                    window = int(indicator.split('_')[2])
                    indicators[indicator] = self.ti.sma(indicators['obv'], window)
                except (IndexError, ValueError):
                    pass
        
        # VWAP
        if 'vwap' in required_indicators:
            indicators['vwap'] = self.ti.vwap(data['high'], data['low'], data['close'], data['volume'])
        
        # Volume indicators
        if 'volume' in required_indicators:
            indicators['volume'] = data['volume']
        if 'volume_ma' in required_indicators:
            indicators['volume_ma'] = data['volume'].rolling(window=20).mean()
        
        # Williams %R
        williams_r_indicators = [ind for ind in required_indicators if ind.startswith('williams_r')]
        for indicator in williams_r_indicators:
            if indicator == 'williams_r':
                # Default 14-period Williams %R
                indicators['williams_r'] = self.ti.williams_r(data['high'], data['low'], data['close'])
            else:
                # Custom period Williams %R (e.g., williams_r_10)
                try:
                    window = int(indicator.split('_')[2])
                    indicators[indicator] = self.ti.williams_r(data['high'], data['low'], data['close'], window)
                except (IndexError, ValueError):
                    pass
        
        # Money Flow Index (MFI)
        mfi_indicators = [ind for ind in required_indicators if ind.startswith('mfi')]
        for indicator in mfi_indicators:
            if indicator == 'mfi':
                # Default 14-period MFI
                indicators['mfi'] = self.ti.mfi(data['high'], data['low'], data['close'], data['volume'])
            else:
                # Custom period MFI (e.g., mfi_10)
                try:
                    window = int(indicator.split('_')[1])
                    indicators[indicator] = self.ti.mfi(data['high'], data['low'], data['close'], data['volume'], window)
                except (IndexError, ValueError):
                    pass
        
        # Parabolic SAR
        if 'parabolic_sar' in required_indicators:
            indicators['parabolic_sar'] = self.ti.parabolic_sar(data['high'], data['low'], data['close'])
        
        # Donchian Channels
        donchian_indicators = {'donchian_upper', 'donchian_middle', 'donchian_lower'}
        if donchian_indicators.intersection(required_indicators):
            # Check for custom window (e.g., donchian_upper_50)
            custom_donchian = [ind for ind in required_indicators if ind.startswith('donchian_') and len(ind.split('_')) > 2]
            if custom_donchian:
                # Handle custom windows
                donchian_windows = set()
                for ind in custom_donchian:
                    try:
                        window = int(ind.split('_')[2])
                        donchian_windows.add(window)
                    except (IndexError, ValueError):
                        pass
                
                for window in donchian_windows:
                    donchian_data = self.ti.donchian_channels(data['high'], data['low'], window)
                    if f'donchian_upper_{window}' in required_indicators:
                        indicators[f'donchian_upper_{window}'] = donchian_data['upper']
                    if f'donchian_middle_{window}' in required_indicators:
                        indicators[f'donchian_middle_{window}'] = donchian_data['middle']
                    if f'donchian_lower_{window}' in required_indicators:
                        indicators[f'donchian_lower_{window}'] = donchian_data['lower']
            else:
                # Default 20-period Donchian Channels
                donchian_data = self.ti.donchian_channels(data['high'], data['low'])
                if 'donchian_upper' in required_indicators:
                    indicators['donchian_upper'] = donchian_data['upper']
                if 'donchian_middle' in required_indicators:
                    indicators['donchian_middle'] = donchian_data['middle']
                if 'donchian_lower' in required_indicators:
                    indicators['donchian_lower'] = donchian_data['lower']
        
        # Standard Deviation (Volatility)
        if 'std' in required_indicators:
            # Default 20-period standard deviation
            indicators['std'] = self.ti.standard_deviation(data['close'])
        
        # Custom period standard deviation (e.g., std_10, std_30)
        std_custom_indicators = [ind for ind in required_indicators if ind.startswith('std_')]
        for indicator in std_custom_indicators:
            try:
                window = int(indicator.split('_')[1])
                indicators[indicator] = self.ti.standard_deviation(data['close'], window)
            except (IndexError, ValueError):
                pass
        
        # Historical Volatility
        if 'hist_vol' in required_indicators:
            # Default 20-period historical volatility
            indicators['hist_vol'] = self.ti.historical_volatility(data['close'])
        
        # Custom period historical volatility (e.g., hist_vol_30)
        hist_vol_custom_indicators = [ind for ind in required_indicators if ind.startswith('hist_vol_')]
        for indicator in hist_vol_custom_indicators:
            try:
                window = int(indicator.split('_')[2])
                indicators[indicator] = self.ti.historical_volatility(data['close'], window)
            except (IndexError, ValueError):
                pass
        
        # Chaikin Volatility
        if 'chaikin_vol' in required_indicators:
            # Default Chaikin volatility
            indicators['chaikin_vol'] = self.ti.chaikin_volatility(data['high'], data['low'])
        
        # Custom parameters Chaikin volatility (e.g., chaikin_vol_14_7)
        chaikin_vol_custom_indicators = [ind for ind in required_indicators if ind.startswith('chaikin_vol_')]
        for indicator in chaikin_vol_custom_indicators:
            try:
                parts = indicator.split('_')
                if len(parts) >= 3:
                    window = int(parts[2])
                    period = int(parts[3]) if len(parts) > 3 else 10
                    indicators[indicator] = self.ti.chaikin_volatility(data['high'], data['low'], window, period)
            except (IndexError, ValueError):
                pass
        
        # Pivot Points
        pivot_indicators = {'pivot', 'pivot_r1', 'pivot_r2', 'pivot_r3', 'pivot_s1', 'pivot_s2', 'pivot_s3'}
        if pivot_indicators.intersection(required_indicators):
            pivot_data = self.ti.pivot_points(data['high'], data['low'], data['close'])
            if 'pivot' in required_indicators:
                indicators['pivot'] = pivot_data['pivot']
            if 'pivot_r1' in required_indicators:
                indicators['pivot_r1'] = pivot_data['r1']
            if 'pivot_r2' in required_indicators:
                indicators['pivot_r2'] = pivot_data['r2']
            if 'pivot_r3' in required_indicators:
                indicators['pivot_r3'] = pivot_data['r3']
            if 'pivot_s1' in required_indicators:
                indicators['pivot_s1'] = pivot_data['s1']
            if 'pivot_s2' in required_indicators:
                indicators['pivot_s2'] = pivot_data['s2']
            if 'pivot_s3' in required_indicators:
                indicators['pivot_s3'] = pivot_data['s3']
        
        # Fibonacci Retracements
        fib_indicators = {'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'fib_high', 'fib_low'}
        if fib_indicators.intersection(required_indicators):
            # Default uptrend fibonacci
            fib_data = self.ti.fibonacci_retracements(data['high'], data['low'])
            if 'fib_236' in required_indicators:
                indicators['fib_236'] = fib_data['fib_236']
            if 'fib_382' in required_indicators:
                indicators['fib_382'] = fib_data['fib_382']
            if 'fib_500' in required_indicators:
                indicators['fib_500'] = fib_data['fib_500']
            if 'fib_618' in required_indicators:
                indicators['fib_618'] = fib_data['fib_618']
            if 'fib_786' in required_indicators:
                indicators['fib_786'] = fib_data['fib_786']
            if 'fib_high' in required_indicators:
                indicators['fib_high'] = fib_data['fib_high']
            if 'fib_low' in required_indicators:
                indicators['fib_low'] = fib_data['fib_low']
        
        # Psychological Levels
        psych_indicators = [ind for ind in required_indicators if ind.startswith('psych_')]
        if psych_indicators:
            psych_data = self.ti.psychological_levels(data['close'])
            for indicator in psych_indicators:
                if indicator in psych_data:
                    indicators[indicator] = psych_data[indicator]
        
        # ROC (Rate of Change)
        if 'roc' in required_indicators:
            indicators['roc'] = self.ti.roc(data['close'])
        
        # Custom period ROC (e.g., roc_10, roc_20)
        roc_custom_indicators = [ind for ind in required_indicators if ind.startswith('roc_')]
        for indicator in roc_custom_indicators:
            try:
                window = int(indicator.split('_')[1])
                indicators[indicator] = self.ti.roc(data['close'], window)
            except (IndexError, ValueError):
                pass
        
        # TSI (True Strength Index)
        tsi_indicators = {'tsi', 'tsi_signal'}
        if tsi_indicators.intersection(required_indicators):
            tsi_data = self.ti.tsi(data['close'])
            if 'tsi' in required_indicators:
                indicators['tsi'] = tsi_data['tsi']
            if 'tsi_signal' in required_indicators:
                indicators['tsi_signal'] = tsi_data['signal']
        
        # Custom period TSI (e.g., tsi_13_25_13)
        tsi_custom_indicators = [ind for ind in required_indicators if ind.startswith('tsi_') and ind != 'tsi_signal']
        for indicator in tsi_custom_indicators:
            try:
                parts = indicator.split('_')
                if len(parts) >= 4:
                    fast = int(parts[1])
                    slow = int(parts[2])
                    signal = int(parts[3])
                    tsi_data = self.ti.tsi(data['close'], fast, slow, signal)
                    indicators[indicator] = tsi_data['tsi']
                    indicators[f'{indicator}_signal'] = tsi_data['signal']
            except (IndexError, ValueError):
                pass
        
        # Aroon Indicator
        aroon_indicators = {'aroon_up', 'aroon_down', 'aroon_oscillator'}
        if aroon_indicators.intersection(required_indicators):
            aroon_data = self.ti.aroon(data['high'], data['low'])
            if 'aroon_up' in required_indicators:
                indicators['aroon_up'] = aroon_data['aroon_up']
            if 'aroon_down' in required_indicators:
                indicators['aroon_down'] = aroon_data['aroon_down']
            if 'aroon_oscillator' in required_indicators:
                indicators['aroon_oscillator'] = aroon_data['aroon_oscillator']
        
        # Custom period Aroon (e.g., aroon_up_14, aroon_down_14)
        aroon_custom_indicators = [ind for ind in required_indicators if ind.startswith('aroon_') and 
                                  ind not in ['aroon_up', 'aroon_down', 'aroon_oscillator']]
        if aroon_custom_indicators:
            # Group by window period
            aroon_windows = set()
            for ind in aroon_custom_indicators:
                try:
                    parts = ind.split('_')
                    if len(parts) >= 3:
                        window = int(parts[2])
                        aroon_windows.add(window)
                except (IndexError, ValueError):
                    pass
            
            for window in aroon_windows:
                aroon_data = self.ti.aroon(data['high'], data['low'], window)
                if f'aroon_up_{window}' in required_indicators:
                    indicators[f'aroon_up_{window}'] = aroon_data['aroon_up']
                if f'aroon_down_{window}' in required_indicators:
                    indicators[f'aroon_down_{window}'] = aroon_data['aroon_down']
                if f'aroon_oscillator_{window}' in required_indicators:
                    indicators[f'aroon_oscillator_{window}'] = aroon_data['aroon_oscillator']
        
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