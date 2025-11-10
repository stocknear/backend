"""
Python translation of the frontend stock screener filterWorker.ts
Matches the exact logic and behavior of the frontend implementation
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable
import json
from pathlib import Path

# Lists from frontend utils (these should match the frontend exactly)
SECTOR_LIST = [
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical", 
    "Industrials", "Communication Services", "Consumer Defensive", "Energy", 
    "Utilities", "Real Estate", "Basic Materials"
]

INDUSTRY_LIST = [
    "Software—Application", "Biotechnology", "Banks—Regional", "Auto Manufacturers",
    "Semiconductors", "Drug Manufacturers—General", "Software—Infrastructure",
    "Banks—Diversified", "Aerospace & Defense", "Oil & Gas E&P", "Utilities—Regulated Electric"
    # Add more as needed - this should match frontend exactly
]

COUNTRIES_LIST = [
    "United States", "China", "Japan", "Germany", "United Kingdom", 
    "France", "India", "Italy", "Brazil", "Canada", "Russia", "South Korea"
    # Add more as needed
]

def generate_moving_average_conditions():
    """Generate moving average conditions matching frontend logic"""
    conditions = {}
    periods = [20, 50, 100, 200]
    ma_types = ['ema', 'sma']
    
    # Generate conditions for each MA type
    for ma_type in ma_types:
        ma_type_upper = ma_type.upper()
        
        # Price above MA conditions
        for period in periods:
            key = f"Price above {ma_type_upper}{period}"
            conditions[key] = lambda item, mt=ma_type, p=period: item.get('price', 0) > item.get(f'{mt}{p}', 0)
        
        # Price below MA conditions
        for period in periods:
            key = f"Price below {ma_type_upper}{period}"
            conditions[key] = lambda item, mt=ma_type, p=period: item.get('price', 0) < item.get(f'{mt}{p}', 0)
        
        # MA cross conditions
        for i, period1 in enumerate(periods):
            for j, period2 in enumerate(periods):
                if i != j:
                    key = f"{ma_type_upper}{period1} above {ma_type_upper}{period2}"
                    conditions[key] = lambda item, mt=ma_type, p1=period1, p2=period2: item.get(f'{mt}{p1}', 0) > item.get(f'{mt}{p2}', 0)
    
    return conditions

# Generate moving average conditions
MOVING_AVERAGE_CONDITIONS = {
    **generate_moving_average_conditions(),
    "Price > Graham Number": lambda item: item.get('price', 0) > item.get('grahamNumber', 0),
    "Price < Graham Number": lambda item: item.get('price', 0) < item.get('grahamNumber', 0),
    "Price > Lynch Fair Value": lambda item: item.get('price', 0) > item.get('lynchFairValue', 0),
    "Price < Lynch Fair Value": lambda item: item.get('price', 0) < item.get('lynchFairValue', 0),
}

def convert_unit_to_value(input_val: Union[str, float, int, List]) -> Any:
    """
    Convert units to values exactly matching frontend logic
    """
    try:
        if isinstance(input_val, list):
            return [convert_unit_to_value(item) for item in input_val]
        
        if isinstance(input_val, (int, float)):
            return input_val
        
        if not isinstance(input_val, str):
            return input_val
        
        lower_input = input_val.lower()
        
        # Non-numeric values that should be returned as-is
        non_numeric_values = {
            "any",
            *[s.lower() for s in SECTOR_LIST],
            *[i.lower() for i in INDUSTRY_LIST],
            *[c.lower() for c in COUNTRIES_LIST],
            'before market open',
            'after market close',
            'quarterly',
            'monthly',
            'annual',
            'semi-annual',
            "hold",
            "sell", 
            "buy",
            "strong buy",
            "strong sell",
            "compliant",
            "non-compliant",
            "stock price",
        }
        
        if lower_input in non_numeric_values:
            return input_val
        
        # Handle percentage values
        if input_val.endswith("%"):
            numeric_value = float(input_val[:-1])
            return numeric_value  # Frontend doesn't divide by 100
        
        # Handle units (B, M, K)
        units = {'B': 1_000_000_000, 'M': 1_000_000, 'K': 1_000}
        match = re.match(r'^(-?\d+(?:\.\d+)?)([BMK])?$', input_val)
        
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            return value * units.get(unit, 1) if unit else value
        
        # Default numeric conversion
        try:
            return float(input_val)
        except ValueError:
            return input_val
            
    except Exception as e:
        print(f"Error converting value: {input_val}, error: {e}")
        return input_val

def create_rule_check(rule: Dict, rule_name: str, rule_value: Any) -> Callable:
    """
    Create rule checking function matching frontend logic exactly
    """
    # Handle 'any' condition quickly
    if rule.get('value') == 'any':
        return lambda item: True
    
    # Earnings date handling
    if rule.get('name') == 'earningsDate':
        return create_earnings_date_check(rule, rule_value)
    
    # Categorical field checks
    categorical_fields = [
        'analystRating', 'topAnalystRating', 'earningsTime', 'halalStocks', 'score',
        'sector', 'industry', 'country', 'payoutFrequency'
    ]
    
    if rule.get('name') in categorical_fields:
        def categorical_check(item):
            item_value = item.get(rule['name'])
            if isinstance(rule_value, list):
                return item_value in rule_value
            return item_value == rule_value
        return categorical_check
    
    # Moving average field checks
    moving_average_fields = [
        'ema20', 'ema50', 'ema100', 'ema200',
        'sma20', 'sma50', 'sma100', 'sma200',
        'grahamnumber', 'lynchfairvalue'
    ]
    
    if rule_name in moving_average_fields:
        def ma_check(item):
            if isinstance(rule_value, list):
                return all(
                    MOVING_AVERAGE_CONDITIONS.get(condition, lambda x: True)(item)
                    for condition in rule_value
                )
            return True
        return ma_check
    
    # Between condition
    if rule.get('condition') == 'between' and isinstance(rule_value, list):
        def between_check(item):
            item_value = item.get(rule['name'])
            if item_value is None:
                return False
                
            converted_values = [convert_unit_to_value(v) for v in rule_value]
            min_val, max_val = converted_values
            
            # Handle empty/undefined min and max
            if (min_val in ('', None) and max_val in ('', None)):
                return True
            
            if min_val in ('', None):
                return item_value < max_val
            
            if max_val in ('', None):
                return item_value > min_val
            
            return min_val < item_value < max_val
        return between_check
    
    # Default numeric comparisons
    def numeric_check(item):
        item_value = item.get(rule['name'])
        if item_value is None or rule_value is None:
            return False
        
        condition = rule.get('condition')
        
        try:
            if condition == 'exactly' and item_value != rule_value:
                return False
            if condition == 'over' and item_value <= rule_value:
                return False
            if condition == 'under' and item_value > rule_value:
                return False
        except (TypeError, ValueError):
            return False
        
        return True
    
    return numeric_check

def create_earnings_date_check(rule: Dict, rule_value: Any) -> Callable:
    """Create earnings date check matching frontend logic"""
    # Get current date in UTC
    now = datetime.utcnow()
    today_utc = datetime(now.year, now.month, now.day)
    
    def fmt_date(d: datetime) -> str:
        return d.strftime('%Y-%m-%d')
    
    # Pre-compute ranges for each label
    ranges = {
        'today': (fmt_date(today_utc), fmt_date(today_utc)),
        'tomorrow': (
            fmt_date(today_utc + timedelta(days=1)),
            fmt_date(today_utc + timedelta(days=1))
        ),
        'next 7d': (
            fmt_date(today_utc),
            fmt_date(today_utc + timedelta(days=6))
        ),
        'next 30d': (
            fmt_date(today_utc),
            fmt_date(today_utc + timedelta(days=29))
        ),
        'this month': (
            fmt_date(today_utc.replace(day=1)),
            fmt_date((today_utc.replace(month=today_utc.month+1) if today_utc.month < 12 else today_utc.replace(year=today_utc.year+1, month=1)).replace(day=1) - timedelta(days=1))
        ),
        'next month': (
            fmt_date((today_utc.replace(month=today_utc.month+1) if today_utc.month < 12 else today_utc.replace(year=today_utc.year+1, month=1)).replace(day=1)),
            fmt_date((today_utc.replace(month=today_utc.month+2) if today_utc.month < 11 else today_utc.replace(year=today_utc.year+1, month=today_utc.month-10)).replace(day=1) - timedelta(days=1))
        )
    }
    
    # Handle both single string and array
    labels = [str(rule_value).strip().lower()] if not isinstance(rule_value, list) else [str(v).strip().lower() for v in rule_value]
    
    # Find widest date range
    min_date = '9999-12-31'
    max_date = '0000-01-01'
    
    for label in labels:
        if label in ranges:
            start, end = ranges[label]
            if start < min_date:
                min_date = start
            if end > max_date:
                max_date = end
    
    if min_date == '9999-12-31' or max_date == '0000-01-01':
        return lambda item: True
    
    def earnings_check(item):
        earnings_date = item.get('earningsDate')
        if not earnings_date:
            return False
        
        try:
            # Parse the earnings date
            if isinstance(earnings_date, str):
                date_obj = datetime.fromisoformat(earnings_date.replace('Z', '+00:00'))
            else:
                date_obj = earnings_date
            
            item_date_str = date_obj.strftime('%Y-%m-%d')
            return min_date <= item_date_str <= max_date
        except:
            return False
    
    return earnings_check

async def filter_stock_screener_data(stock_screener_data: List[Dict], rule_of_list: List[Dict]) -> List[Dict]:
    """
    Filter stock screener data exactly matching frontend logic
    """
    # Early return if no data or no rules
    if not stock_screener_data or not rule_of_list:
        return stock_screener_data or []
    
    # Precompile rule conditions
    compiled_rules = []
    for rule in rule_of_list:
        rule_name = rule.get('name', '').lower()
        rule_value = convert_unit_to_value(rule.get('value'))
        
        compiled_rules.append({
            **rule,
            'compiledCheck': create_rule_check(rule, rule_name, rule_value)
        })
    
    # Filter data using compiled rules
    filtered_data = []
    for item in stock_screener_data:
        if all(rule['compiledCheck'](item) for rule in compiled_rules):
            filtered_data.append(item)
    
    # Sort by market cap descending (matching frontend)
    filtered_data.sort(key=lambda x: (x.get('marketCap') is None, x.get('marketCap', 0)), reverse=True)
    
    return filtered_data

class PythonStockScreener:
    """Python version of the stock screener matching frontend behavior exactly"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else Path(__file__).parent.parent / "json" / "all-symbols" / "stocks.json"
        self.stock_data = None
    
    async def load_data(self):
        """Load stock screener data"""
        if self.stock_data is None:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                # Filter out OTC stocks like frontend
                self.stock_data = [item for item in data if item.get('exchange') != 'OTC']
        return self.stock_data
    
    async def screen(self, rules: List[Dict], limit: Optional[int] = None) -> Dict:
        """Screen stocks using the provided rules"""
        stock_data = await self.load_data()
        
        # Convert rules to match expected format
        formatted_rules = []
        rule_names = set()  # Track rule names for output filtering
        
        for rule in rules:
            # Handle both old and new rule formats
            if isinstance(rule, dict):
                # If it's already in the right format, use it
                if 'name' in rule and rule.get('value') is not None:
                    formatted_rules.append(rule)
                    rule_names.add(rule.get('name'))
                # Convert from simple format
                elif 'metric' in rule and rule.get('value') is not None:
                    formatted_rules.append({
                        'name': rule.get('metric'),
                        'value': rule.get('value'),
                        'condition': rule.get('operator', 'over').replace('>', 'over').replace('<', 'under').replace('==', 'exactly')
                    })
                    rule_names.add(rule.get('metric'))
        
        # Filter the data
        filtered_data = await filter_stock_screener_data(stock_data, formatted_rules)
        
        # Always include these essential fields
        essential_fields = {'symbol', 'name', 'price', 'changesPercentage', 'marketCap', 'volume'}
        
        # Combine essential fields with rule-based fields
        output_fields = essential_fields | rule_names
        
        # Filter each stock item to include only relevant fields
        filtered_output = []
        for stock in filtered_data:
            filtered_stock = {}
            for field in output_fields:
                if field in stock:
                    filtered_stock[field] = stock[field]
            filtered_output.append(filtered_stock)
        
        # Apply limit if specified
        if limit and limit > 0:
            filtered_output = filtered_output[:limit]
        
        return {
            'matched_stocks': filtered_output,
            'total_matches': len(filtered_data),  # Use original filtered count for total
            'original_data_length': len(stock_data),
            'query_time': datetime.now().isoformat()
        }

# Create global instance
python_screener = PythonStockScreener()

# Export main functions
__all__ = ['python_screener', 'filter_stock_screener_data', 'convert_unit_to_value', 'create_rule_check']