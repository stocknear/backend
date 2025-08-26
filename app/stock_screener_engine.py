"""
Enhanced Stock Screener Engine
Handles complex temporal and conditional queries for stock screening
"""

import asyncio
import aiofiles
import orjson
import json
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from pathlib import Path
import operator
import re
from dataclasses import dataclass, field
import numpy as np

# Define base directory
BASE_DIR = Path(__file__).parent / "json"

# Operators mapping
OPERATORS = {
    '>': operator.gt,
    '>=': operator.ge, 
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '=': operator.eq,
    '!=': operator.ne,
    'above': operator.gt,
    'below': operator.lt,
    'over': operator.gt,
    'under': operator.lt,
    'between': lambda x, v: v[0] <= x <= v[1] if isinstance(v, (list, tuple)) and len(v) == 2 else False,
    'exactly': operator.eq,
    'not': operator.ne,
}

# Time period mappings
TIME_PERIODS = {
    'past_day': 1,
    'past_week': 7,
    'past_month': 30,
    'past_3_months': 90,
    'past_6_months': 180,
    'past_year': 365,
    'past_2_years': 730,
    'past_5_years': 1825,
    '1d': 1,
    '1w': 7,
    '1m': 30,
    '3m': 90,
    '6m': 180,
    '1y': 365,
    '2y': 730,
    '5y': 1825,
}

@dataclass
class TemporalCondition:
    """Represents a time-based condition for stock screening"""
    metric: str
    start_condition: Dict[str, Any]  # e.g., {'operator': '<', 'value': 5}
    end_condition: Dict[str, Any]    # e.g., {'operator': '>', 'value': 5}
    time_period: str  # e.g., 'past_year'
    duration_days: Optional[int] = None  # minimum days condition must be met
    
@dataclass
class ScreenerRule:
    """Enhanced screener rule with support for complex conditions"""
    metric: str
    operator: str
    value: Any
    rule_type: str = 'simple'  # 'simple', 'temporal', 'compound'
    temporal_condition: Optional[TemporalCondition] = None
    sub_rules: List['ScreenerRule'] = field(default_factory=list)
    logical_operator: str = 'AND'  # 'AND' or 'OR' for compound rules

class StockScreenerEngine:
    """Enhanced stock screener with temporal and complex query support"""
    
    def __init__(self):
        self.base_dir = BASE_DIR
        self.stock_data_cache = {}
        self.historical_data_cache = {}
        
    async def load_stock_data(self) -> List[Dict]:
        """Load current stock screener data"""
        if 'current' in self.stock_data_cache:
            return self.stock_data_cache['current']
            
        file_path = self.base_dir / "stock-screener/data.json"
        async with aiofiles.open(file_path, 'rb') as file:
            data = orjson.loads(await file.read())
            # Filter out OTC stocks
            filtered_data = [item for item in data if item.get('exchange') != 'OTC']
            self.stock_data_cache['current'] = filtered_data
            return filtered_data
    
    async def load_historical_prices(self, symbol: str, period: str = 'one-year') -> Optional[List[Dict]]:
        """Load historical price data for a symbol"""
        cache_key = f"{symbol}_{period}"
        if cache_key in self.historical_data_cache:
            return self.historical_data_cache[cache_key]
        
        # Map period to directory
        period_map = {
            'one-week': 'one-week',
            'one-month': 'one-month', 
            'six-months': 'six-months',
            'one-year': 'one-year',
            'five-years': 'five-years',
            'max': 'max'
        }
        
        period_dir = period_map.get(period, 'one-year')
        file_path = self.base_dir / f"historical-price/{period_dir}/{symbol}.json"
        
        try:
            if file_path.exists():
                async with aiofiles.open(file_path, 'rb') as file:
                    data = orjson.loads(await file.read())
                    self.historical_data_cache[cache_key] = data
                    return data
        except Exception as e:
            print(f"Error loading historical data for {symbol}: {e}")
        
        return None
    
    async def check_temporal_condition(self, symbol: str, condition: TemporalCondition) -> bool:
        """Check if a stock meets a temporal condition"""
        # Determine the period to load based on time_period
        days = TIME_PERIODS.get(condition.time_period, 365)
        
        if days <= 30:
            period = 'one-month'
        elif days <= 180:
            period = 'six-months'
        elif days <= 365:
            period = 'one-year'
        else:
            period = 'five-years'
        
        # Load historical data
        historical_data = await self.load_historical_prices(symbol, period)
        if not historical_data:
            return False
        
        # Get relevant time window
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter data to time window
        relevant_data = []
        for point in historical_data:
            if 'date' in point:
                try:
                    point_date = datetime.strptime(point['date'], '%Y-%m-%d')
                    if start_date <= point_date <= end_date:
                        relevant_data.append(point)
                except:
                    continue
        
        if not relevant_data:
            return False
        
        # Check temporal condition
        metric_key = 'close' if condition.metric == 'price' else condition.metric
        
        # Find periods where start condition was met
        start_met_periods = []
        for i, point in enumerate(relevant_data):
            value = point.get(metric_key)
            if value is not None:
                start_op = OPERATORS.get(condition.start_condition['operator'], operator.eq)
                if start_op(value, condition.start_condition['value']):
                    start_met_periods.append(i)
        
        if not start_met_periods:
            return False
        
        # Check if end condition was met after start condition
        for start_idx in start_met_periods:
            # Look forward from this point
            for point in relevant_data[start_idx + 1:]:
                value = point.get(metric_key)
                if value is not None:
                    end_op = OPERATORS.get(condition.end_condition['operator'], operator.eq)
                    if end_op(value, condition.end_condition['value']):
                        # Check duration requirement if specified
                        if condition.duration_days:
                            start_date = datetime.strptime(relevant_data[start_idx]['date'], '%Y-%m-%d')
                            end_date = datetime.strptime(point['date'], '%Y-%m-%d')
                            if (end_date - start_date).days >= condition.duration_days:
                                return True
                        else:
                            return True
        
        return False
    
    def check_simple_rule(self, stock: Dict, rule: ScreenerRule) -> bool:
        """Check if stock meets a simple screening rule"""
        metric_value = stock.get(rule.metric)
        if metric_value is None:
            return False
        
        op = OPERATORS.get(rule.operator, operator.eq)
        try:
            return op(metric_value, rule.value)
        except (TypeError, ValueError):
            return False
    
    async def check_rule(self, stock: Dict, rule: ScreenerRule) -> bool:
        """Check if stock meets a screening rule (simple or temporal)"""
        if rule.rule_type == 'simple':
            return self.check_simple_rule(stock, rule)
        
        elif rule.rule_type == 'temporal':
            if rule.temporal_condition:
                symbol = stock.get('symbol')
                if symbol:
                    return await self.check_temporal_condition(symbol, rule.temporal_condition)
            return False
        
        elif rule.rule_type == 'compound':
            results = []
            for sub_rule in rule.sub_rules:
                result = await self.check_rule(stock, sub_rule)
                results.append(result)
            
            if rule.logical_operator == 'AND':
                return all(results)
            elif rule.logical_operator == 'OR':
                return any(results)
        
        return False
    
    async def screen_stocks(
        self,
        rules: List[ScreenerRule],
        sort_by: Optional[str] = None,
        sort_order: str = 'desc',
        limit: int = 100
    ) -> Dict[str, Any]:
        """Screen stocks based on provided rules"""
        # Load current stock data
        stocks = await self.load_stock_data()
        
        # Filter stocks based on rules
        matched_stocks = []
        
        # Process in batches for better performance with async operations
        batch_size = 50
        for i in range(0, len(stocks), batch_size):
            batch = stocks[i:i + batch_size]
            
            # Check each stock in parallel
            tasks = []
            for stock in batch:
                async def check_stock(s):
                    for rule in rules:
                        if not await self.check_rule(s, rule):
                            return None
                    return s
                
                tasks.append(check_stock(stock))
            
            results = await asyncio.gather(*tasks)
            matched_stocks.extend([r for r in results if r is not None])
        
        # Sort if requested
        if sort_by and matched_stocks and sort_by in matched_stocks[0]:
            matched_stocks.sort(
                key=lambda x: (x.get(sort_by) is None, x.get(sort_by)),
                reverse=(sort_order.lower() == 'desc')
            )
        
        # Apply limit
        if limit:
            matched_stocks = matched_stocks[:limit]
        
        # Format results
        formatted_results = []
        for stock in matched_stocks:
            result = {
                'symbol': stock.get('symbol'),
                'name': stock.get('name'),
                'price': stock.get('price'),
                'marketCap': stock.get('marketCap'),
                'exchange': stock.get('exchange'),
                'sector': stock.get('sector'),
                'industry': stock.get('industry'),
            }
            
            # Add requested metrics
            for rule in rules:
                if rule.metric in stock:
                    result[rule.metric] = stock[rule.metric]
            
            formatted_results.append(result)
        
        return {
            'matched_stocks': formatted_results,
            'total_matches': len(formatted_results),
            'query_time': datetime.now().isoformat()
        }
    
    def parse_natural_language_query(self, query: str) -> List[ScreenerRule]:
        """Parse natural language query into screener rules"""
        rules = []
        query_lower = query.lower()
        
        # Example patterns for temporal queries
        temporal_patterns = [
            # Pattern: "moved from below X to above Y"
            (r'moved from below \$?(\d+(?:\.\d+)?)\s*(?:per share)?\s*to above \$?(\d+(?:\.\d+)?)', 'price_movement'),
            # Pattern: "crossed above/below X"
            (r'crossed (above|below) \$?(\d+(?:\.\d+)?)', 'price_cross'),
            # Pattern: "increased/decreased by X%"
            (r'(increased|decreased) by (\d+(?:\.\d+)?)%', 'percentage_change'),
        ]
        
        # Check for temporal patterns
        for pattern, pattern_type in temporal_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if pattern_type == 'price_movement':
                    below_value = float(match.group(1))
                    above_value = float(match.group(2))
                    
                    # Determine time period
                    time_period = 'past_year'  # default
                    for period_key, days in TIME_PERIODS.items():
                        if period_key.replace('_', ' ') in query_lower:
                            time_period = period_key
                            break
                    
                    # Check for duration requirement
                    duration_match = re.search(r'for (?:at least )?(\d+) days?', query_lower)
                    duration_days = int(duration_match.group(1)) if duration_match else 1
                    
                    temporal_condition = TemporalCondition(
                        metric='price',
                        start_condition={'operator': '<', 'value': below_value},
                        end_condition={'operator': '>', 'value': above_value},
                        time_period=time_period,
                        duration_days=duration_days
                    )
                    
                    rules.append(ScreenerRule(
                        metric='price',
                        operator='temporal',
                        value=None,
                        rule_type='temporal',
                        temporal_condition=temporal_condition
                    ))
        
        # Add current price constraint if mentioned
        if 'current price' in query_lower:
            price_match = re.search(r'current price\s*(above|below|over|under|between)?\s*\$?(\d+(?:\.\d+)?)', query_lower)
            if price_match:
                op = price_match.group(1) or '>'
                value = float(price_match.group(2))
                rules.append(ScreenerRule(
                    metric='price',
                    operator=op,
                    value=value,
                    rule_type='simple'
                ))
        
        # Add market cap constraints
        if 'market cap' in query_lower or 'billion' in query_lower or 'million' in query_lower:
            cap_match = re.search(r'(\d+(?:\.\d+)?)\s*(billion|million)', query_lower)
            if cap_match:
                value = float(cap_match.group(1))
                multiplier = 1e9 if cap_match.group(2) == 'billion' else 1e6
                rules.append(ScreenerRule(
                    metric='marketCap',
                    operator='>',
                    value=value * multiplier,
                    rule_type='simple'
                ))
        
        # Add volume constraints
        if 'volume' in query_lower:
            vol_match = re.search(r'volume\s*(above|over|greater than)?\s*(\d+(?:\.\d+)?)\s*(million|k)?', query_lower)
            if vol_match:
                value = float(vol_match.group(2))
                if vol_match.group(3) == 'million':
                    value *= 1e6
                elif vol_match.group(3) == 'k':
                    value *= 1e3
                rules.append(ScreenerRule(
                    metric='avgVolume',
                    operator='>',
                    value=value,
                    rule_type='simple'
                ))
        
        # Add sector/industry filters
        sectors = ['technology', 'healthcare', 'financial', 'energy', 'consumer', 'industrial']
        for sector in sectors:
            if sector in query_lower:
                rules.append(ScreenerRule(
                    metric='sector',
                    operator='==',
                    value=sector.capitalize(),
                    rule_type='simple'
                ))
        
        return rules

# Export the engine
screener_engine = StockScreenerEngine()