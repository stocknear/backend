"""
Enhanced Stock Screener Integration
Bridges the AI agent with the stock screener engine
"""

import os
import json
import orjson
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from stock_screener_python import python_screener
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enhanced extraction function with detailed examples
ENHANCED_EXTRACT_RULE_FUNCTION = {
    "name": "extract_screening_rules",
    "description": "Extracts stock screening rules from natural language queries, including temporal and complex conditions",
    "parameters": {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_type": {
                            "type": "string",
                            "enum": ["simple", "temporal", "compound"],
                            "description": "Type of rule: simple (current value), temporal (time-based), compound (multiple conditions)"
                        },
                        "metric": {
                            "type": "string",
                            "description": "The metric to filter on (e.g., price, marketCap, volume, pe)"
                        },
                        "operator": {
                            "type": "string",
                            "description": "Comparison operator (>, <, >=, <=, ==, !=, between)"
                        },
                        "value": {
                            "description": "The value to compare against"
                        },
                        "temporal_data": {
                            "type": "object",
                            "properties": {
                                "start_condition": {
                                    "type": "object",
                                    "properties": {
                                        "operator": {"type": "string"},
                                        "value": {}
                                    }
                                },
                                "end_condition": {
                                    "type": "object",
                                    "properties": {
                                        "operator": {"type": "string"},
                                        "value": {}
                                    }
                                },
                                "time_period": {
                                    "type": "string",
                                    "description": "Time period (past_day, past_week, past_month, past_year)"
                                },
                                "duration_days": {
                                    "type": "integer",
                                    "description": "Minimum days the condition must be met"
                                }
                            }
                        },
                        "sub_rules": {
                            "type": "array",
                            "items": {
                                "type": "object"
                            },
                            "description": "Sub-rules for compound conditions"
                        },
                        "logical_operator": {
                            "type": "string",
                            "enum": ["AND", "OR"],
                            "description": "Logical operator for compound rules"
                        }
                    },
                    "required": ["rule_type", "metric"]
                }
            },
            "sort_by": {
                "type": "string",
                "description": "Field to sort results by"
            },
            "sort_order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "Sort order"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results"
            }
        },
        "required": ["rules"]
    }
}

ENHANCED_SYSTEM_PROMPT = """You are an expert at extracting stock screening rules from natural language queries.

Parse the user's query and extract structured screening rules. Pay special attention to:
1. Temporal conditions (e.g., "moved from below $5 to above $5 in the past year")
2. Complex conditions with multiple criteria
3. Time periods mentioned (past day, week, month, year, etc.)
4. Sorting and limit requirements

Examples:
- "Stocks that moved from below $5 to above $5 in the past year" → temporal rule with price movement
- "Tech stocks with PE < 20 and market cap > 10B" → compound rule with multiple conditions
- "Top 10 gainers today with volume > 1M" → simple rules with sorting and limit

Available metrics include: price, marketCap, pe, volume, avgVolume, dividendYield, beta, rsi, sma20, sma50, sma200, change1D, change1W, change1M, change1Y, sector, industry, exchange, and many more.

For temporal queries:
- Extract the starting condition (e.g., "below $5")
- Extract the ending condition (e.g., "above $5")
- Extract the time period (e.g., "past year")
- Extract any duration requirements (e.g., "for at least a day")
"""

async def extract_enhanced_rules(user_query: str) -> Dict[str, Any]:
    """Extract enhanced screening rules from user query using LLM"""
    try:
        response = await async_client.chat.completions.create(
            model=os.getenv("CHAT_MODEL"),
            messages=[
                {"role": "system", "content": ENHANCED_SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            tools=[{"type": "function", "function": ENHANCED_EXTRACT_RULE_FUNCTION}],
            tool_choice={"type": "function", "function": {"name": "extract_screening_rules"}}
        )
        
        if response.choices[0].message.tool_calls:
            args = response.choices[0].message.tool_calls[0].function.arguments
            return orjson.loads(args)
        return {"rules": []}
        
    except Exception as e:
        print(f"Error extracting rules: {e}")
        # Fallback to simple rules extraction
        return {"rules": [], "limit": 100}

def convert_to_frontend_rules(extracted_rules: List[Dict]) -> List[Dict]:
    """Convert extracted rules to frontend format for stock_screener_python.py"""
    frontend_rules = []
    
    for rule_dict in extracted_rules:
        # Skip rules without valid values
        if not rule_dict.get('metric') or rule_dict.get('value') is None:
            continue
            
        # Convert to frontend format
        frontend_rule = {
            'name': rule_dict.get('metric', ''),
            'value': rule_dict.get('value'),
            'condition': rule_dict.get('operator', 'over')
        }
        
        # Map operators to frontend conditions
        operator_mapping = {
            '>': 'over',
            '>=': 'over', 
            '<': 'under',
            '<=': 'under',
            '==': 'exactly',
            '=': 'exactly',
            '!=': 'not',
            'above': 'over',
            'below': 'under',
            'over': 'over',
            'under': 'under',
            'exactly': 'exactly',
            'between': 'between'
        }
        
        frontend_rule['condition'] = operator_mapping.get(
            rule_dict.get('operator', 'over'), 
            'over'
        )
        
        frontend_rules.append(frontend_rule)
    
    print(frontend_rules)
    return frontend_rules

async def process_stock_screener_query(user_query: str) -> Dict[str, Any]:
    """
    Main function to process stock screener queries using stock_screener_python.py
    Returns filtered stocks based on the query
    """
    # Extract rules from the query
    extracted_data = await extract_enhanced_rules(user_query)
    
    # Convert to frontend format rules
    rules = convert_to_frontend_rules(extracted_data.get("rules", []))

    
    # Screen stocks using the Python screener
    results = await python_screener.screen(
        rules,
        limit=extracted_data.get("limit", 100)
    )
    
    # Apply sorting if requested
    sort_by = extracted_data.get("sort_by")
    if sort_by and results['matched_stocks']:
        reverse = extracted_data.get("sort_order", "desc").lower() == "desc"
        results['matched_stocks'].sort(
            key=lambda x: (x.get(sort_by) is None, x.get(sort_by)),
            reverse=reverse
        )
    
    return results

async def get_enhanced_stock_screener(user_query: str) -> Dict[str, Any]:
    try:
        print(user_query)
        result = await process_stock_screener_query(user_query)
        print(result)
        return result
    except Exception as e:
        print(e)
        return {
            "matched_stocks": [],
            "total_matches": 0,
            "error": "Invalid input format"
        }

# Export main functions
__all__ = ['get_enhanced_stock_screener', 'process_stock_screener_query', 'python_screener']