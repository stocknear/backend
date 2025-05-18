import os
import json
import requests
import aiohttp
import asyncio
from dotenv import load_dotenv
import orjson
from pathlib import Path
from info import *

load_dotenv()


async def get_historical_stock_price(ticker):
    pass


async def get_income_statement(tickers,time_period = "annual",keep_keys = None):
    res_dict = {}
    for ticker in tickers:
        try:
            file_path = Path(f"../../json/financial-statements/income-statement/{time_period}/{ticker}.json")
            
            # Load the raw data
            with file_path.open("rb") as f:
                raw_data = orjson.loads(f.read())
            
            # Keys we want to strip out by default
            remove_keys = [
                "symbol", "reportedCurrency",
                "acceptedDate", "cik", "filingDate"
            ]
            
            if keep_keys:
                # Make a local copy and ensure required fields are present
                keys_to_keep = list(keep_keys)
                if "date" not in keys_to_keep:
                    keys_to_keep.append("date")
                if "fiscalYear" not in keys_to_keep:
                    keys_to_keep.append("fiscalYear")
                if "period" not in keys_to_keep:
                    keys_to_keep.append("period")
                
                # Keep only those keys
                filtered = [
                    {k: v for k, v in stmt.items() if k in keys_to_keep}
                    for stmt in raw_data
                ]
            else:
                # Remove the unwanted keys
                filtered = [
                    {k: v for k, v in stmt.items() if k not in remove_keys}
                    for stmt in raw_data
                ]

            res_dict[ticker] = filtered
        except:
            res_dict[ticker] = []
    print(res_dict)
    return res_dict


async def get_balance_sheet_statement(ticker,time_period = "annual",keep_keys = None):
    file_path = Path(f"../../json/financial-statements/balance-sheet-statement/{time_period}/{ticker}.json")
    
    # Load the raw data
    with file_path.open("rb") as f:
        raw_data = orjson.loads(f.read())
    
    # Keys we want to strip out by default
    remove_keys = [
        "symbol", "reportedCurrency",
        "acceptedDate", "cik", "filingDate"
    ]
    
    if keep_keys:
        # Make a local copy and ensure required fields are present
        keys_to_keep = list(keep_keys)
        if "date" not in keys_to_keep:
            keys_to_keep.append("date")
        if "fiscalYear" not in keys_to_keep:
            keys_to_keep.append("fiscalYear")
        if "period" not in keys_to_keep:
            keys_to_keep.append("period")
        
        # Keep only those keys
        filtered = [
            {k: v for k, v in stmt.items() if k in keys_to_keep}
            for stmt in raw_data
        ]
    else:
        # Remove the unwanted keys
        filtered = [
            {k: v for k, v in stmt.items() if k not in remove_keys}
            for stmt in raw_data
        ]
    return filtered

async def get_cash_flow_statement(ticker,time_period = "annual",keep_keys = None):
    file_path = Path(f"../../json/financial-statements/cash-flow-statement/{time_period}/{ticker}.json")
    
    # Load the raw data
    with file_path.open("rb") as f:
        raw_data = orjson.loads(f.read())
    
    # Keys we want to strip out by default
    remove_keys = [
        "symbol", "reportedCurrency",
        "acceptedDate", "cik", "filingDate"
    ]
    
    if keep_keys:
        # Make a local copy and ensure required fields are present
        keys_to_keep = list(keep_keys)
        if "date" not in keys_to_keep:
            keys_to_keep.append("date")
        if "fiscalYear" not in keys_to_keep:
            keys_to_keep.append("fiscalYear")
        if "period" not in keys_to_keep:
            keys_to_keep.append("period")
        
        # Keep only those keys
        filtered = [
            {k: v for k, v in stmt.items() if k in keys_to_keep}
            for stmt in raw_data
        ]
    else:
        # Remove the unwanted keys
        filtered = [
            {k: v for k, v in stmt.items() if k not in remove_keys}
            for stmt in raw_data
        ]
    return filtered


async def get_ratios_statement(ticker,time_period = "annual",keep_keys = None):
    file_path = Path(f"../../json/financial-statements/ratios/{time_period}/{ticker}.json")
    
    # Load the raw data
    with file_path.open("rb") as f:
        raw_data = orjson.loads(f.read())
    
    # Keys we want to strip out by default
    remove_keys = [
        "symbol", "reportedCurrency",
        "acceptedDate", "cik", "filingDate"
    ]

    
    if keep_keys:
        # Make a local copy and ensure required fields are present
        keys_to_keep = list(keep_keys)
        if "date" not in keys_to_keep:
            keys_to_keep.append("date")
        if "fiscalYear" not in keys_to_keep:
            keys_to_keep.append("fiscalYear")
        if "period" not in keys_to_keep:
            keys_to_keep.append("period")
        
        # Keep only those keys
        filtered = [
            {k: v for k, v in stmt.items() if k in keys_to_keep}
            for stmt in raw_data
        ]
    else:
        # Remove the unwanted keys
        filtered = [
            {k: v for k, v in stmt.items() if k not in remove_keys}
            for stmt in raw_data
        ]
    return filtered


async def get_hottest_contracts(ticker,time_period = "annual",keep_keys = None):
    file_path = Path(f"../../json/financial-statements/ratios/{time_period}/{ticker}.json")
    
    # Load the raw data
    with file_path.open("rb") as f:
        raw_data = orjson.loads(f.read())
    
    # Keys we want to strip out by default
    remove_keys = [
        "symbol", "reportedCurrency",
        "acceptedDate", "cik", "filingDate"
    ]

    
    if keep_keys:
        # Make a local copy and ensure required fields are present
        keys_to_keep = list(keep_keys)
        if "date" not in keys_to_keep:
            keys_to_keep.append("date")
        if "fiscalYear" not in keys_to_keep:
            keys_to_keep.append("fiscalYear")
        if "period" not in keys_to_keep:
            keys_to_keep.append("period")
        
        # Keep only those keys
        filtered = [
            {k: v for k, v in stmt.items() if k in keys_to_keep}
            for stmt in raw_data
        ]
    else:
        # Remove the unwanted keys
        filtered = [
            {k: v for k, v in stmt.items() if k not in remove_keys}
            for stmt in raw_data
        ]
    return filtered



async def get_stock_screener():
    pass


def get_function_definitions():
    return [
        {
          "name": "get_income_statement",
          "description": "Retrieves historical income statements (profit and loss statements) for a given stock ticker and time period (annual, quarter, ttm). This includes key financial data such as revenue, netIncome, etc. Useful for assessing a company's financial performance over time. Data is available annually, quarterly and trailing-twelve-months (ttm).",
          "parameters": {
            "type": "object",
            "properties": {
              "tickers": {
                "type": "array",
                "description": "The ticker list of stock symbols for the company (e.g., ['AAPL', 'GOOGL'])."
              },
              "time_period": {
                "type": "string",
                "enum": ["annual", "quarter", "ttm"],
                "description": "The time period for the company data (e.g., 'annual', 'quarter', 'ttm')."
              },
              "keep_keys": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "List of keys to retain in the result (e.g., ['revenue', 'netIncome'])."
              }
            },
            "required": ["ticker", "time_period", "keep_keys"]
          }
        },
        {
          "name": "get_balance_sheet_statement",
          "description": f"Fetches historical balance sheet statements for a specified stock ticker. This statement provides a snapshot of a company's assets, liabilities, and shareholders' equity at a specific point in time. Key data points include {', '.join(key_balance_sheet)}. Data is available annually, quarterly and trailing-twelve-months (ttm).",
          "parameters": {
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "The stock ticker symbol for the company (e.g., 'AAPL', 'GOOGL')."
              },
              "time_period": {
                "type": "string",
                "enum": ["annual", "quarter", "ttm"],
                "description": "The time period for the company data (e.g., 'annual', 'quarter', 'ttm')."
              },
              "keep_keys": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "List of keys to retain in the result (e.g., ['cashAndCashEquivalents', 'shortTermInvestments'])."
              }
            },
            "required": ["ticker", "time_period", "keep_keys"]
          }
        },
        {
          "name": "get_cash_flow_statement",
          "description": f"Obtains historical cash flow statements for a stock ticker. This statement shows how changes in balance sheet accounts and income affect cash and cash equivalents, and breaks the analysis down to operating, investing, and financing activities. Includes items like {', '.join(key_cash_flow)}. Data is available annually, quarterly and trailing-twelve-months (ttm).",
          "parameters": {
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "The stock ticker symbol for the company (e.g., 'AAPL', 'GOOGL')."
              },
              "time_period": {
                "type": "string",
                "enum": ["annual", "quarter", "ttm"],
                "description": "The time period for the company data (e.g., 'annual', 'quarter', 'ttm')."
              },
              "keep_keys": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "List of keys to retain in the result (e.g., ['freeCashFlow', 'stockBasedCompensation'])."
              }
            },
            "required": ["ticker", "time_period", "keep_keys"]
          }
        },
        {
          "name": "get_ratios_statement",
          "description": f"Retrieves various historical financial ratios for a given stock ticker. These ratios are used to evaluate a company's performance, valuation, liquidity, solvency, and efficiency. Examples include {', '.join(key_ratios)}. Data is available annually and quarterly",
          "parameters": {
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "The stock ticker symbol for the company (e.g., 'AAPL', 'GOOGL')."
              },
              "time_period": {
                "type": "string",
                "enum": ["annual", "quarter"],
                "description": "The time period for the company data (e.g., 'annual', 'quarter')."
              },
              "keep_keys": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "List of keys to retain in the result (e.g., ['grossProfitMargin', 'ebitMargin'])."
              }
            },
            "required": ["ticker", "time_period", "keep_keys"]
          }
        },
    ]


'''
{
            "name": "get_historical_stock_price",
            "description": "Fetches historical stock price (open, high, low, close) and volume data for a specific stock ticker. Useful for analyzing past stock performance and trends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol for the company (e.g., 'AAPL' for Apple Inc., 'MSFT' for Microsoft Corp.)."
                    }
                },
                "required": ["ticker"]
            },
        },
    {
        "name": "get_stock_screener",
        "description": "Filters and sorts a list of companies based on various financial metrics and criteria.",
    },
'''

#Testing purposes
data = asyncio.run(get_income_statement(['AAPL','NVDA'], 'annual',['revenue','netIncome']))
#print(get_stock_screener())