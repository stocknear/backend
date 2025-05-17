import os
import json
import requests
import aiohttp
import asyncio
from dotenv import load_dotenv
from info import *

load_dotenv()

STOCKNEAR_API_KEY = os.getenv("STOCKNEAR_API_KEY")
API_URL = "http://localhost:8000"


async def fetch_data(session, endpoint, payload):
    try:
        url = f"{API_URL}/{endpoint}"
        async with session.post(
            url,
            json=payload,
            headers={"X-API-KEY": STOCKNEAR_API_KEY}
        ) as response:
            response.raise_for_status()
            return {endpoint: await response.json()}
    except Exception as e:
        return {endpoint: {"error": str(e)}}

async def get_historical_stock_price(ticker):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_data(session, "historical-price", {"ticker": ticker, "timePeriod": "max"}),
        ]
        results = await asyncio.gather(*tasks)
        combined = {k: v for result in results for k, v in result.items()}
        return combined


async def get_income_statement(ticker):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_data(session, "financial-statement", {"ticker": ticker, "statement": 'income-statement'}),
        ]
        results = await asyncio.gather(*tasks)
        combined = {k: v for result in results for k, v in result.items()}
        return combined


async def get_balance_sheet_statement(ticker):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_data(session, "financial-statement", {"ticker": ticker, "statement": 'balance-sheet-statement'}),
        ]
        results = await asyncio.gather(*tasks)
        combined = {k: v for result in results for k, v in result.items()}
        return combined

async def get_cash_flow_statement(ticker):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_data(session, "financial-statement", {"ticker": ticker, "statement": 'cash-flow-statement'}),
        ]
        results = await asyncio.gather(*tasks)
        combined = {k: v for result in results for k, v in result.items()}
        return combined

async def get_ratios_statement(ticker):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_data(session, "financial-statement", {"ticker": ticker, "statement": 'ratios'}),
        ]
        results = await asyncio.gather(*tasks)
        combined = {k: v for result in results for k, v in result.items()}
        return combined

async def get_stock_screener():
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_data(session, "stock-screener-data", {"ruleOfList": []}),
        ]
        results = await asyncio.gather(*tasks)
        combined = {k: v for result in results for k, v in result.items()}
        return combined


def get_function_definitions():
    return [
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
            "name": "get_income_statement",
            "description": f"Retrieves historical income statements (profit and loss statements) for a given stock ticker. This includes key financial data such as {', '.join(key_income)}. Useful for assessing a company's financial performance over time. Data is available annually, quarterly and trailing-twelve-months (ttm).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol for the company (e.g., 'AAPL', 'GOOGL')."
                    }
                },
                "required": ["ticker"]
            },
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
                    }
                },
                "required": ["ticker"]
            },
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
                    }
                },
                "required": ["ticker"]
            },
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
                    }
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "get_stock_screener",
            "description": "Filters and sorts a list of companies based on various financial metrics and criteria.",
        },
    ]


#Testing purposes
if __name__ == '__main__':
    print(get_function_definitions())
    #data = asyncio.run(get_bulk_ticker_information('AAPL'))
    #print(data)
    #print(get_stock_screener())