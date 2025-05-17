import os
import json
import requests
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

STOCKNEAR_API_KEY = os.getenv("STOCKNEAR_API_KEY")
API_URL = "http://localhost:8000"


async def fetch_bulk_data(session, endpoint, payload):
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

async def get_ticker_information(ticker):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_bulk_data(session, "financial-statement", {"ticker": ticker, "statement": 'income-statement'}),
            fetch_bulk_data(session, "financial-statement", {"ticker": ticker, "statement": 'balance-sheet-statement'}),
            fetch_bulk_data(session, "financial-statement", {"ticker": ticker, "statement": 'cash-flow-statement'}),
            fetch_bulk_data(session, "financial-statement", {"ticker": ticker, "statement": 'ratios'}),
            fetch_bulk_data(session, "historical-price", {"ticker": ticker, "timePeriod": "max"}),
        ]
        results = await asyncio.gather(*tasks)
        combined = {k: v for result in results for k, v in result.items()}
        return combined

async def get_stock_screener():
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_bulk_data(session, "stock-screener-data", {"ruleOfList": []}),
        ]
        results = await asyncio.gather(*tasks)
        combined = {k: v for result in results for k, v in result.items()}
        return combined


def get_function_definitions():
    return [
        {
            "name": "get_ticker_information",
            "description": "Retrieves comprehensive financial data for a given stock ticker, including historical prices, income statements, balance sheets, cash flow statements, and key ratios.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, like 'TSLA'"
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
    pass
    #data = asyncio.run(get_bulk_ticker_information('TSLA'))
    #print(data)
    #print(get_stock_screener())