import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

STOCKNEAR_API_KEY = os.getenv("STOCKNEAR_API_KEY")
API_URL = "http://localhost:8000"


def get_financial_statements(ticker, statement):
    try:
        response = requests.post(
            f"{API_URL}/financial-statement",
            json={"ticker": ticker, "statement": statement},
            headers={"X-API-KEY": STOCKNEAR_API_KEY}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_function_definitions():

    return [
        {
            "name": "get_financial_statements",
            "description": "Get a specific financial statement (Income Statement, Balance Sheet, Cash Flow Statement) or key financial ratios for a given stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, like 'TSLA'"
                    },
                    "statement": {
                        "type": "string",
                        "description": "It can either be 'income-statement', 'balance-sheet-statement', 'cash-flow-statement' or 'ratios'. "
                    }
                },
                "required": ["ticker","statement"]
            },
        }
    ] 


#Testing purposes
if __name__ == '__main__':
    pass