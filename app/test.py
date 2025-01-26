from __future__ import print_function
import asyncio
import time
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
from datetime import datetime, timedelta
import ast
import orjson
from tqdm import tqdm
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import re

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('INTRINIO_API_KEY')

intrinio.ApiClient().set_api_key(api_key)
# intrinio.ApiClient().allow_retries(True)

today = datetime.today()
start_date = (today - timedelta(150)).strftime("%Y-%m-%d")
end_date = (today + timedelta(30)).strftime("%Y-%m-%d")

next_page = ''
page_size = 1000
activity_type = ''
sentiment = ''
minimum_total_value = 0
maximum_total_value = 2E10

# Database connection and symbol retrieval
def get_total_symbols():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
        stocks_symbols = [row[0] for row in cursor.fetchall()]

    with sqlite3.connect('etf.db') as etf_con:
        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    return stocks_symbols + etf_symbols


def get_tickers_from_directory():
    directory = "json/options-historical-data/companies"
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        
        # Get all tickers from filenames
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


async def save_json(data, symbol):
    directory = "json/unusual-activity"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))


def parse_option_symbol(option_symbol):
    # Define regex pattern to match the symbol structure
    match = re.match(r"([A-Z]+)(\d{6})([CP])(\d+)", option_symbol)
    if not match:
        raise ValueError(f"Invalid option_symbol format: {option_symbol}")
    
    ticker, expiration, option_type, strike_price = match.groups()
    
    # Convert expiration to datetime
    date_expiration = datetime.strptime(expiration, "%y%m%d").date()
    
    # Convert strike price to float
    strike_price = int(strike_price) / 1000

    return date_expiration, option_type, strike_price


async def get_data(symbol):
    response = intrinio.OptionsApi().get_unusual_activity_intraday(
        symbol, 
        next_page=next_page, 
        page_size=page_size, 
        activity_type=activity_type, 
        sentiment=sentiment, 
        start_date=start_date, 
        end_date=end_date, 
        minimum_total_value=minimum_total_value, 
        maximum_total_value=maximum_total_value
    )
    data = (response.__dict__['_trades'])
    res_list = []
    if len(data) > 0:
        for item in data:
            try:
                trade_data = item.__dict__
                trade_data = {key.lstrip('_'): value for key, value in trade_data.items()}
                option_symbol = trade_data['contract'].replace("___", "").replace("__", "").replace("_", "")
                date_expiration, option_type, strike_price = parse_option_symbol(option_symbol)

                res_list.append({
                    'date': trade_data['timestamp'].strftime("%Y-%m-%d"),
                    'askprice': trade_data['ask_at_execution'],
                    'bidPrice': trade_data['bid_at_execution'],
                    'premium': trade_data['total_value'],
                    'sentiment': trade_data['sentiment'].capitalize(),
                    'avgPrice': trade_data['average_price'],
                    'price': trade_data['underlying_price_at_execution'],
                    'unusualType': trade_data['type'].capitalize(),
                    'size': trade_data['total_size'],
                    'optionSymbol': option_symbol,
                    'strike': strike_price,
                    'expiry': date_expiration.strftime("%Y-%m-%d"),
                    'optionType': option_type.replace("P", "Put").replace("C", "Call")
                })
            except Exception as e:
                print(e)

        res_list = sorted(res_list, key=lambda x: x['date'], reverse=True)
        res_list = [item for item in res_list if item['date'] == '2025-01-24']
        print(len(res_list))


async def main():
    total_symbols = get_tickers_from_directory()
    if len(total_symbols) < 3000:
        total_symbols = get_total_symbols()

    for symbol in tqdm(['XLU']):
        try:
            data = await get_data(symbol)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
