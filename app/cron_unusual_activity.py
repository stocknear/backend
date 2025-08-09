import asyncio
import time
from datetime import datetime, timedelta
import ast
import orjson
from tqdm import tqdm
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import re
import os


today = datetime.now().date()

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


async def save_json(data, symbol):
    directory = "json/unusual-activity"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))


async def get_dataset():
    today = datetime.today()
    start_date = today - timedelta(days=365)
    date_list = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(365)]

    unique_data = {}
    
    for date in tqdm(date_list):
        try:
            with open(f"json/options-historical-data/flow-data/{date}.json", "r") as file:
                data = orjson.loads(file.read())
                data = [item for item in data if item['cost_basis'] >=500_000]
                for item in data:
                    if "id" in item:
                        unique_data[item["id"]] = item  # Store unique items based on "id"
        except:
            pass

    try:
        with open(f"json/options-flow/feed/data.json", "r") as file:
            data = orjson.loads(file.read())
            data = [item for item in data if item['cost_basis'] >=500_000]
            for item in data:
                if "id" in item:
                    unique_data[item["id"]] = item
    except:
        pass

    all_data = list(unique_data.values())  # Convert back to a list

    return all_data


async def get_data(symbol, data):
    res_list = []

    if data:
        for item in data:
            try:
                expiry_date = datetime.strptime(item['date_expiration'], "%Y-%m-%d").date()
                if item['ticker'] == symbol and expiry_date >= today:
                    res_list.append({
                        'date': item['date'],
                        'premium': item['cost_basis'],
                        'sentiment': item['sentiment'],
                        'executionEst': item['execution_estimate'],
                        'price': item['underlying_price'],
                        'unusualType': item['option_activity_type'],
                        'size': item['size'],
                        'oi': item['open_interest'],
                        'optionSymbol': item['option_symbol'],
                        'strike': item['strike_price'],
                        'expiry': item['date_expiration'],
                        'optionType': item['put_call'],
                    })
            except Exception as e:
                print(e)

    res_list = sorted(res_list, key=lambda x: x['date'], reverse=True)

    if res_list:
        await save_json(res_list, symbol)



async def main():
    total_symbols = get_total_symbols()
    data = await get_dataset()

    for symbol in tqdm(total_symbols):
        try:
            await get_data(symbol, data)
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
if __name__ == "__main__":
    asyncio.run(main())
