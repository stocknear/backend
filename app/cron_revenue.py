from datetime import datetime, timedelta
import orjson
import ujson
import time
import sqlite3
import asyncio
import aiohttp
import random
from tqdm import tqdm
import os


current_year = datetime.now().year
cutoff_year = current_year - 5

# Load stock screener data
with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

async def save_json(symbol, data):
    path = f"json/revenue/companies/{symbol}.json"
    directory = os.path.dirname(path)

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Write the JSON data
    with open(path, 'w') as file:
        ujson.dump(data, file)


async def get_statistics(symbol):
    """Extract specified columns data for a given symbol."""
    columns = ['revenue','growthRevenue','priceToSalesRatio','revenuePerEmployee','employees']
    
    if symbol in stock_screener_data_dict:
        result = {}
        for column in columns:
            try:
                result[column] = stock_screener_data_dict[symbol].get(column, None)
            except:
                pass
        return result
    return {}



async def get_data(symbol):
    with open(f"json/financial-statements/income-statement/annual/{symbol}.json", "r") as file:
        annual_data = orjson.loads(file.read())

    with open(f"json/financial-statements/income-statement/quarter/{symbol}.json", "r") as file:
        quarter_data = orjson.loads(file.read())

    # Filter the data for the last 5 years
    annual_data = [
        {"date": item["date"], "fiscalYear": item['fiscalYear'], "revenue": item["revenue"]}
        for item in annual_data]

    # Filter the data for the last 5 years
    quarter_data = [
        {"date": item["date"], "fiscalYear": item['fiscalYear'], "period": item['period'], "revenue": item["revenue"]}
        for item in quarter_data]

    stats = await get_statistics(symbol)
    res_dict = {**stats, 'annual': annual_data, 'quarter': quarter_data}


    if annual_data and quarter_data:
        await save_json(symbol, res_dict)

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    total_symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    
    #Testing mode
    #total_symbols = ['DGICA']
    
    for symbol in tqdm(total_symbols):
        try:
            await get_data(symbol)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())