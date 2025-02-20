from datetime import datetime, timedelta
import ujson
import time
import sqlite3
import asyncio
import aiohttp
import random
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')


today = datetime.today().strftime('%Y-%m-%d')
years = list(range(1995, datetime.today().year, 5))
dates = [f"{year}-01-01" for year in years] + [today]

print(dates)

async def save_json(symbol, data):
    with open(f"json/market-cap/companies/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


async def get_data(session, symbol):
    res_list = []
    start_date = '1990-01-01'
    
    for end_date in dates:

        # Construct the API URL
        url = f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{symbol}?from={start_date}&to={end_date}&limit=2000&apikey={api_key}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if len(data) > 0:
                        res_list.extend(data)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
        
        start_date = end_date

    if len(res_list) > 0:
        # Remove duplicates based on the 'date' field
        unique_res_list = {item['date']: item for item in res_list}.values()
        unique_res_list = sorted(unique_res_list, key=lambda x: x['date'])

        # Filter out 'symbol' from each item
        filtered_data = [{k: v for k, v in item.items() if k != 'symbol'} for item in unique_res_list]
        
        # Save the filtered data
        if filtered_data:
            await save_json(symbol, filtered_data)

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, symbol in enumerate(tqdm(symbols), 1):
            try:
                tasks.append(get_data(session, symbol))
                if i % 100 == 0:
                    await asyncio.gather(*tasks)
                    tasks = []
                    print(f'sleeping mode: {i}')
                    await asyncio.sleep(60)  # Pause for 60 seconds
            except:
                pass
        
        if tasks:
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())