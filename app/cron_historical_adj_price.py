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


async def save_json(symbol, data):
    path = f"json/historical-price/adj"
    os.makedirs(path, exist_ok=True)  # Create directories if they don't exist
    with open(f"{path}/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

def get_symbols(db_name, table_name):
    """
    Fetch symbols from the SQLite database
    """
    with sqlite3.connect(db_name) as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute(f"SELECT DISTINCT symbol FROM {table_name} WHERE symbol NOT LIKE '%.%'")
        return [row[0] for row in cursor.fetchall()]

async def get_data(session, symbol):
    res_list = []
    start_date = '2000-01-01'
    
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted?symbol={symbol}&from={start_date}&to={today}&apikey={api_key}"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                res_list = await response.json()
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

    if len(res_list) > 0:
        await save_json(symbol, res_list)

async def run():
    stock_symbols = get_symbols('stocks.db', 'stocks')
    etf_symbols = get_symbols('etf.db', 'etfs')
    index_symbols = get_symbols('index.db','indices')
    total_symbols = stock_symbols + etf_symbols + index_symbols
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, symbol in enumerate(tqdm(total_symbols), 1):
            try:
                tasks.append(get_data(session, symbol))
                if i % 500 == 0:
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