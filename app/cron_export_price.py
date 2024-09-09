from datetime import datetime, timedelta
import ujson
import sqlite3
import asyncio
import aiohttp
from tqdm import tqdm
import os
from dotenv import load_dotenv
from aiohttp import TCPConnector

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 100
request_semaphore = asyncio.Semaphore(MAX_REQUESTS_PER_MINUTE)
last_request_time = datetime.min

async def fetch_data(session, url):
    global last_request_time
    async with request_semaphore:
        # Ensure at least 60 seconds between batches of MAX_REQUESTS_PER_MINUTE
        current_time = datetime.now()
        if (current_time - last_request_time).total_seconds() < 60:
            await asyncio.sleep(60 - (current_time - last_request_time).total_seconds())
        last_request_time = datetime.now()

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error status {response.status} for URL: {url}")
                    return []
        except Exception as e:
            print(f"Error fetching data from {url}: {e}")
            return []

def get_existing_data(symbol, interval):
    file_path = f"json/export/price/{interval}/{symbol}.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return ujson.load(file)
    return []

async def get_data(session, symbol, time_period):
    existing_data = get_existing_data(symbol, time_period)
    if not existing_data:
        return await fetch_all_data(session, symbol, time_period)
    
    last_date = datetime.strptime(existing_data[-1]['date'], "%Y-%m-%d %H:%M:%S")
    current_date = datetime.utcnow()
    
    if (current_date - last_date).days < 1:
        return  # Data is up to date, skip to next symbol
    
    # Fetch only missing data
    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = current_date.strftime("%Y-%m-%d")
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/{time_period}/{symbol}?serietype=bar&extend=false&from={start_date}&to={end_date}&apikey={api_key}"
    
    new_data = await fetch_data(session, url)
    if new_data:
        existing_data.extend(new_data)
        existing_data.sort(key=lambda x: x['date'])
        await save_json(symbol, existing_data, time_period)

async def fetch_all_data(session, symbol, time_period):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/{time_period}/{symbol}?serietype=bar&extend=false&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey={api_key}"
    
    data = await fetch_data(session, url)
    if data:
        data.sort(key=lambda x: x['date'])
        await save_json(symbol, data, time_period)

async def save_json(symbol, data, interval):
    os.makedirs(f"json/export/price/{interval}", exist_ok=True)
    with open(f"json/export/price/{interval}/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

async def process_symbol(session, symbol):
    await get_data(session, symbol, '30min')
    await get_data(session, symbol, '1hour')

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    con.close()
    etf_con.close()

    total_symbols = stock_symbols + etf_symbols

    connector = TCPConnector(limit=MAX_REQUESTS_PER_MINUTE)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_symbol(session, symbol) for symbol in total_symbols]
        for i, _ in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks)), 1):
            if i % MAX_REQUESTS_PER_MINUTE == 0:
                print(f'Processed {i} symbols')
                await asyncio.sleep(60)  # Sleep for 60 seconds after every MAX_REQUESTS_PER_MINUTE symbols

if __name__ == "__main__":
    asyncio.run(run())