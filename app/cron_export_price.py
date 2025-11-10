from datetime import datetime, timedelta
import ujson
import sqlite3
import asyncio
import aiohttp
from tqdm import tqdm
import os
from dotenv import load_dotenv
from aiohttp import TCPConnector
import gc

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 500
request_semaphore = asyncio.Semaphore(MAX_REQUESTS_PER_MINUTE)

async def fetch_data(session, url):
    async with request_semaphore:
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
        # If no existing data, fetch all data
        return await fetch_all_data(session, symbol, time_period)
    
    last_date = datetime.strptime(existing_data[-1]['date'], "%Y-%m-%d %H:%M:%S")
    current_date = datetime.utcnow()
    
    # If data is up to date, skip fetching
    if (current_date - last_date).days < 1:
        return  # Data is recent, skip further fetch
    
    # Fetch missing data only from the last saved date to the current date
    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = current_date.strftime("%Y-%m-%d")
    print(start_date, end_date)
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/{time_period}/{symbol}?serietype=bar&extend=false&from={start_date}&to={end_date}&apikey={api_key}"
    
    new_data = await fetch_data(session, url)
    if new_data:
        existing_data.extend(new_data)
        existing_data.sort(key=lambda x: x['date'])  # Sort by date
        await save_json(symbol, existing_data, time_period)

async def fetch_all_data(session, symbol, time_period):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)
    
    step = timedelta(days=5)  # Step of 5 days
    current_start_date = start_date
    
    all_data = []  # To accumulate all the data
    while current_start_date < end_date:
        current_end_date = min(current_start_date + step, end_date)
        
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{time_period}/{symbol}?serietype=bar&extend=false&from={current_start_date.strftime('%Y-%m-%d')}&to={current_end_date.strftime('%Y-%m-%d')}&apikey={api_key}"
        
        data = await fetch_data(session, url)
        print("api endpoint called")
        
        if data:
            all_data.extend(data)  # Accumulate the fetched data
            print(f"Fetched {len(data)} records from {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")
        
        # Move the window forward by 5 days
        current_start_date = current_end_date

    if all_data:
        # Sort the data by date before saving
        all_data.sort(key=lambda x: x['date'])
        await save_json(symbol, all_data, time_period)
        gc.collect()


async def save_json(symbol, data, interval):
    os.makedirs(f"json/export/price/{interval}", exist_ok=True)
    file_path = f"json/export/price/{interval}/{symbol}.json"
    with open(file_path, 'w') as file:
        ujson.dump(data, file)

async def process_symbol(session, symbol):
    await get_data(session, symbol, '1hour')
    await get_data(session, symbol, '30min')

async def run():
    # Load symbols from databases
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    
    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]
    
    con.close()
    etf_con.close()

    # List of total symbols to process
    total_symbols = stock_symbols  # Use stock_symbols + etf_symbols if needed
    
    chunk_size = len(total_symbols) // 500  # Divide the list into N chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]

    for chunk in tqdm(chunks):
        print(len(chunk))
        connector = TCPConnector(limit=MAX_REQUESTS_PER_MINUTE)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [process_symbol(session, symbol) for symbol in chunk]
            
            # Use tqdm to track progress of tasks
            for i, task in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks)), 1):
                await task  # Ensure all tasks are awaited properly
                if i % MAX_REQUESTS_PER_MINUTE == 0:
                    print(f'Processed {i} symbols, sleeping to respect rate limits...')
                    gc.collect()
                    await asyncio.sleep(30)  # Pause for 60 seconds to avoid hitting rate limits

        gc.collect()
        await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(run())