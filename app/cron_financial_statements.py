import os
import ujson
import asyncio
import aiohttp
import sqlite3
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

# Configurations
include_current_quarter = False
max_concurrent_requests = 100  # Limit concurrent requests

async def fetch_data(session, url, symbol, attempt=0):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                print(f"Error fetching data for {symbol}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Exception during fetching data for {symbol}: {e}")
        return None

async def save_json(symbol, period, data_type, data):
    os.makedirs(f"json/financial-statements/{data_type}/{period}/", exist_ok=True)
    with open(f"json/financial-statements/{data_type}/{period}/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

async def get_financial_statements(session, symbol, semaphore, request_counter):
    base_url = "https://financialmodelingprep.com/api/v3"
    periods = ['quarter', 'annual']
    financial_data_types = ['income-statement', 'balance-sheet-statement', 'cash-flow-statement', 'ratios']
    growth_data_types = ['income-statement-growth', 'balance-sheet-statement-growth', 'cash-flow-statement-growth']
    
    async with semaphore:
        for period in periods:
            # Fetch regular financial statements
            for data_type in financial_data_types:
                url = f"{base_url}/{data_type}/{symbol}?period={period}&apikey={api_key}"
                data = await fetch_data(session, url, symbol)
                if data:
                    await save_json(symbol, period, data_type, data)
                
                request_counter[0] += 1  # Increment the request counter
                if request_counter[0] >= 500:
                    await asyncio.sleep(60)  # Pause for 60 seconds
                    request_counter[0] = 0  # Reset the request counter after the pause
            
            # Fetch financial statement growth data
            for growth_type in growth_data_types:
                growth_url = f"{base_url}/{growth_type}/{symbol}?period={period}&apikey={api_key}"
                growth_data = await fetch_data(session, growth_url, symbol)
                if growth_data:
                    await save_json(symbol, period, growth_type, growth_data)

                request_counter[0] += 1  # Increment the request counter
                if request_counter[0] >= 500:
                    await asyncio.sleep(60)  # Pause for 60 seconds
                    request_counter[0] = 0  # Reset the request counter after the pause

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    semaphore = asyncio.Semaphore(max_concurrent_requests)
    request_counter = [0]  # Using a list to keep a mutable counter across async tasks

    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in tqdm(symbols):
            task = asyncio.create_task(get_financial_statements(session, symbol, semaphore, request_counter))
            tasks.append(task)
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run())
