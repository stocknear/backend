import ujson
import asyncio
import aiohttp
import sqlite3
from tqdm import tqdm
from dotenv import load_dotenv
import os
import time

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

class RateLimiter:
    def __init__(self, rate_limit=200, sleep_time=60):
        self.rate_limit = rate_limit
        self.sleep_time = sleep_time
        self.request_count = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            self.request_count += 1
            if self.request_count >= self.rate_limit:
                print(f"Processed {self.rate_limit} requests. Sleeping for {self.sleep_time} seconds...")
                await asyncio.sleep(self.sleep_time)
                self.request_count = 0

async def filter_and_deduplicate(data, excluded_domains=None, deduplicate_key='title'):
    """
    Filter out items with specified domains in their URL and remove duplicates based on a specified key.
    """
    if excluded_domains is None:
        excluded_domains = ['prnewswire.com', 'globenewswire.com', 'accesswire.com']
    seen_keys = set()
    filtered_data = []
    for item in data:
        if not any(domain in item['url'] for domain in excluded_domains) and item['image'] is not None:
            key = item.get(deduplicate_key)
            if key and key not in seen_keys:
                filtered_data.append(item)
                seen_keys.add(key)
    return filtered_data

async def save_json(symbol, data):
    """
    Save data as JSON in a batch to reduce disk I/O
    """
    async with asyncio.Lock():  # Ensure thread-safe writes
        with open(f"json/market-news/companies/{symbol}.json", 'w') as file:
            ujson.dump(data, file)

async def get_data(session, chunk, rate_limiter):
    """
    Fetch data for a chunk of tickers using a single session
    """
    await rate_limiter.acquire()
    company_tickers = ','.join(chunk)
    url = f'https://financialmodelingprep.com/stable/news/stock?symbols={company_tickers}&limit=100&apikey={api_key}'
    
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        return []

def get_symbols(db_name, table_name):
    """
    Fetch symbols from the SQLite database
    """
    with sqlite3.connect(db_name) as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute(f"SELECT DISTINCT symbol FROM {table_name} WHERE symbol NOT LIKE '%.%'")
        return [row[0] for row in cursor.fetchall()]

async def process_chunk(session, chunk, rate_limiter):
    """
    Process a chunk of symbols
    """
    data = await get_data(session, chunk, rate_limiter)
    tasks = []
    for symbol in chunk:
        try:
            filtered_data = [item for item in data if item['symbol'] == symbol]
            filtered_data = await filter_and_deduplicate(filtered_data)
            if filtered_data:
                tasks.append(save_json(symbol, filtered_data))
        except Exception as e:
            print(e)
    if tasks:
        await asyncio.gather(*tasks)

async def main():
    """
    Main function to coordinate fetching and processing
    """
    stock_symbols = get_symbols('stocks.db', 'stocks')
    etf_symbols = get_symbols('etf.db', 'etfs')
    crypto_symbols = get_symbols('crypto.db', 'cryptos')
    total_symbols = stock_symbols + etf_symbols + crypto_symbols
    #total_symbols = ['AAPL']
    # Dynamically adjust chunk size
    chunk_size = 1  # Adjust based on your needs
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
    
    rate_limiter = RateLimiter(rate_limit=300, sleep_time=60)
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_chunk(session, chunk, rate_limiter) for chunk in chunks]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")