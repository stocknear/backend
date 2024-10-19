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

async def filter_and_deduplicate(data, excluded_domains=None, deduplicate_key='title'):
    """
    Filter out items with specified domains in their URL and remove duplicates based on a specified key.
    
    Args:
    data (list): List of dictionaries containing item data.
    excluded_domains (list): List of domain strings to exclude. Defaults to ['prnewswire.com', 'globenewswire.com', 'accesswire.com'].
    deduplicate_key (str): The key to use for deduplication. Defaults to 'title'.
    
    Returns:
    list: Filtered and deduplicated list of items.
    """
    if excluded_domains is None:
        excluded_domains = ['prnewswire.com', 'globenewswire.com', 'accesswire.com']
    
    seen_keys = set()
    filtered_data = []
    
    for item in data:
        if not any(domain in item['url'] for domain in excluded_domains):
            key = item.get(deduplicate_key)
            if key and key not in seen_keys:
                filtered_data.append(item)
                seen_keys.add(key)
    
    return filtered_data

async def save_quote_as_json(symbol, data):
    with open(f"json/market-news/companies/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

async def get_data(chunk):
    company_tickers = ','.join(chunk)
    async with aiohttp.ClientSession() as session:
        url = f'https://financialmodelingprep.com/api/v3/stock_news?tickers={company_tickers}&page=0&limit=2000&apikey={api_key}'
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return []

def get_symbols(db_name, table_name):
    with sqlite3.connect(db_name) as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute(f"SELECT DISTINCT symbol FROM {table_name} WHERE symbol NOT LIKE '%.%'")
        return [row[0] for row in cursor.fetchall()]

async def main():
    stock_symbols = get_symbols('stocks.db', 'stocks')
    etf_symbols = get_symbols('etf.db', 'etfs')
    crypto_symbols = get_symbols('crypto.db', 'cryptos')
    total_symbols = stock_symbols + etf_symbols + crypto_symbols

    chunk_size = len(total_symbols) // 70  # Divide the list into N chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]

    for chunk in tqdm(chunks):
        data = await get_data(chunk)
        for symbol in chunk:
            filtered_data = [item for item in data if item['symbol'] == symbol]
            filtered_data = await filter_and_deduplicate(filtered_data)
            if len(filtered_data) > 0:
                await save_quote_as_json(symbol, filtered_data)
            

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")