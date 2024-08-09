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

async def fetch_news(session, url):
    async with session.get(url) as response:
        return await response.json()

async def save_news(data, symbol):
    #os.makedirs("json/market-news/companies", exist_ok=True)
    with open(f"json/market-news/companies/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

async def process_symbols(symbols):
    limit = 200
    chunk_size = 50  # Adjust this value based on API limitations
    
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(symbols), chunk_size)):
            chunk = symbols[i:i+chunk_size]
            company_tickers = ','.join(chunk)
            url = f'https://financialmodelingprep.com/api/v3/stock_news?tickers={company_tickers}&limit={limit}&apikey={api_key}'
            
            data = await fetch_news(session, url)

            custom_domains = ['prnewswire.com', 'globenewswire.com', 'accesswire.com']
            data = await filter_and_deduplicate(data, excluded_domains=custom_domains)

            grouped_data = {}
            for item in data:
                symbol = item['symbol']
                if symbol in chunk:
                    if symbol not in grouped_data:
                        grouped_data[symbol] = []
                    grouped_data[symbol].append(item)
            
            # Save the filtered data for each symbol in the chunk
            tasks = []
            for symbol in chunk:
                filtered_data = grouped_data.get(symbol, [])
                tasks.append(save_news(filtered_data, symbol))
            
            await asyncio.gather(*tasks)

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

    await process_symbols(total_symbols)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")