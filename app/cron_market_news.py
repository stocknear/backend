import ujson
import asyncio
import aiohttp
import sqlite3
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')


headers = {"accept": "application/json"}


def filter_and_deduplicate(data, excluded_domains=None, deduplicate_key='title'):
 
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



async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 10E9 AND symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    limit = 200
    urls = [
        f'https://financialmodelingprep.com/stable/news/stock-latest?limit={limit}&apikey={api_key}',
        f'https://financialmodelingprep.com/stable/news/general-latest?limit={limit}&apikey={api_key}',
        f"https://financialmodelingprep.com/stable/news/press-releases-latest?limit={limit}&apikey={api_key}"
    ]
    for url in urls:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    
                    if "stock-latest" in url or "press-releases-latest" in url:
                        data = [item for item in data if item['symbol'] in stock_symbols]

            if "stock-latest" in url:
                custom_domains = ['prnewswire.com', 'globenewswire.com', 'accesswire.com']
                data = filter_and_deduplicate(data, excluded_domains=custom_domains)
                data_name = 'stock-news'
            
            if "general-latest" in url:
                custom_domains = ['prnewswire.com', 'globenewswire.com', 'accesswire.com']
                data = filter_and_deduplicate(data, excluded_domains=custom_domains)
                data_name = 'general-news'
            
            if "press-releases-latest" in url:
                data_name = 'press-news'

            if len(data) > 0:
                with open(f"json/market-news/{data_name}.json", 'w') as file:
                    ujson.dump(data, file)
        except:
            pass

try:
    asyncio.run(run())
except Exception as e:
    print(e)