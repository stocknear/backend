import ujson
import asyncio
import aiohttp
import sqlite3
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')



def filter_and_deduplicate(data,deduplicate_key='title'):
 
    seen_keys = set()
    filtered_data = []
    excluded_domains = ['accesswire.com']
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
        f'https://financialmodelingprep.com/api/v3/stock_news?limit={limit}&apikey={api_key}',
        f'https://financialmodelingprep.com/api/v4/general_news?limit={limit}&apikey={api_key}',
        f"https://financialmodelingprep.com/stable/news/press-releases-latest?limit={limit}&apikey={api_key}"
    ]
    for url in urls:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    
                    if "stock_news" in url or "press-releases-latest" in url:
                        data = [item for item in data if item['symbol'] in stock_symbols]

            if "stock_news" in url:
                data = filter_and_deduplicate(data)
                data_name = 'stock-news'
            
            if "general_news" in url:
                data = filter_and_deduplicate(data)
                data_name = 'general-news'
            
            if "press-releases-latest" in url:
                data_name = 'press-news'

            if len(data) > 0:
                with open(f"json/market-news/{data_name}.json", 'w') as file:
                    ujson.dump(data, file)
        except Exception as e:
            print(e)

try:
    asyncio.run(run())
except Exception as e:
    print(e)