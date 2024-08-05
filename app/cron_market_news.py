import ujson
import asyncio
import aiohttp
import finnhub
import sqlite3
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')
finnhub_api_key = os.getenv('FINNHUB_API_KEY')
finnhub_client = finnhub.Client(api_key=finnhub_api_key)


headers = {"accept": "application/json"}


def filter_and_deduplicate(data, excluded_domains=None, deduplicate_key='title'):
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

'''
async def run():
    limit = 200
    urls = [
    f'https://financialmodelingprep.com/api/v3/stock_news?limit={limit}&apikey={api_key}',
    f"https://financialmodelingprep.com/api/v4/general_news?limit={limit}&apikey={api_key}",
    f"https://financialmodelingprep.com/api/v4/crypto_news?limit={limit}&apikey={api_key}",
    ]
    for url in urls:
        res_list = []

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
        if "stock_news" in url:
            data_name = 'stock-news'
        elif "general_news" in url:
            data_name = 'general-news'
        elif "crypto_news" in url:
            data_name = 'crypto-news'

        with open(f"json/market-news/{data_name}.json", 'w') as file:
            ujson.dump(data, file)
'''



async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 10E9 AND symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    print(len(stock_symbols))
    con.close()
    limit = 200
    company_tickers = ','.join(stock_symbols)
    urls = [
        f'https://financialmodelingprep.com/api/v3/stock_news?tickers={company_tickers}&limit={limit}&apikey={api_key}',
    ]
    for url in urls:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()

        if "stock_news" in url:
            custom_domains = ['prnewswire.com', 'globenewswire.com', 'accesswire.com']
            data = filter_and_deduplicate(data, excluded_domains=custom_domains)
            data_name = 'stock-news'

            
        #elif "press-releases" in url:
        #    data_name = 'press-releases'

        with open(f"json/market-news/{data_name}.json", 'w') as file:
            ujson.dump(data, file)



    general_news = finnhub_client.general_news('general')
    general_news = [item for item in general_news if item["source"] != "" and item["image"] != ""]
    with open(f"json/market-news/general-news.json", 'w') as file:
            ujson.dump(general_news, file)

try:
    asyncio.run(run())
except Exception as e:
    print(e)