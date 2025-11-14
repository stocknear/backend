import ujson
import asyncio
import aiohttp
import sqlite3
from dotenv import load_dotenv
import os

from utils.image_validation import ImageValidator

load_dotenv()
api_key = os.getenv('FMP_API_KEY')


def filter_and_deduplicate(data, deduplicate_key='title'):
    seen_keys = set()
    filtered_data = []
    excluded_domains = [] #['accesswire.com']
    for item in data:
        url = item.get('url', '')
        image_url = item.get('image')
        if not any(domain in url for domain in excluded_domains) or image_url:
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
    stock_symbols = {row[0] for row in cursor.fetchall()}
    con.close()

    limit = 200
    urls = [
        f'https://financialmodelingprep.com/api/v3/stock_news?limit={limit}&apikey={api_key}',
        f'https://financialmodelingprep.com/api/v4/general_news?limit={limit}&apikey={api_key}',
        f"https://financialmodelingprep.com/stable/news/press-releases-latest?limit={limit}&apikey={api_key}"
    ]

    timeout = aiohttp.ClientTimeout(total=15)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        image_validator = ImageValidator(session)
        for url in urls:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()

                if "stock_news" in url or "press-releases-latest" in url:
                    data = [item for item in data if item.get('symbol') in stock_symbols]

                if "stock_news" in url:
                    data = filter_and_deduplicate(data)
                    data_name = 'stock-news'
                elif "general_news" in url:
                    data = filter_and_deduplicate(data)
                    data_name = 'general-news'
                elif "press-releases-latest" in url:
                    data_name = 'press-news'
                else:
                    data_name = 'market-news'

                data = await image_validator.filter(data)

                if data:
                    with open(f"json/market-news/{data_name}.json", 'w') as file:
                        ujson.dump(data, file)
            except Exception as e:
                print(e)


try:
    asyncio.run(run())
except Exception as e:
    print(e)
