import ujson
import asyncio
import aiohttp
import finnhub
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')
finnhub_api_key = os.getenv('FINNHUB_API_KEY')
finnhub_client = finnhub.Client(api_key=finnhub_api_key)


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
#Finnhub data
async def run():
    limit = 200
    urls = [
        f'https://financialmodelingprep.com/api/v3/stock_news?limit={limit}&apikey={api_key}',
        f'https://financialmodelingprep.com/api/v4/crypto_news?limit={limit}&apikey={api_key}',
    ]
    for url in urls:
        res_list = []

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
        if "stock_news" in url:
            data_name = 'stock-news'
        elif "crypto_news" in url:
            data_name = 'crypto-news'
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