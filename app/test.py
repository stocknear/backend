import aiohttp
import aiofiles
import ujson
import sqlite3
import pandas as pd
import asyncio
import pytz
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tqdm import tqdm
import pytz


date_format = "%a, %d %b %Y %H:%M:%S %z"

load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

headers = {"accept": "application/json"}

async def get_latest_wiim(session):
    url = "https://api.benzinga.com/api/v2/news"
    querystring = {"token": api_key,"dateFrom":"2025-01-16","dateTo":"2025-01-17","sort":"created:desc", "pageSize": 1000, "channels":"WIIM"}

    try:
        async with session.get(url, params=querystring, headers=headers) as response:
            res_list = []
            data = ujson.loads(await response.text())

            for item in data:
                try:
                    if len(item['stocks']) ==1:
                        item['ticker'] = item['stocks'][0].get('name',None)

                        with open(f"json/quote/{item['ticker']}.json","r") as file:
                            quote_data = ujson.load(file)
                            item['marketCap'] = quote_data.get('marketCap',None)
                        
                        res_list.append({'date': item['created'], 'text': item['title'], 'marketCap': item['marketCap'],'ticker': item['ticker']})
                except:
                    pass
            res_list = sorted(
                res_list,
                key=lambda item: (item['marketCap'], datetime.strptime(item['date'], '%a, %d %b %Y %H:%M:%S %z')),
                reverse=True
            )

            print(res_list[:10])

            '''
            for item in res:
                for el in item['stocks']:
                    # Update the 'name' key to 'ticker'
                    if 'name' in el:
                        el['ticker'] = el.pop('name')
                        if el['ticker'] in stock_symbols:
                            el['assetType'] = 'stock'
                        elif el['ticker'] in etf_symbols:
                            el['assetType'] = 'etf'
                res_list.append({'date': item['created'], 'text': item['title'], 'stocks': item['stocks']})
            with open(f"json/wiim/rss-feed/data.json", 'w') as file:
                    ujson.dump(res_list, file)
            '''

    except Exception as e:
        print(e)

async def run():
    async with aiohttp.ClientSession() as session:
        await get_latest_wiim(session)

try:
    asyncio.run(run())
except Exception as e:
    print(e)