import aiohttp
import aiofiles
import ujson
import orjson
import sqlite3
import asyncio
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tqdm import tqdm
import pytz

headers = {"accept": "application/json"}
url = "https://api.benzinga.com/api/v1/bulls_bears_say"
load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')


async def save_json(data, symbol, dir_path):
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, f"{symbol}.json")
    async with aiofiles.open(file_path, 'w') as file:
        await file.write(ujson.dumps(data))

async def get_data(session, ticker, con):
    querystring = {"token": api_key, "symbols": ticker}
    try:
        async with session.get(url, params=querystring, headers=headers) as response:
            data = ujson.loads(await response.text())['bulls_say_bears_say']
            data = [{k: v for k, v in item.items() if k not in ['id','securities','ticker','updated']} for item in data]
            await save_json(data, ticker, 'json/bull_vs_bear')

    except Exception as e:
        print(e)

async def run(stock_symbols, con):
    async with aiohttp.ClientSession() as session:
        tasks = [get_data(session, symbol, con) for symbol in stock_symbols]
        for f in tqdm(asyncio.as_completed(tasks), total=len(stock_symbols)):
            await f

try:
    
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    #stock_symbols = ['TSLA']

    asyncio.run(run(stock_symbols, con))
    
except Exception as e:
    print(e)
finally:
    con.close()