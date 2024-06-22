from datetime import datetime, timedelta
import ujson
import time
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
import time 
import asyncio
import aiohttp
from faker import Faker
from tqdm import tqdm

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')


async def get_data(session, symbol):
    url = f"https://financialmodelingprep.com/api/v4/institutional-ownership/symbol-ownership?symbol={symbol}&includeCurrentQuarter=true&apikey={api_key}"
    async with session.get(url) as response:
        data = await response.json()
        if len(data) > 0:
            await save_json(symbol, data[0]) #return only the latest ownership stats


async def save_json(symbol, data):
    with open(f"json/ownership-stats/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

async def run():
    
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    

    async with aiohttp.ClientSession() as session:
        tasks = []
        i = 0
        for symbol in tqdm(symbols):
            tasks.append(get_data(session, symbol))
            i += 1
            if i % 400 == 0:
                await asyncio.gather(*tasks)
                tasks = []
                print('sleeping mode: ', i)
                await asyncio.sleep(60)  # Pause for 60 seconds
        
        if tasks:
            await asyncio.gather(*tasks)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())
