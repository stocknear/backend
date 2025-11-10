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


async def fetch_sec_filings(session, symbol, filing_type):
    url = f"https://financialmodelingprep.com/api/v3/sec_filings/{symbol}?type={filing_type}&page=0&apikey={api_key}"
    async with session.get(url) as response:
        data = await response.json()
    return [{'date': entry['fillingDate'], 'type': entry['type'],'link': entry['finalLink']} for entry in data]

async def save_sec_filings(session, symbol):
    tasks = [
        fetch_sec_filings(session, symbol, '8-k'),
        fetch_sec_filings(session, symbol, '10-k'),
        fetch_sec_filings(session, symbol, '10-q')
    ]

    res_eight_k, res_ten_k, res_ten_q = await asyncio.gather(*tasks)

    if len(res_eight_k) == 0 and len(res_ten_k) == 0 and len(res_ten_q) == 0:
        pass
    else:
        res = {'eightK': res_eight_k, 'tenK': res_ten_k, 'tenQ': res_ten_q}
        with open(f"json/sec-filings/{symbol}.json", 'w') as file:
            ujson.dump(res, file)


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
            tasks.append(save_sec_filings(session, symbol))

            i += 1
            if i % 300 == 0:
                await asyncio.gather(*tasks)
                tasks = []
                print('sleeping mode: ', i)
                await asyncio.sleep(60)  # Pause for 60 seconds

        #tasks.append(self.save_ohlc_data(session, "%5EGSPC"))
        
        if tasks:
            await asyncio.gather(*tasks)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())
