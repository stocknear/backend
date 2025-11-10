from datetime import datetime, timedelta
import ujson
import time
import sqlite3
import time 
import asyncio
import aiohttp
import random
from tqdm import tqdm
from utils.helper import get_last_completed_quarter

from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv('FMP_API_KEY')


quarter, year = get_last_completed_quarter()


async def get_data(session, symbol, max_retries=3, initial_delay=1):
    url = f"https://financialmodelingprep.com/stable/institutional-ownership/symbol-positions-summary?symbol={symbol}&year={year}&quarter={quarter}&apikey={api_key}"
    
    for attempt in range(max_retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        data = await response.json()
                        if len(data) > 0:
                            await save_json(symbol, data[0])
                            print(data[0])
                        return
                    else:
                        print(f"Unexpected content type for {symbol}: {content_type}")
                elif response.status == 504:
                    if attempt < max_retries - 1:
                        delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Gateway Timeout for {symbol}. Retrying in {delay:.2f} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"Max retries reached for {symbol} after Gateway Timeout")
                else:
                    print(f"Error fetching data for {symbol}: HTTP {response.status}")
                    return
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            else:
                print(f"Max retries reached for {symbol}")


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
