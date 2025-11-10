from datetime import datetime, timedelta
import ujson
import time
import sqlite3
import asyncio
import aiohttp
import random
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

async def save_json(symbol, data):
    with open(f"json/financial-score/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


async def get_data(session, symbol):
    # Construct the API URL
    url = f"https://financialmodelingprep.com/api/v4/score?symbol={symbol}&apikey={api_key}"
    
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if len(data) > 0:
                    filtered_data = [
                        {
                            k: round(v, 2) if k == 'altmanZScore' else v 
                            for k, v in item.items() 
                            if k in ['altmanZScore', 'piotroskiScore', 'workingCapital', 'totalAssets']
                        } 
                        for item in data
                    ]
                    await save_json(symbol, filtered_data[0])
    except:
        pass

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, symbol in enumerate(tqdm(symbols), 1):
            tasks.append(get_data(session, symbol))
            if i % 100 == 0:
                await asyncio.gather(*tasks)
                tasks = []
                print(f'sleeping mode: {i}')
                await asyncio.sleep(60)  # Pause for 60 seconds
        
        if tasks:
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())