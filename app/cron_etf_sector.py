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
directory_path = "json/etf-sector"



async def get_data(session, symbol):
    url = f"https://financialmodelingprep.com/api/v3/etf-sector-weightings/{symbol}?apikey={api_key}"
    res_list = []
    try:
        async with session.get(url) as response:
            data = await response.json()
            if len(data) > 0:
                for item in data:
                    try:
                        if 'sector' in item and 'weightPercentage' in item:
                            res_list.append({'sector': item['sector'], 'weightPercentage': round(float(item['weightPercentage'].replace("%","")),2)})
                    except:
                        pass
                res_list = sorted(res_list, key=lambda x: x['weightPercentage'], reverse=True)
                if res_list:
                    save_json(res_list, symbol)  # Removed await since it's not async
                
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")

def save_json(data, symbol):
    os.makedirs(directory_path, exist_ok=True)
    file_path = f"{directory_path}/{symbol}.json"
    try:
        with open(file_path, 'w') as file:  # Changed to text mode since we're using ujson
            ujson.dump(data, file)  # Added the file argument
    except Exception as e:
        print(f"Error saving JSON for {symbol}: {str(e)}")

async def run():
    con = sqlite3.connect('etf.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM etfs")
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

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())