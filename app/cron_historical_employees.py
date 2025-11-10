import ujson
import asyncio
import aiohttp
import aiofiles
import sqlite3
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')


async def save_json_data(symbol, data):
    folder_path = "json/historical-employees"
    os.makedirs(folder_path, exist_ok=True)  # Ensure the folder exists

    file_path = f"{folder_path}/{symbol}.json"
    async with aiofiles.open(file_path, 'w') as file:
        await file.write(ujson.dumps(data))

async def get_data(symbols, session):
    tasks = []
    for symbol in symbols:
        task = asyncio.create_task(get_endpoints(symbol, session))
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    
    if len(responses) > 0:
        for symbol, response in zip(symbols, responses):
            if response:
                await save_json_data(symbol, response)

async def get_endpoints(symbol, session):
    data = []
    try:
        url= f"https://financialmodelingprep.com/stable/employee-count?symbol={symbol}&apikey={api_key}"
        
        async with session.get(url) as response:
            data = []
            data = await response.json()

            res = []

            for item in data:
                try:
                    res.append({'date': item['periodOfReport'], 'employeeCount': item['employeeCount']})
                except:
                    pass

            return res

                
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")

    return data



async def run():
    try:
        con = sqlite3.connect('stocks.db')
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks")
        stock_symbols = [row[0] for row in cursor.fetchall()]
        con.close()

        total_symbols = stock_symbols
        chunk_size = 1000

    except Exception as e:
        print(f"Failed to fetch symbols: {e}")
        return

    try:
        connector = aiohttp.TCPConnector(limit=100)  # Adjust the limit as needed
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in range(0, len(total_symbols), chunk_size):
                symbols_chunk = total_symbols[i:i + chunk_size]
                await get_data(symbols_chunk, session)
                print('sleeping for 60 sec')
                await asyncio.sleep(60)  # Wait for 60 seconds between chunks
    except Exception as e:
        print(f"Failed to run fetch and save data: {e}")

try:
    asyncio.run(run())
except Exception as e:
    print(e)

