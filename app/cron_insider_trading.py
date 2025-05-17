import orjson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
from aiofiles import open as async_open
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')


keys_to_remove_insider_history = {"symbol", "link", "filingDate", "reportingCik"}
keys_to_remove_insider_statistics = {"symbol", "cik", "purchases", "sales", "pPurchases", "sSales"}

def save_json(symbol, data):
    path = f"json/insider-trading/history"
    os.makedirs(path, exist_ok=True)  # Ensure directory exists

    with open(f"json/insider-trading/history/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))



async def get_insider_trading_endpoints(session, symbol):
    res = []
    for page in range(10):  # Pages from 0 to 100
        url = f"https://financialmodelingprep.com/stable/insider-trading/search?symbol={symbol}&page={page}&limit=100&apikey={api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if not data:
                    break  # Break if the result is empty
                res += data
            else:
                break  # Break if response status is not 200


    if len(res) > 0:
        res = [{k: v for k, v in item.items() if k not in keys_to_remove_insider_history} for item in res]
        save_json(symbol, res)



async def process_symbols(session, symbols):
    #History
    tasks = [get_insider_trading_endpoints(session, symbol) for symbol in symbols]
    await asyncio.gather(*tasks)
    
    #Statistics
    '''
    tasks = [get_statistics_endpoints(session, symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    for symbol, data in results:
        if data:
            filtered_data = [{k: v for k, v in item.items() if k not in keys_to_remove_insider_statistics} for item in data]
            await save_statistics_as_json(symbol, filtered_data)
    '''

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    chunk_size = len(stock_symbols) // 80  # Divide the list into N chunks
    chunks = [stock_symbols[i:i + chunk_size] for i in range(0, len(stock_symbols), chunk_size)]

    async with aiohttp.ClientSession() as session:
        for chunk in tqdm(chunks):
            await process_symbols(session, chunk)
            await asyncio.sleep(60)
           
try:
    asyncio.run(run())
except Exception as e:
    print(e)
