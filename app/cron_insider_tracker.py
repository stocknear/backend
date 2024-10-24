import ujson
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


async def save_json(data):
    async with async_open(f"json/tracker/insider/data.json", 'w') as file:
        await file.write(ujson.dumps(data))


async def get_data(session, symbols):
    res_list = []
    for page in range(0, 3):  # Adjust the number of pages as needed
        url = f"https://financialmodelingprep.com/api/v4/insider-trading?page={page}&apikey={api_key}"
        async with session.get(url) as response:
            try:
                if response.status == 200:
                    data = await response.json()
                    
                    # Filter and adjust transactionType based on acquistionOrDisposition
                    filtered_data = [
                        {
                            "reportingName": item.get("reportingName"),
                            "symbol": item.get("symbol"),
                            "filingDate": item.get("filingDate"),
                            "value": round(item.get("securitiesTransacted") * item.get("price"),2),
                            "transactionType": "Buy" if item.get("acquistionOrDisposition") == "A" 
                                                else "Sell" if item.get("acquistionOrDisposition") == "D" 
                                                else None,  # None if neither "A" nor "D"
                        }
                        for item in data
                        if item.get("acquistionOrDisposition") in ["A", "D"] and item.get('price') > 0 and item.get("securitiesTransacted") > 0  # Filter out if not "A" or "D"
                    ]
                    
                    res_list += filtered_data
                else:
                    print(f"Failed to fetch data. Status code: {response.status}")
            except Exception as e:
                print(f"Error while fetching data: {e}")
                break

    new_data = []
    for item in res_list:
        try:
            symbol = item['symbol']
            with open(f"json/quote/{symbol}.json") as file:
                stock_data = ujson.load(file)
                item['marketCap'] = stock_data['marketCap']
                item['price'] = round(stock_data['price'],2)
                item['changesPercentage'] = round(stock_data['changesPercentage'],2)
                new_data.append({**item})
        except:
            pass

    return new_data


async def run():
    # Connect to SQLite
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    
    # Fetch stock symbols
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    # Fetch data asynchronously using aiohttp
    async with aiohttp.ClientSession() as session:
        data = await get_data(session, stock_symbols)
        if len(data) > 0:
            print(data)
            print(f"Fetched {len(data)} records.")
            await save_json(data)


try:
    asyncio.run(run())
except Exception as e:
    print(f"Error: {e}")
