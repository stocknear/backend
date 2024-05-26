import ujson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
import pytz
from aiofiles import open as async_open

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

# Function to check if the year is at least 2015
def is_at_least_2015(date_string):
    year = datetime.strptime(date_string, "%Y-%m-%d").year
    return year >= 2015

async def get_statistics_endpoints(session, symbol):
    url = f"https://financialmodelingprep.com/api/v4/insider-roaster-statistic?symbol={symbol}&apikey={api_key}"
    async with session.get(url) as response:
        if response.status == 200:
            return symbol, await response.json()
        else:
            return symbol, []

async def get_insider_trading_endpoints(session, symbol):
    aggregated_data = []
    for page in range(101):  # Pages from 0 to 100
        url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={symbol}&page={page}&apikey={api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if not data:
                    break  # Break if the result is empty
                aggregated_data.extend(data)
            else:
                break  # Break if response status is not 200
    filtered_data = [item for item in aggregated_data if is_at_least_2015(item["transactionDate"][:10])]

    if len(filtered_data) > 0:
        await save_insider_trading_as_json(symbol, filtered_data)



async def save_statistics_as_json(symbol, data):
    async with async_open(f"json/insider-trading/statistics/{symbol}.json", 'w') as file:
        await file.write(ujson.dumps(data))

async def save_insider_trading_as_json(symbol, data):
    async with async_open(f"json/insider-trading/history/{symbol}.json", 'w') as file:
        await file.write(ujson.dumps(data))

async def aggregate_statistics(symbol, data):
    aggregated_data = {}
    for entry in data:
        year = entry['year']
        if year >= 2015:
            if year not in aggregated_data:
                aggregated_data[year] = {
                    'year': year,
                    'totalBought': 0,
                    'totalSold': 0,
                    'purchases': 0,
                    'sales': 0
                }
            aggregated_data[year]['totalBought'] += entry['totalBought']
            aggregated_data[year]['totalSold'] += entry['totalSold']
            aggregated_data[year]['purchases'] += entry['purchases']
            aggregated_data[year]['sales'] += entry['sales']

    res = list(aggregated_data.values())

    await save_statistics_as_json(symbol, res)


async def process_symbols(session, symbols):
    #History
    tasks = [get_insider_trading_endpoints(session, symbol) for symbol in symbols]
    await asyncio.gather(*tasks)
    
    #Statistics
    
    tasks = [get_statistics_endpoints(session, symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    for symbol, data in results:
        if data:
            await aggregate_statistics(symbol, data)
    

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    chunk_size = len(stock_symbols) // 70  # Divide the list into N chunks
    chunks = [stock_symbols[i:i + chunk_size] for i in range(0, len(stock_symbols), chunk_size)]

    async with aiohttp.ClientSession() as session:
        for chunk in chunks:
            await process_symbols(session, chunk)
            await asyncio.sleep(60)
            print('sleep')


try:
    asyncio.run(run())
except Exception as e:
    print(e)
