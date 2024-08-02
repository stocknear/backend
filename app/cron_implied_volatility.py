import ujson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime,timedelta
from tqdm import tqdm
import pandas as pd
import time

from dotenv import load_dotenv
import os

keys_to_keep = {'date','stockpx', 'iv60', 'iv90', '252dclshv','60dorhv'}

load_dotenv()
api_key = os.getenv('NASDAQ_API_KEY')


# Get today's date
today = datetime.today()
# Calculate the date 12 months ago
dates = [today - timedelta(days=i) for i in range(365)]
date_str = ','.join(date.strftime('%Y-%m-%d') for date in dates)

async def save_json(symbol, data):
    with open(f"json/implied-volatility/companies/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


# Function to filter the list
def filter_past_six_months(data):
    filtered_data = []
    for entry in data:
        entry_date = datetime.strptime(entry['date'], '%Y-%m-%d')
        if entry_date >= six_months_ago:
            filtered_data.append(entry)
    sorted_data = sorted(filtered_data, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
    return sorted_data

async def get_data(ticker):
    #ticker_str = ','.join(ticker_list)
    
    async with aiohttp.ClientSession() as session:
        url = url = f"https://data.nasdaq.com/api/v3/datatables/ORATS/OPT?date={date_str}&ticker={ticker}&api_key={api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                res = await response.json()
                data = res['datatable']['data']
                columns = res['datatable']['columns']
                return data, columns
            else:
                return [], []


async def run():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketcap >=10E6 AND symbol NOT LIKE '%.%'")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]


    
    total_symbols = stocks_symbols+etf_symbols
    
    chunk_size = len(total_symbols) // 70  # Divide the list into N chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
    
    for ticker in tqdm(total_symbols):
        data, columns = await get_data(ticker)
        filtered_data = []
        for element in tqdm(data):
            # Assuming the number of columns matches the length of each element in `data`
            filtered_data.append({columns[i]["name"]: element[i] for i in range(len(columns))})

        filtered_data = [{k: v for k, v in item.items() if k in keys_to_keep} for item in filtered_data]

        try:
            sorted_data = sorted(filtered_data, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
            if len(sorted_data) > 0:
                await save_json(ticker, sorted_data)
        except Exception as e:
            print(e)

        '''
        for symbol in chunk:
            try:
                filtered_data = [item for item in transformed_data if symbol == item['ticker']]               
                sorted_data = sorted(filtered_data, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
                if len(sorted_data) > 0:
                    await save_json(symbol, sorted_data)
            except Exception as e:
                print(e)
        '''
    
    con.close()
    etf_con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)