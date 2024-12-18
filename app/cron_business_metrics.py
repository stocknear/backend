from datetime import datetime, timedelta
import orjson
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

def convert_to_dict(data):
    result = {}
    
    for entry in data:
        for date, categories in entry.items():
            if date not in result:
                result[date] = {}
            for category, amount in categories.items():
                result[date][category] = amount
                
    return result

async def save_json(data, symbol):
    with open(f"json/business-metrics/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))

def prepare_dataset(data):
    data = convert_to_dict(data)
    res_list = {}
    revenue_name_list = []
    revenue_history_list = []
    index = 0
    for date, info in data.items():
        value_list = []
        for name, val in info.items():
            if index == 0:
                revenue_name_list.append(name)
            if name in revenue_name_list:
                value_list.append(val)
        if len(value_list) > 0:
            revenue_history_list.append({'date': date, 'value': value_list})
        index +=1


    revenue_history_list = sorted(revenue_history_list, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))

   # Initialize 'valueGrowth' as None for all entries
    for item in revenue_history_list:
        item['valueGrowth'] = [None] * len(item['value'])

    # Calculate valueGrowth for each item based on the previous date value
    for i in range(1, len(revenue_history_list)):  # Start from the second item
        current_item = revenue_history_list[i]
        prev_item = revenue_history_list[i - 1]
        
        value_growth = []
        for cur_value, prev_value in zip(current_item['value'], prev_item['value']):
            growth = round(((cur_value - prev_value) / prev_value) * 100, 2)
            value_growth.append(growth)
        
        current_item['valueGrowth'] = value_growth


    revenue_history_list = sorted(revenue_history_list, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=True)


    res_list = {'revenue': {'names': revenue_name_list, 'history': revenue_history_list}}

    return res_list


async def get_data(session, total_symbols):
    for symbol in total_symbols:
        url = f"https://financialmodelingprep.com/api/v4/revenue-product-segmentation?symbol={symbol}&structure=flat&period=quarter&apikey={api_key}"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if len(data) > 0:
                        data = prepare_dataset(data)
                        await save_json(data, symbol)

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            pass


async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    total_symbols = [row[0] for row in cursor.fetchall()]
    total_symbols = ['AAPL']  # For testing purposes
    con.close()
    
    async with aiohttp.ClientSession() as session:
        await get_data(session, total_symbols)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
