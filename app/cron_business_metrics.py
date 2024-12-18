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

def standardize_strings(string_list):
    return [string.title() for string in string_list]

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

import orjson
from datetime import datetime

def convert_to_dict(data):
    result = {}
    
    for entry in data:
        for date, categories in entry.items():
            if date not in result:
                result[date] = {}
            for category, amount in categories.items():
                result[date][category] = amount
                
    return result

def prepare_expense_dataset(symbol):
    # Define the list of key elements you want to track
    expense_keys = [
        'researchAndDevelopmentExpenses',
        'generalAndAdministrativeExpenses',
        'sellingAndMarketingExpenses',
        'operatingExpenses',
        'costOfRevenue'
    ]
    
    # Open the financial statement data for the symbol
    with open(f"json/financial-statements/income-statement/quarter/{symbol}.json", 'rb') as file:
        data = orjson.loads(file.read())
    # Convert the data into a dictionary
    
    # Initialize a dictionary to hold the history and growth for each key
    expense_data = {}
    
    for key in expense_keys:
        expense_data[key] = []
        
        # Prepare the data for the current key
        for entry in data:
            date = entry.get('date')
            value = entry.get(key, 0)  # Default to 0 if the key is missing
            expense_data[key].append({'date': date, 'value': value})
        
        # Sort the list by date
        expense_data[key] = sorted(expense_data[key], key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
        
        # Initialize 'valueGrowth' as None for all entries
        for item in expense_data[key]:
            item['valueGrowth'] = None

        # Calculate valueGrowth for each item based on the previous date value
        for i in range(1, len(expense_data[key])):
            try:
                current_item = expense_data[key][i]
                prev_item = expense_data[key][i - 1]
                growth = round(((current_item['value'] - prev_item['value']) / prev_item['value']) * 100, 2) if prev_item['value'] != 0 else None
                current_item['valueGrowth'] = growth
            except:
                current_item['valueGrowth'] = None

    # Return the results as a dictionary with all keys
    return expense_data

def prepare_geo_dataset(data):
    data = convert_to_dict(data)
    res_list = {}
    geo_name_list = []
    geo_history_list = []
    index = 0
    for date, info in data.items():
        value_list = []
        for name, val in info.items():
            if index == 0:
                geo_name_list.append(name)
            if name in geo_name_list:
                value_list.append(val)
        if len(value_list) > 0:
            geo_history_list.append({'date': date, 'value': value_list})
        index +=1

    geo_history_list = sorted(geo_history_list, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))

    # Initialize 'valueGrowth' as None for all entries
    for item in geo_history_list:
        item['valueGrowth'] = [None] * len(item['value'])

    # Calculate valueGrowth for each item based on the previous date value
    for i in range(1, len(geo_history_list)):  # Start from the second item
        current_item = geo_history_list[i]
        prev_item = geo_history_list[i - 1]
        
        value_growth = []
        for cur_value, prev_value in zip(current_item['value'], prev_item['value']):
            try:
                growth = round(((cur_value - prev_value) / prev_value) * 100, 2)
            except:
                growth = None
            value_growth.append(growth)
        
        current_item['valueGrowth'] = value_growth

    geo_history_list = sorted(geo_history_list, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=True)

    res_list = {'geographic': {'names': standardize_strings(geo_name_list), 'history': geo_history_list}}

    return res_list

def prepare_dataset(data, geo_data, symbol):
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
            try:
                growth = round(((cur_value - prev_value) / prev_value) * 100, 2)
            except:
                growth = None
            value_growth.append(growth)
        
        current_item['valueGrowth'] = value_growth

    revenue_history_list = sorted(revenue_history_list, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=True)

    res_list = {'revenue': {'names': revenue_name_list, 'history': revenue_history_list}}

    geo_data = prepare_geo_dataset(geo_data)
    #operating_expense_data = prepare_expense_dataset(symbol)


    #res_list = {**res_list, **geo_data, 'expense': operating_expense_data}
    res_list = {**res_list, **geo_data}
    return res_list

async def get_data(session, total_symbols):
    batch_size = 300  # Process 300 symbols at a time
    for i in tqdm(range(0, len(total_symbols), batch_size)):
        batch = total_symbols[i:i+batch_size]
        for symbol in batch:
            product_data = []
            geo_data = []

            urls = [f"https://financialmodelingprep.com/api/v4/revenue-product-segmentation?symbol={symbol}&structure=flat&period=quarter&apikey={api_key}",
                    f"https://financialmodelingprep.com/api/v4/revenue-geographic-segmentation?symbol={symbol}&structure=flat&apikey={api_key}"
                    ]

            for url in urls:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "product" in url:
                                product_data = data
                            else:
                                geo_data = data
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    pass

            if len(product_data) > 0 and len(geo_data) > 0:
                data = prepare_dataset(product_data, geo_data, symbol)
                await save_json(data, symbol)

        # Wait 60 seconds after processing each batch of 300 symbols
        if i + batch_size < len(total_symbols):
            print(f"Processed {i + batch_size} symbols, waiting 60 seconds...")
            await asyncio.sleep(60)

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    total_symbols = [row[0] for row in cursor.fetchall()]
    total_symbols = ['TSLA']  # For testing purposes
    con.close()
    
    async with aiohttp.ClientSession() as session:
        await get_data(session, total_symbols)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
