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
        try:
            for date, categories in entry.items():
                try:
                    if date not in result:
                        result[date] = {}
                    for category, amount in categories.items():
                        result[date][category] = amount
                except:
                    pass
        except:
            pass
                
    return result

async def save_json(data, symbol):
    with open(f"json/business-metrics/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))

def prepare_expense_dataset(data):
    data = convert_to_dict(data)
    res_list = {}
    operating_name_list = []
    operating_history_list = []
    index = 0
    for date, info in data.items():
        value_list = []
        for name, val in info.items():
            if index == 0:
                operating_name_list.append(name)
            if name in operating_name_list:
                value_list.append(val)
        if len(value_list) > 0:
            operating_history_list.append({'date': date, 'value': value_list})
        index += 1

    operating_history_list = sorted(operating_history_list, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))

    # Initialize 'valueGrowth' as None for all entries
    for item in operating_history_list:
        item['valueGrowth'] = [None] * len(item['value'])

    # Calculate valueGrowth for each item based on the previous date value
    for i in range(1, len(operating_history_list)):  # Start from the second item
        current_item = operating_history_list[i]
        prev_item = operating_history_list[i - 1]
        
        value_growth = []
        for cur_value, prev_value in zip(current_item['value'], prev_item['value']):
            try:
                growth = round(((cur_value - prev_value) / prev_value) * 100, 2)
            except:
                growth = None
            value_growth.append(growth)
        
        current_item['valueGrowth'] = value_growth

    operating_history_list = sorted(operating_history_list, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=True)

    res_list = {'operatingExpenses': {'names': operating_name_list, 'history': operating_history_list}}
    return res_list

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
        index += 1

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

def process_revenue_segmentation_data(data):
    """Process data from the new revenue product segmentation endpoint"""
    result = []
    
    # Convert the data into the format we need
    for item in data:
        entry = {
            item['date']: {}
        }
        
        # Extract categories from the data field
        if 'data' in item:
            for category, amount in item['data'].items():
                entry[item['date']][category] = amount
        
        result.append(entry)
    
    return result

def process_geographic_segmentation_data(data):
    """Process data from the geographic segmentation endpoint"""
    result = []
    
    # Convert the data into the format we need
    for item in data:
        entry = {
            item['date']: {}
        }
        
        # Extract geographic data
        if 'data' in item:
            for region, amount in item['data'].items():
                entry[item['date']][region] = amount
        
        result.append(entry)
    
    return result

def prepare_dataset(data, geo_data, income_data, symbol, rev_segment_data=None):
    data = convert_to_dict(data)
    
    # If we have revenue segmentation data, use it instead of the original product data
    if rev_segment_data and len(rev_segment_data) > 0:
        rev_segment_processed = process_revenue_segmentation_data(rev_segment_data)
        data = convert_to_dict(rev_segment_processed)
    
    res_list = {}
    revenue_name_list = []
    revenue_history_list = []
    index = 0
    for date, info in data.items():
        try:
            value_list = []
            for name, val in info.items():
                try:
                    if index == 0:
                        revenue_name_list.append(name)
                    if name in revenue_name_list:
                        value_list.append(val)
                except:
                    pass
            if len(value_list) > 0:
                revenue_history_list.append({'date': date, 'value': value_list})
            index += 1
        except:
            pass

    revenue_history_list = sorted(revenue_history_list, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))

    # Initialize 'valueGrowth' as None for all entries
    for item in revenue_history_list:
        try:
            item['valueGrowth'] = [None] * len(item['value'])
        except:
            pass

    # Calculate valueGrowth for each item based on the previous date value
    for i in range(1, len(revenue_history_list)):
        try:
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
        except:
            pass

    revenue_history_list = sorted(revenue_history_list, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=True)

    res_list = {'revenue': {'names': revenue_name_list, 'history': revenue_history_list}}

    geo_data = prepare_geo_dataset(geo_data)
    operating_expense_data = prepare_expense_dataset(income_data)

    res_list = {**res_list, **geo_data, **operating_expense_data}
    return res_list

async def get_data(session, total_symbols):
    batch_size = 300  # Process 300 symbols at a time
    for i in tqdm(range(0, len(total_symbols), batch_size)):
        batch = total_symbols[i:i+batch_size]
        for symbol in batch:
            try:
                with open(f"json/financial-statements/income-statement/quarter/{symbol}.json",'r') as file:
                    income_data = orjson.loads(file.read())

                include_selling_and_marketing = income_data[0].get('sellingAndMarketingExpenses', 0) > 0 if income_data else False
                # Process the income_data
                income_data = [
                    {
                        'date': entry['date'],
                        'Selling, General, and Administrative': entry.get('sellingGeneralAndAdministrativeExpenses', 0),
                        'Research and Development': entry.get('researchAndDevelopmentExpenses', 0),
                        **({'Sales and Marketing': entry.get('sellingAndMarketingExpenses', 0)} if include_selling_and_marketing else {})
                    }
                    for entry in income_data
                    if datetime.strptime(entry['date'], '%Y-%m-%d') > datetime(2015, 1, 1)
                ]

                income_data = [
                    {
                        entry['date']: {
                            key: value
                            for key, value in entry.items()
                            if key != 'date'
                        }
                    }
                    for entry in income_data
                ]
            except:
                income_data = []
               
            product_data = []
            geo_data = []
            revenue_segmentation_data = []

            urls = [
                f"https://financialmodelingprep.com/stable/revenue-product-segmentation?symbol={symbol}&period=quarter&apikey={api_key}",
                f"https://financialmodelingprep.com/stable/revenue-geographic-segmentation?symbol={symbol}&period=quarter&apikey={api_key}"
            ]

            for url in urls:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "revenue-geographic-segmentation" in url:
                                geo_data = process_geographic_segmentation_data(data)
                            if "revenue-product-segmentation" in url:
                                revenue_segmentation_data = data
                except Exception as e:
                    print(f"Error fetching data for {symbol} from {url}: {e}")
                    pass

            # Only save data if we have at least one type of data
            if len(product_data) > 0 or len(geo_data) > 0 or len(revenue_segmentation_data) > 0 or len(income_data) > 0:
                data = prepare_dataset(product_data, geo_data, income_data, symbol, revenue_segmentation_data)
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
    #Testing
    #total_symbols = ['AAPL']
    con.close()

    async with aiohttp.ClientSession() as session:
        await get_data(session, total_symbols)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())