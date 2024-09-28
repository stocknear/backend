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
load_dotenv()
api_key = os.getenv('NASDAQ_API_KEY')


# Get today's date
today = datetime.now()
# Calculate the date six months ago
six_months_ago = today - timedelta(days=6*30)  # Rough estimate, can be refined

query_stock_template = """
    SELECT 
        name, price, volume
    FROM 
        stocks 
    WHERE
        symbol = ?
"""

query_etf_template = """
    SELECT 
        name, price, volume
    FROM 
        etfs 
    WHERE
        symbol = ?
"""


async def save_json(symbol, data):
    with open(f"json/retail-volume/companies/{symbol}.json", 'w') as file:
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


async def get_data(ticker_list):
    ticker_str = ','.join(ticker_list)
    async with aiohttp.ClientSession() as session:
        url = f"https://data.nasdaq.com/api/v3/datatables/NDAQ/RTAT?api_key={api_key}&ticker={ticker_str}"
        async with session.get(url) as response:
            if response.status == 200:
                data = (await response.json())['datatable']['data']
                return data
            else:
                return []



async def run():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]


    
    total_symbols = stocks_symbols+etf_symbols
    
    chunk_size = len(total_symbols) // 700  # Divide the list into N chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
    
    most_retail_volume = []

    for chunk in tqdm(chunks):
        data = await get_data(chunk)
        # Transforming the list of lists into a list of dictionaries
        transformed_data = [
            {
                'date': entry[0],
                'symbol': entry[1],
                'traded': entry[2]*30*10**9, #data is normalized to $30B per day
                'sentiment': entry[3]
            }
            for entry in data
        ]
        for symbol in chunk:
            try:
                filtered_data = [item for item in transformed_data if symbol == item['symbol']]
                res = filter_past_six_months(filtered_data)
                query_template = query_stock_template if symbol in stocks_symbols else query_etf_template
                connection = con if symbol in stocks_symbols else etf_con

                #Compute strength of retail investors
                last_trade = res[-1]['traded']
                last_sentiment = int(res[-1]['sentiment'])
                last_date = res[-1]['date']
                data = pd.read_sql_query(query_template, connection, params=(symbol,))
                price = float(data['price'].iloc[0])
                retail_volume = int(last_trade/price)
                total_volume = int(data['volume'].iloc[0])
                retailer_strength = round(((retail_volume/total_volume))*100,2)
                name = data['name'].iloc[0]

                company_data = {'lastDate': last_date, 'lastTrade': last_trade, 'lastSentiment': last_sentiment, 'retailStrength': retailer_strength, 'history': res}
                await save_json(symbol, company_data)

                #Add stocks for most retail volume
                if symbol in stocks_symbols:
                    most_retail_volume.append({'symbol': res[-1]['symbol'], 'name': name, 'assetType': 'stocks','traded': res[-1]['traded'], 'sentiment': res[-1]['sentiment'], 'retailStrength': retailer_strength})
                elif symbol in etf_symbols:
                    most_retail_volume.append({'symbol': res[-1]['symbol'], 'name': name, 'assetType': 'etf', 'traded': res[-1]['traded'], 'sentiment': res[-1]['sentiment'], 'retailStrength': retailer_strength})
            except Exception as e:
                print(e)

        
    most_retail_volume = [item for item in most_retail_volume if item['retailStrength'] <= 100]
    most_retail_volume = sorted(most_retail_volume, key=lambda x: x['traded'], reverse=True)[:100] # top 100 retail volume stocks
    with open(f"json/retail-volume/data.json", 'w') as file:
        ujson.dump(most_retail_volume, file)

    con.close()
    etf_con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)