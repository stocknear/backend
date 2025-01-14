import aiohttp
import aiofiles
import ujson
import orjson
import sqlite3
import asyncio
import pandas as pd
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from ta.momentum import *
from tqdm import tqdm
import pytz

headers = {"accept": "application/json"}
url = "https://api.benzinga.com/api/v2.1/calendar/earnings"
load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

ny_tz = pytz.timezone('America/New_York')
today = datetime.now(ny_tz).replace(hour=0, minute=0, second=0, microsecond=0)
min_date = ny_tz.localize(datetime.strptime("2020-01-01", "%Y-%m-%d"))
N_days_ago = today - timedelta(days=10)



async def save_json(data, symbol, dir_path):
    file_path = os.path.join(dir_path, f"{symbol}.json")
    async with aiofiles.open(file_path, 'w') as file:
        await file.write(ujson.dumps(data))


from datetime import datetime, timedelta
import pytz

ny_tz = pytz.timezone("America/New_York")

async def compute_rsi(price_history, time_period=14):
    df_price = pd.DataFrame(price_history)
    df_price['rsi'] = rsi(df_price['close'], window=time_period)
    result = df_price.to_dict(orient='records')
    return result
    

async def calculate_price_reactions(ticker, filtered_data, price_history):
    # Ensure price_history is sorted by date
    price_history.sort(key=lambda x: x['time'])

    results = []

    with open(f"json/implied-volatility/{ticker}.json",'r') as file:
        iv_data = ujson.load(file)

    for item in filtered_data:
        report_date = item['date']

        # Find the index of the report date in the price history
        report_index = next((i for i, entry in enumerate(price_history) if entry['time'] == report_date), None)
        
        if report_index is None:
            continue  # Skip if report date is not found in the price history

        # Initialize a dictionary for price reactions
        iv_value = next((entry['implied_volatility'] for entry in iv_data if entry['date'] == report_date), None)

        #if iv_value is None:
        #    continue  # Skip if no matching iv_data is found for the report_date

        price_reactions = {
            'date': report_date,
            'quarter': item['quarter'],
            'year': item['year'],
            'time': item['time'],
            'rsi': int(price_history[report_index]['rsi']),
            'iv': iv_value,
        }

        

        for offset in [-4,-3,-2,-1,0,1,2,3,4,6]:
            target_index = report_index + offset

            # Ensure the target index is within bounds
            if 0 <= target_index < len(price_history):
                target_price_data = price_history[target_index]
                previous_index = target_index - 1

        

                # Ensure the previous index is within bounds
                if 0 <= previous_index < len(price_history):
                    previous_price_data = price_history[previous_index]

                    # Calculate close price and percentage change
                    direction = "forward" if offset >= 0 else "backward"
                    days_key = f"{direction}_{abs(offset)}_days"

                    if offset != 1:
                        price_reactions[f"{days_key}_close"] = target_price_data['close']
                        price_reactions[f"{days_key}_change_percent"] = round(
                            (target_price_data['close'] / previous_price_data['close'] - 1) * 100, 2
                        )

                    if offset ==1:
                        price_reactions['open'] = target_price_data['open']
                        price_reactions['high'] = target_price_data['high']
                        price_reactions['low'] = target_price_data['low']
                        price_reactions['close'] = target_price_data['close']

                        price_reactions[f"open_change_percent"] = round((target_price_data['open'] / previous_price_data['close'] - 1) * 100, 2)
                        price_reactions[f"high_change_percent"] = round((target_price_data['high'] / previous_price_data['close'] - 1) * 100, 2)
                        price_reactions[f"low_change_percent"] = round((target_price_data['low'] / previous_price_data['close'] - 1) * 100, 2)
                        price_reactions[f"close_change_percent"] = round((target_price_data['close'] / previous_price_data['close'] - 1) * 100, 2)


        results.append(price_reactions)

    return results

async def get_past_data(data, ticker, con):
    # Filter data based on date constraints
    filtered_data = []
    for item in data:
        try:
            item_date = ny_tz.localize(datetime.strptime(item["date"], "%Y-%m-%d"))
            if min_date <= item_date <= today:
                filtered_data.append(
                    {   
                        'revenue': float(item['revenue']),
                        'revenueEst': float(item['revenue_est']),
                        'revenueSurprisePercent': round(float(item['revenue_surprise_percent'])*100, 2),
                        'eps': round(float(item['eps']), 2),
                        'epsEst': round(float(item['eps_est']), 2),
                        'epsSurprisePercent': round(float(item['eps_surprise_percent'])*100, 2),
                        'year': item['period_year'],
                        'quarter': item['period'],
                        'date': item['date'],
                        'time': item['time']
                    }
                )
        except:
            pass

    # Sort the filtered data by date
    if len(filtered_data) > 0:
        filtered_data.sort(key=lambda x: x['date'], reverse=True)

        try:
            # Load the price history data
            with open(f"json/historical-price/max/{ticker}.json") as file:
                price_history = orjson.loads(file.read())

            price_history = await compute_rsi(price_history)
            results = await calculate_price_reactions(ticker, filtered_data, price_history)
            #print(results[0])
            await save_json(results, ticker, 'json/earnings/past')
            
        except:
            pass


async def get_data(session, ticker, con):
    querystring = {"token": api_key, "parameters[tickers]": ticker}
    try:
        async with session.get(url, params=querystring, headers=headers) as response:
            data = ujson.loads(await response.text())['earnings']
            
            await get_past_data(data, ticker, con)
            
    except Exception as e:
        print(e)
        #pass

async def run(stock_symbols, con):
    async with aiohttp.ClientSession() as session:
        tasks = [get_data(session, symbol, con) for symbol in stock_symbols]
        for f in tqdm(asyncio.as_completed(tasks), total=len(stock_symbols)):
            await f

try:
    
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    stock_symbols = ['AMD']

    asyncio.run(run(stock_symbols, con))
    
except Exception as e:
    print(e)
finally:
    con.close()