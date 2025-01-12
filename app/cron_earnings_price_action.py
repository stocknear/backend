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

async def calculate_price_reactions(filtered_data, price_history):
    # Ensure price_history is sorted by date
    price_history.sort(key=lambda x: datetime.strptime(x['time'], "%Y-%m-%d"))

    # Convert price history to a dictionary for quick lookup
    price_dict = {entry['time']: entry for entry in price_history}

    results = []

    for earnings in filtered_data:
        report_date = earnings['date']
        report_datetime = ny_tz.localize(datetime.strptime(report_date, "%Y-%m-%d"))

        # Initialize a dictionary for price reactions
        price_reactions = {'date': report_date, 'quarter': earnings['quarter'], 'year': earnings['year']}

        for offset in [0,1,2]:  # Days around earnings
            # Calculate initial target date with offset
            target_date = report_datetime - timedelta(days=offset)

            # Adjust target_date to the latest weekday if it falls on a weekend
            if target_date.weekday() == 5:  # Saturday
                target_date -= timedelta(days=1)  # Move to Friday
            elif target_date.weekday() == 6:  # Sunday
                target_date -= timedelta(days=2)  # Move to Friday

            target_date_str = target_date.strftime("%Y-%m-%d")
            while target_date_str not in price_dict:  # Ensure target_date exists in price_dict
                target_date -= timedelta(days=1)
                target_date_str = target_date.strftime("%Y-%m-%d")

            price_data = price_dict[target_date_str]

            # Find the previous day's price data
            previous_date = target_date - timedelta(days=1)
            if previous_date.weekday() == 5:  # Saturday
                previous_date -= timedelta(days=1)  # Move to Friday
            elif previous_date.weekday() == 6:  # Sunday
                previous_date -= timedelta(days=2)  # Move to Friday

            previous_date_str = previous_date.strftime("%Y-%m-%d")
            while previous_date_str not in price_dict:  # Ensure previous_date exists in price_dict
                previous_date -= timedelta(days=1)
                previous_date_str = previous_date.strftime("%Y-%m-%d")

            previous_price_data = price_dict[previous_date_str]

            # Calculate close price and percentage change
            price_reactions[f"{offset+1}_days_close"] = price_data['close']
            price_reactions[f"{offset+1}_days_change_percent"] = round(
                (price_data['close'] / previous_price_data['close'] - 1) * 100, 2
            )

            print(target_date_str, previous_date_str)
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
                        'date': item['date']
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

            results = await calculate_price_reactions(filtered_data, price_history)
            print(filtered_data[0])
            print(results[1])
            # Save the updated filtered_data
            #await save_json(filtered_data, ticker, 'json/earnings/past')
            
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
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    stock_symbols = ['AMD']

    asyncio.run(run(stock_symbols, con))
    
except Exception as e:
    print(e)
finally:
    con.close()