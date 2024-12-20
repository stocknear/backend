import aiohttp
import aiofiles
import ujson
import orjson
import sqlite3
import asyncio
import pandas as pd
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
min_date = ny_tz.localize(datetime.strptime("2015-01-01", "%Y-%m-%d"))
N_days_ago = today - timedelta(days=10)

query_template = """
    SELECT date, open, high, low, close
    FROM "{ticker}"
    WHERE date >= ?
"""


def check_existing_file(ticker, folder_name):
    file_path = f"json/earnings/{folder_name}/{ticker}.json"
    still_new = False
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                existing_data = ujson.load(file)
                date_obj = datetime.strptime(existing_data['date'], "%Y-%m-%d")
                if date_obj.tzinfo is None:
                    date_obj = date_obj.replace(tzinfo=pytz.UTC)

                if folder_name == 'surprise':
                    if date_obj+timedelta(1) >= N_days_ago:
                        still_new = True
                elif folder_name == 'next':
                    if date_obj+timedelta(1) >= today:
                        still_new = True

            if still_new == False:
                os.remove(file_path)
                print(f"Deleted file for {ticker}.")
        except Exception as e:
            print(f"Error processing existing file for {ticker}: {e}")


async def save_json(data, symbol, dir_path):
    file_path = os.path.join(dir_path, f"{symbol}.json")
    async with aiofiles.open(file_path, 'w') as file:
        await file.write(ujson.dumps(data))

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

            # Convert price_history dates to datetime objects for easy comparison
            price_history_dict = {
                datetime.strptime(item['time'], "%Y-%m-%d"): item for item in price_history
            }

            # Calculate volatility for each earnings release
            for entry in filtered_data:
                earnings_date = datetime.strptime(entry['date'], "%Y-%m-%d")
                volatility_prices = []

                # Collect prices from (X-2) to (X+1)
                for i in range(-2, 2):
                    current_date = earnings_date + timedelta(days=i)
                    if current_date in price_history_dict:
                        volatility_prices.append(price_history_dict[current_date])

                # Calculate volatility if we have at least one price entry
                if volatility_prices:
                    high_prices = [day['high'] for day in volatility_prices]
                    low_prices = [day['low'] for day in volatility_prices]
                    close_prices = [day['close'] for day in volatility_prices]

                    max_high = max(high_prices)
                    min_low = min(low_prices)
                    avg_close = sum(close_prices) / len(close_prices)

                    # Volatility percentage calculation
                    volatility = round(((max_high - min_low) / avg_close) * 100, 2)
                else:
                    volatility = None  # No data available for volatility calculation

                # Add the volatility to the entry
                entry['volatility'] = volatility

            # Save the updated filtered_data
            await save_json(filtered_data, ticker, 'json/earnings/past')

        except:
            pass


async def get_data(session, ticker, con):
    querystring = {"token": api_key, "parameters[tickers]": ticker}
    try:
        async with session.get(url, params=querystring, headers=headers) as response:
            data = ujson.loads(await response.text())['earnings']
            
            await get_past_data(data, ticker, con)

            # Filter for future earnings
            future_dates = [item for item in data if ny_tz.localize(datetime.strptime(item["date"], "%Y-%m-%d")) >= today]
            if future_dates:
                nearest_future = min(future_dates, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
                try:
                    symbol = nearest_future['ticker']
                    time = nearest_future['time']
                    date = nearest_future['date']
                    eps_prior = float(nearest_future['eps_prior']) if nearest_future['eps_prior'] else None
                    eps_est = float(nearest_future['eps_est']) if nearest_future['eps_est'] else None
                    revenue_est = float(nearest_future['revenue_est']) if nearest_future['revenue_est'] else None
                    revenue_prior = float(nearest_future['revenue_prior']) if nearest_future['revenue_prior'] else None
                    if revenue_est is not None and revenue_prior is not None and eps_prior is not None and eps_est is not None:
                        res_list = {
                            'date': date,
                            'time': time,
                            'epsPrior': eps_prior,
                            'epsEst': eps_est,
                            'revenuePrior': revenue_prior,
                            'revenueEst': revenue_est
                        }
                        await save_json(res_list, symbol, 'json/earnings/next')
                except Exception as e:
                    print(e)
                    pass
                else:
                    check_existing_file(ticker, "next")

                # Filter for past earnings within the last 20 days
                recent_dates = [item for item in data if N_days_ago <= ny_tz.localize(datetime.strptime(item["date"], "%Y-%m-%d")) <= today]
                if recent_dates:
                    nearest_recent = min(recent_dates, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
                    try:
                        date = nearest_recent['date']
                        eps_prior = float(nearest_recent['eps_prior']) if nearest_recent['eps_prior'] != '' else None
                        eps_surprise = float(nearest_recent['eps_surprise']) if nearest_recent['eps_surprise'] != '' else None
                        eps = float(nearest_recent['eps']) if nearest_recent['eps'] != '' else None
                        revenue_prior = float(nearest_recent['revenue_prior']) if nearest_recent['revenue_prior'] != '' else None
                        revenue_surprise = float(nearest_recent['revenue_surprise']) if nearest_recent['revenue_surprise'] != '' else None
                        revenue = float(nearest_recent['revenue']) if nearest_recent['revenue'] != '' else None
                        if revenue is not None and revenue_prior is not None and eps_prior is not None and eps is not None and revenue_surprise is not None and eps_surprise is not None:
                            res_list = {
                                'epsPrior':eps_prior,
                                'epsSurprise': eps_surprise,
                                'eps': eps,
                                'revenuePrior': revenue_prior,
                                'revenueSurprise': revenue_surprise,
                                'revenue': revenue,
                                'date': date,
                                }
                            await save_json(res_list, symbol, 'json/earnings/surprise')
                    except Exception as e:
                        print(e)
                else:
                    check_existing_file(ticker, "surprise")
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
    #stock_symbols = ['TSLA']

    asyncio.run(run(stock_symbols, con))
    
except Exception as e:
    print(e)
finally:
    con.close()