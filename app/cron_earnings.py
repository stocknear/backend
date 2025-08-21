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


def filter_keep_nearest_future_only(data):
    if not data:
        return []

    ref_date = datetime.now().date()

    past_present = []
    future_records = []

    for record in data:
        try:
            record_date = datetime.strptime(record['date'], '%Y-%m-%d').date()
        except (ValueError, KeyError):
            continue  # Skip records with invalid or missing date

        if record_date <= ref_date:
            past_present.append(record)
        else:
            future_records.append(record)

    nearest_future = None
    if future_records:
        nearest_future = min(
            future_records,
            key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d').date()
        )

    result = past_present.copy()
    if nearest_future:
        result.append(nearest_future)

    # Optional: sort by date descending (latest first)
    result.sort(key=lambda x: x['date'], reverse=True)

    return result



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
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, f"{symbol}.json")
    async with aiofiles.open(file_path, 'w') as file:
        await file.write(ujson.dumps(data))

async def get_data(session, ticker, con):
    if ticker == "BRK-A":
        api_ticker = "BRK/A"
    elif ticker == "BRK-B":
        api_ticker = "BRK/B"
    else:
        api_ticker = ticker

    querystring = {"token": api_key, "parameters[tickers]": api_ticker}
    try:
        async with session.get(url, params=querystring, headers=headers) as response:
            data = ujson.loads(await response.text())['earnings']
            
            raw_data = [{k: v for k, v in item.items() if k not in ['currency','exchange','eps_type','revenue_type', 'name', 'notes','updated', 'ticker', 'importance', 'id','date_confirmed']} for item in data]
            raw_data = filter_keep_nearest_future_only(raw_data)
            #save all rawdata for llm

            await save_json(raw_data, ticker, 'json/earnings/raw')

            # Filter for future earnings
            future_dates = [item for item in raw_data if ny_tz.localize(datetime.strptime(item["date"], "%Y-%m-%d")).date() >= today.date()]

            if future_dates:
                nearest_future = min(future_dates, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
                try:
                    symbol = ticker
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

                # Filter for past earnings within the last N days
                recent_dates = [item for item in data if N_days_ago <= ny_tz.localize(datetime.strptime(item["date"], "%Y-%m-%d")) <= today]
                if recent_dates:
                    nearest_recent = min(recent_dates, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
                    try:
                        date = nearest_recent['date']
                        eps_est = float(nearest_future['eps_est']) if nearest_future['eps_est'] else None
                        revenue_est = float(nearest_future['revenue_est']) if nearest_future['revenue_est'] else None

                        eps_prior = float(nearest_recent['eps_prior']) if nearest_recent['eps_prior'] != '' else None
                        eps = float(nearest_recent['eps']) if nearest_recent['eps'] != '' else None
                        if nearest_recent.get('eps_surprise') not in ('', None):
                            eps_surprise = float(nearest_recent['eps_surprise'])
                        else:
                            eps_surprise = round(eps - eps_est, 2) if eps is not None and eps_est not in (None, 0) else None


                        revenue_prior = float(nearest_recent['revenue_prior']) if nearest_recent['revenue_prior'] != '' else None
                        revenue = float(nearest_recent['revenue']) if nearest_recent['revenue'] != '' else None
                        if nearest_recent.get('revenue_surprise') not in ('', None):
                            revenue_surprise = float(nearest_recent['revenue_surprise'])
                        else:
                            revenue_surprise = round(revenue - revenue_est, 0) if revenue is not None and revenue_est not in (None, 0) else None

                        if all(v is not None for v in [revenue, revenue_prior, eps_prior, eps, revenue_surprise, eps_surprise]):
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
    #stock_symbols = ['TGT']

    asyncio.run(run(stock_symbols, con))
    
except Exception as e:
    print(e)
finally:
    con.close()