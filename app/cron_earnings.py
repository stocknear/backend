import aiohttp
import aiofiles
import ujson
import sqlite3
import asyncio
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
N_days_ago = today - timedelta(days=10)


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

async def get_data(session, ticker):
    querystring = {"token": api_key, "parameters[tickers]": ticker}
    try:
        async with session.get(url, params=querystring, headers=headers) as response:
            data = ujson.loads(await response.text())['earnings']
            # Filter for future earnings
            future_dates = [item for item in data if ny_tz.localize(datetime.strptime(item["date"], "%Y-%m-%d")) >= today]
            if future_dates:
                nearest_future = min(future_dates, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
                try:
                    symbol = nearest_future['ticker']
                    time = nearest_future['time']
                    date = nearest_future['date']
                    eps_prior = float(nearest_future['eps_prior']) if nearest_future['eps_prior'] else 0
                    eps_est = float(nearest_future['eps_est']) if nearest_future['eps_est'] else 0
                    revenue_est = float(nearest_future['revenue_est']) if nearest_future['revenue_est'] else 0
                    revenue_prior = float(nearest_future['revenue_prior']) if nearest_future['revenue_prior'] else 0
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
                        eps_prior = float(nearest_recent['eps_prior']) if nearest_recent['eps_prior'] != '' else 0
                        eps_surprise = float(nearest_recent['eps_surprise']) if nearest_recent['eps_surprise'] != '' else 0
                        eps = float(nearest_recent['eps']) if nearest_recent['eps'] != '' else 0
                        revenue_prior = float(nearest_recent['revenue_prior']) if nearest_recent['revenue_prior'] != '' else 0
                        revenue_surprise = float(nearest_recent['revenue_surprise']) if nearest_recent['revenue_surprise'] != '' else 0
                        revenue = float(nearest_recent['revenue']) if nearest_recent['revenue'] != '' else 0
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

async def run(stock_symbols):
    async with aiohttp.ClientSession() as session:
        tasks = [get_data(session, symbol) for symbol in stock_symbols]
        for f in tqdm(asyncio.as_completed(tasks), total=len(stock_symbols)):
            await f

try:
    
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    asyncio.run(run(stock_symbols))

except Exception as e:
    print(e)