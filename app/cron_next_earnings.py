import aiohttp
import aiofiles
import ujson
import sqlite3
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

headers = {"accept": "application/json"}
url = "https://api.benzinga.com/api/v2.1/calendar/earnings"
load_dotenv()
benzinga_api_key_extra = os.getenv('BENZINGA_API_KEY_EXTRA')

today = datetime.today()

async def save_json(data, symbol):
    async with aiofiles.open(f"json/next-earnings/companies/{symbol}.json", 'w') as file:
        await file.write(ujson.dumps(data))

async def get_data(session, ticker):
    querystring = {"token": benzinga_api_key_extra, "parameters[tickers]": ticker}
    try:
        async with session.get(url, params=querystring, headers=headers) as response:
            if response.status == 200:
                data = ujson.loads(await response.text())['earnings']
                future_dates = [item for item in data if datetime.strptime(item["date"], "%Y-%m-%d") > today]
                if future_dates:
                    data = min(future_dates, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
                    try:
                        symbol = data['ticker']
                        time = data['time']
                        date = data['date']
                        eps_prior = float(data['eps_prior']) if data['eps_prior'] else 0
                        eps_est = float(data['eps_est']) if data['eps_est'] else 0
                        revenue_est = float(data['revenue_est']) if data['revenue_est'] else 0
                        revenue_prior = float(data['revenue_prior']) if data['revenue_prior'] else 0
                        if revenue_est and revenue_prior and eps_prior and eps_est:
                            res_list = {
                            	'date': date,
                                'time': time,
                                'epsPrior': eps_prior,
                                'epsEst': eps_est,
                                'revenuePrior': revenue_prior,
                                'revenueEst': revenue_est
                            }
                            await save_json(res_list, symbol)
                    except KeyError:
                        pass
    except Exception as e:
        pass

async def run(stock_symbols):
    async with aiohttp.ClientSession() as session:
        tasks = [get_data(session, symbol) for symbol in stock_symbols]
        for f in tqdm(asyncio.as_completed(tasks), total=len(stock_symbols)):
            await f

try:
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    asyncio.run(run(stock_symbols))

except Exception as e:
    print(e)
