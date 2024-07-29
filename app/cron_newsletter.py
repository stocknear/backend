import aiohttp
import aiofiles
import ujson
import sqlite3
import pandas as pd
import asyncio
import pytz
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import sqlite3

headers = {"accept": "application/json"}




load_dotenv()
benzinga_api_key = os.getenv('BENZINGA_API_KEY_EXTRA')



query_template = """
    SELECT 
        marketCap
    FROM 
        stocks 
    WHERE
        symbol = ?
"""

async def save_json(data):
    with open(f"json/newsletter/data.json", 'w') as file:
        ujson.dump(data, file)


async def get_upcoming_earnings(session):
	url = "https://api.benzinga.com/api/v2.1/calendar/earnings"
	querystring = {"token": benzinga_api_key,"parameters[date_from]":"2024-07-30","parameters[date_to]":"2024-07-30"}
	try:
		async with session.get(url, params=querystring, headers=headers) as response:
			res_list = []
			res = ujson.loads(await response.text())['earnings']
			for item in res:
				try:
					symbol = item['ticker']
					name = item['name']
					time = item['time']
					eps_prior = float(item['eps_prior']) if item['eps_prior'] != '' else 0
					eps_est = float(item['eps_est']) if item['eps_est'] != '' else 0
					revenue_est = float(item['revenue_est']) if item['revenue_est'] != '' else 0
					revenue_prior = float(item['revenue_prior']) if item['revenue_prior'] != '' else 0
					if symbol in stock_symbols and revenue_est != 0 and revenue_prior != 0 and eps_prior != 0 and eps_est != 0:
						df = pd.read_sql_query(query_template, con, params=(symbol,))
						market_cap = float(df['marketCap'].iloc[0]) if df['marketCap'].iloc[0] != '' else 0
						res_list.append({
							'symbol': symbol,
							'name': name,
							'time': time,
							'marketCap': market_cap,
							'epsPrior':eps_prior,
							'epsEst': eps_est,
							'revenuePrior': revenue_prior,
							'revenueEst': revenue_est
							})
				except Exception as e:
					print(e)
					pass
		res_list.sort(key=lambda x: x['marketCap'], reverse=True)
		return res_list
	except Exception as e:
		print(e)


async def run():
	async with aiohttp.ClientSession() as session:
		upcoming_earnings = await get_upcoming_earnings(session)		

		data = {'upcomingEarnings': upcoming_earnings}
		
		if len(data) > 0:
			await save_json(data)
		

try:
	con = sqlite3.connect('stocks.db')
	cursor = con.cursor()
	cursor.execute("PRAGMA journal_mode = wal")
	cursor.execute("SELECT DISTINCT symbol FROM stocks")
	stock_symbols = [row[0] for row in cursor.fetchall()]

	asyncio.run(run())
	con.close()
except Exception as e:
    print(e)