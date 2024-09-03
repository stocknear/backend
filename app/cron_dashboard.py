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
from datetime import datetime, timedelta, date
import sqlite3


headers = {"accept": "application/json"}


load_dotenv()
benzinga_api_key = os.getenv('BENZINGA_API_KEY')
benzinga_api_key_extra = os.getenv('BENZINGA_API_KEY_EXTRA')
fmp_api_key = os.getenv('FMP_API_KEY')


query_template = """
    SELECT 
        marketCap
    FROM 
        stocks 
    WHERE
        symbol = ?
"""


async def save_json(data):
    with open(f"json/dashboard/data.json", 'w') as file:
        ujson.dump(data, file)


def get_sector_path(sector):
    sector_paths = {
        'Financials': "/list/financial-sector",
        'Healthcare': "/list/healthcare-sector",
        'Information Technology': "/list/technology-sector",
        'Technology': "/list/technology-sector",
        'Financial Services': "/list/financial-sector",
        'Industrials': "/list/industrials-sector",
        'Energy': "/list/energy-sector",
        'Utilities': "/list/utilities-sector",
        'Consumer Cyclical': "/list/consumer-cyclical-sector",
        'Real Estate': "/list/real-estate-sector",
        'Basic Materials': "/list/basic-materials-sector",
        'Communication Services': "/list/communication-services-sector",
        'Consumer Defensive': "/list/consumer-defensive-sector"
    }

    # Return the path if the sector exists in the dictionary, otherwise return None or a default path
    return sector_paths.get(sector, None)

def parse_time(time_str):
    try:
        # Try parsing as full datetime
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            # Try parsing as time only
            time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
            # Combine with today's date
            return datetime.combine(date.today(), time_obj)
        except ValueError:
            # If all else fails, return a default datetime
            return datetime.min

def remove_duplicates(elements):
    seen = set()
    unique_elements = []
    
    for element in elements:
        if element['symbol'] not in seen:
            seen.add(element['symbol'])
            unique_elements.append(element)
    
    return unique_elements

def weekday():
    today = datetime.today()
    one_day = timedelta(1)
    yesterday = today - one_day
    
    while yesterday.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        yesterday -= one_day
    
    return yesterday.strftime('%Y-%m-%d')


today = datetime.today().strftime('%Y-%m-%d')
tomorrow = (datetime.today() + timedelta(1))
yesterday = weekday()

if tomorrow.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
    tomorrow = tomorrow + timedelta(days=(7 - tomorrow.weekday()))

tomorrow = tomorrow.strftime('%Y-%m-%d')

async def get_upcoming_earnings(session):
	url = "https://api.benzinga.com/api/v2.1/calendar/earnings"

	importance_list = ["1","2","3","4","5"]
	res_list = []
	for importance in importance_list:

		querystring = {"token": benzinga_api_key_extra,"parameters[importance]":importance,"parameters[date_from]":tomorrow,"parameters[date_to]":tomorrow,"parameters[date_sort]":"date"}
		try:
			async with session.get(url, params=querystring, headers=headers) as response:
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
						print('Upcoming Earnings:', e)
						pass
			res_list = remove_duplicates(res_list)
			res_list.sort(key=lambda x: x['marketCap'], reverse=True)
			res_list = [{k: v for k, v in d.items() if k != 'marketCap'} for d in res_list]
		except Exception as e:
			print(e)
			pass

	return res_list[0:5]


async def get_recent_earnings(session):
	url = "https://api.benzinga.com/api/v2.1/calendar/earnings"
	importance_list = ["3","4","5"]
	res_list = []
	for importance in importance_list:
		querystring = {"token": benzinga_api_key_extra,"parameters[importance]":importance,"parameters[date_from]":yesterday,"parameters[date_to]":today,"parameters[date_sort]":"date"}
		try:
			async with session.get(url, params=querystring, headers=headers) as response:
				res = ujson.loads(await response.text())['earnings']
				for item in res:
					try:
						symbol = item['ticker']
						name = item['name']
						time = item['time']
						eps_prior = float(item['eps_prior']) if item['eps_prior'] != '' else 0
						eps_surprise = float(item['eps_surprise']) if item['eps_surprise'] != '' else 0
						eps = float(item['eps']) if item['eps'] != '' else 0
						revenue_prior = float(item['revenue_prior']) if item['revenue_prior'] != '' else 0
						revenue_surprise = float(item['revenue_surprise']) if item['revenue_surprise'] != '' else 0
						revenue = float(item['revenue']) if item['revenue'] != '' else 0
						if symbol in stock_symbols and revenue != 0 and revenue_prior != 0 and eps_prior != 0 and eps != 0 and revenue_surprise != 0 and eps_surprise != 0:
							df = pd.read_sql_query(query_template, con, params=(symbol,))
							market_cap = float(df['marketCap'].iloc[0]) if df['marketCap'].iloc[0] != '' else 0
							res_list.append({
								'symbol': symbol,
								'name': name,
								'time': time,
								'marketCap': market_cap,
								'epsPrior':eps_prior,
								'epsSurprise': eps_surprise,
								'eps': eps,
								'revenuePrior': revenue_prior,
								'revenueSurprise': revenue_surprise,
								'revenue': revenue
								})
					except Exception as e:
						print('Recent Earnings:', e)
						pass
		except Exception as e:
			pass
	res_list = remove_duplicates(res_list)
	#res_list.sort(key=lambda x: x['marketCap'], reverse=True)
	res_list.sort(key=lambda x: (-parse_time(x['time']).timestamp(), -x['marketCap']))
	res_list = [{k: v for k, v in d.items() if k != 'marketCap'} for d in res_list]
	return res_list[0:5]

async def get_recent_dividends(session):
	url = "https://api.benzinga.com/api/v2.1/calendar/dividends"
	importance_list = ["2","3","4","5"]
	res_list = []
	for importance in importance_list:
		querystring = {"token": benzinga_api_key_extra,"parameters[importance]":importance,"parameters[date_from]":yesterday,"parameters[date_to]":today}
		try:
			async with session.get(url, params=querystring, headers=headers) as response:
				res = ujson.loads(await response.text())['dividends']
				for item in res:
					try:
						symbol = item['ticker']
						name = item['name']
						dividend = float(item['dividend']) if item['dividend'] != '' else 0
						dividend_prior = float(item['dividend_prior']) if item['dividend_prior'] != '' else 0
						dividend_yield = round(float(item['dividend_yield'])*100,2) if item['dividend_yield'] != '' else 0
						ex_dividend_date = item['ex_dividend_date'] if item['ex_dividend_date'] != '' else 0
						payable_date = item['payable_date'] if item['payable_date'] != '' else 0
						record_date = item['record_date'] if item['record_date'] != '' else 0
						if symbol in stock_symbols and dividend != 0 and payable_date != 0 and dividend_prior != 0 and ex_dividend_date != 0 and record_date != 0 and dividend_yield != 0:
							df = pd.read_sql_query(query_template, con, params=(symbol,))
							market_cap = float(df['marketCap'].iloc[0]) if df['marketCap'].iloc[0] != '' else 0
							res_list.append({
								'symbol': symbol,
								'name': name,
								'dividend': dividend,
								'marketCap': market_cap,
								'dividendPrior':dividend_prior,
								'dividendYield': dividend_yield,
								'exDividendDate': ex_dividend_date,
								'payableDate': payable_date,
								'recordDate': record_date,
								'updated': item['updated']
								})
					except Exception as e:
						print('Recent Dividends:', e)
						pass
		except Exception as e:
			print(e)
			pass

	res_list = remove_duplicates(res_list)
	res_list.sort(key=lambda x: x['marketCap'], reverse=True)
	res_list = [{k: v for k, v in d.items() if k != 'marketCap'} for d in res_list]
	return res_list[0:5]

async def get_top_sector(session):
    url = f"https://financialmodelingprep.com/api/v3/sectors-performance?apikey={fmp_api_key}"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                sectors = await response.json()
                sectors = [{'sector': item['sector'], 'changesPercentage': round(float(item['changesPercentage'].strip('%')), 2)} for item in sectors]
                res = max(sectors, key=lambda x: x['changesPercentage'])
                res['link'] = get_sector_path(res['sector'])

                return res
            else:
                print(f"Failed to retrieve data: {response.status}")
                return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

async def get_latest_bezinga_market_news(session):
    url = "https://api.benzinga.com/api/v2/news"
    querystring = {"token": benzinga_api_key,"channels":"News","pageSize":"10","displayOutput":"full"}
    try:
        async with session.get(url, params=querystring, headers=headers) as response:
            res_list = []
            res = ujson.loads(await response.text())
            for item in res:
                res_list.append({'date': item['created'], 'text': item['title'], 'url': item['url']})

        res_list.sort(key=lambda x: datetime.strptime(x['date'], '%a, %d %b %Y %H:%M:%S %z'), reverse=True)
        return res_list
    except Exception as e:
        #pass
        print(e)


async def run():
	async with aiohttp.ClientSession() as session:
		benzinga_news = await get_latest_bezinga_market_news(session)
		recent_earnings = await get_recent_earnings(session)
		upcoming_earnings = await get_upcoming_earnings(session)
		top_sector = await get_top_sector(session)
		recent_dividends = await get_recent_dividends(session)

		try:
			with open(f"json/retail-volume/data.json", 'r') as file:
				retail_tracker = ujson.load(file)[0:5]
		except:
			retail_tracker = []
		try:
			with open(f"json/options-flow/feed/data.json", 'r') as file:
				options_flow = ujson.load(file)
				
				# Filter the options_flow to include only items with ticker in total_symbol
				options_flow = [item for item in options_flow if item['ticker'] in stock_symbols]
				
				highest_volume = sorted(options_flow, key=lambda x: int(x['volume']), reverse=True)
				highest_volume = [{key: item[key] for key in ['cost_basis', 'ticker','assetType', 'date_expiration', 'put_call', 'volume', 'strike_price']} for item in highest_volume[0:4]]

				highest_premium = sorted(options_flow, key=lambda x: int(x['cost_basis']), reverse=True)
				highest_premium = [{key: item[key] for key in ['cost_basis', 'ticker','assetType', 'date_expiration', 'put_call', 'volume', 'strike_price']} for item in highest_premium[0:4]]

				highest_open_interest = sorted(options_flow, key=lambda x: int(x['open_interest']), reverse=True)
				highest_open_interest = [{key: item[key] for key in ['cost_basis', 'ticker','assetType', 'date_expiration', 'put_call', 'open_interest', 'strike_price']} for item in highest_open_interest[0:4]]

				options_flow = {'premium': highest_premium, 'volume': highest_volume, 'openInterest':highest_open_interest}
		except Exception as e:
			print(e)
			options_flow = {}
		try:
			with open(f"json/wiim/rss-feed/data.json", 'r') as file:
				wiim_feed = ujson.load(file)[0:5]

		except:
			wiim_feed = []

		try:
			with open(f"json/market-movers/data.json", 'r') as file:
				data = ujson.load(file)
				market_mover = {'winner': data['gainers']['1D'][0], 'loser': data['losers']['1D'][0], 'active': data['active']['1D'][0]}
		except:
			market_mover = {}

		'''
		try:
			with open(f"json/most-shorted-stocks/data.json", 'r') as file:
				data = ujson.load(file)[0]
				shorted_stock = {key: data[key] for key in ['symbol', 'shortOutStandingPercent']}
				
		except:
			shorted_stock = {}
		'''

		quick_info = {**market_mover, 'topSector': top_sector}

		data = {
		    'quickInfo': quick_info,
		    'optionsFlow': options_flow,
		    #'retailTracker': retail_tracker,
		    'wiimFeed': wiim_feed,
		    'marketNews': benzinga_news,
		    'recentEarnings': recent_earnings,
		    'upcomingEarnings': upcoming_earnings,
		    'recentDividends': recent_dividends,
		}

		
		if len(data) > 0:
			await save_json(data)

try:

	con = sqlite3.connect('stocks.db')
	etf_con = sqlite3.connect('etf.db')

	cursor = con.cursor()
	cursor.execute("PRAGMA journal_mode = wal")
	cursor.execute("SELECT DISTINCT symbol FROM stocks")
	stock_symbols = [row[0] for row in cursor.fetchall()]

	etf_cursor = etf_con.cursor()
	etf_cursor.execute("PRAGMA journal_mode = wal")
	etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
	etf_symbols = [row[0] for row in etf_cursor.fetchall()]

	total_symbols = stock_symbols+etf_symbols
	asyncio.run(run())
	con.close()
	etf_con.close()

except Exception as e:
    print(e)