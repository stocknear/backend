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

headers = {"accept": "application/json"}



load_dotenv()
benzinga_api_key = os.getenv('BENZINGA_API_KEY')


async def save_json(data):
    with open(f"json/dashboard/data.json", 'w') as file:
        ujson.dump(data, file)


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
		try:
			with open(f"json/congress-trading/rss-feed/data.json", 'r') as file:
				congress_flow = ujson.load(file)[0:4]
		except:
			congress_flow = []
		try:
			with open(f"json/options-flow/feed/data.json", 'r') as file:
				options_flow = ujson.load(file)
				options_flow = sorted(options_flow, key=lambda x: x['cost_basis'], reverse=True)
				options_flow = [{key: item[key] for key in ['cost_basis', 'ticker','assetType', 'date_expiration', 'put_call', 'sentiment', 'strike_price']} for item in options_flow[0:4]]
		except:
			options_flow = []

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

		try:
			with open(f"json/most-shorted-stocks/data.json", 'r') as file:
				data = ujson.load(file)[0]
				shorted_stock = {key: data[key] for key in ['symbol', 'shortOutStandingPercent']}
				
		except:
			shorted_stock = {}


		quick_info = {**market_mover, 'shorted': shorted_stock}

		data = {'quickInfo': quick_info, 'optionsFlow': options_flow, 'congressFlow': congress_flow, 'wiimFeed': wiim_feed, 'marketNews': benzinga_news}
		
		if len(data) > 0:
			await save_json(data)

try:
    asyncio.run(run())
except Exception as e:
    print(e)