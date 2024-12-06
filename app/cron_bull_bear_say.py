import aiohttp
import ujson
import sqlite3
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime
import re

load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')



async def get_endpoint(session, symbol):
    url = "https://api.benzinga.com/api/v1/bulls_bears_say"
    querystring = {"token": api_key, "symbols": symbol}
    formatted_data = {}
    try:
        async with session.get(url, params=querystring) as response:
            res = ujson.loads(await response.text())
            try:
                for item in res['bulls_say_bears_say']:
                    date = datetime.fromtimestamp(item['updated'])
                    date = date.strftime("%Y-%m-%d %H:%M:%S")
                    bull_case = item['bull_case']
                    bear_case = item['bear_case']
                    formatted_data = {'date': date, 'bullSays': bull_case, 'bearSays': bear_case}
            except:
                pass
    except:
        pass

    if formatted_data:
        with open(f"json/bull_bear_say/{symbol}.json", 'w') as file:
            ujson.dump(formatted_data, file)

async def run():
    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stocks_symbols = [row[0] for row in cursor.fetchall()]
    #stocks_symbols = ['NVDA']
    con.close()

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(get_endpoint(session, symbol) for symbol in stocks_symbols))

try:
    asyncio.run(run())
except Exception as e:
    print(e)
