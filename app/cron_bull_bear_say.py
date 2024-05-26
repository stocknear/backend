import aiohttp
import aiofiles
import ujson
import sqlite3
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime


load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

async def get_endpoint(session, symbol):
    url = "https://api.benzinga.com/api/v1/bulls_bears_say"
    querystring = {"token": api_key, "symbols": symbol}
    
    try:
        async with session.get(url, params=querystring) as response:
            res = ujson.loads(await response.text())
            try:
                for item in res['bulls_say_bears_say']:
                    date = datetime.fromtimestamp(item['updated'])
                    date = date.strftime("%B %d, %Y")
                    formatted_data = {'date': date, 'bearSays': item['bear_case'], 'bullSays': item['bull_case']}
            except:
                formatted_data = {}
    except Exception as e:
        formatted_data = {}
        print(e)
    with open(f"json/bull_bear_say/{symbol}.json", 'w') as file:
        ujson.dump(formatted_data, file)

async def run():
    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol != ?", ('%5EGSPC',))
    stocks_symbols = [row[0] for row in cursor.fetchall()]
    #stocks_symbols = ['NVDA']
    con.close()

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(get_endpoint(session, symbol) for symbol in stocks_symbols))

try:
    asyncio.run(run())
except Exception as e:
    print(e)
