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

date_format = "%a, %d %b %Y %H:%M:%S %z"

load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

headers = {"accept": "application/json"}

query_template = """
    SELECT
        close
    FROM
        "{symbol}"
    WHERE
        date BETWEEN ? AND ?
"""

# List of holidays when the stock market is closed
holidays = [
    "2024-01-01",
    "2024-03-29",
    "2024-12-25",
]

def is_holiday(date):
    """Check if the given date is a holiday"""
    str_date = date.strftime("%Y-%m-%d")
    return str_date in holidays

def correct_weekday(selected_date):
    # Monday is 0 and Sunday is 6
    if selected_date.weekday() == 0:
        selected_date -= timedelta(3)
    elif selected_date.weekday() <= 4:
        selected_date -= timedelta(1)
    elif selected_date.weekday() == 5:
        selected_date -= timedelta(1)
    elif selected_date.weekday() == 6:
        selected_date -= timedelta(2)
    
    # Check if the selected date is a holiday and adjust if necessary
    while is_holiday(selected_date):
        selected_date -= timedelta(1)
    
    # Adjust again if the resulting date is a Saturday or Sunday
    if selected_date.weekday() >= 5:
        selected_date -= timedelta(selected_date.weekday() - 4)
    
    return selected_date

async def get_endpoint(session, symbol, con):
    url = "https://api.benzinga.com/api/v2/news"
    querystring = {"token": api_key,"tickers": symbol, "channels":"WIIM","pageSize":"20","displayOutput":"full"}
    async with session.get(url, params=querystring, headers=headers) as response:
        res_list = []
        res = ujson.loads(await response.text())

        for item in res:
            date_obj = datetime.strptime(item['created'], date_format)
            date_obj_utc = date_obj.astimezone(pytz.utc)
            
            new_date_obj_utc = date_obj_utc
        
            start_date_obj_utc = correct_weekday(date_obj_utc)

            start_date = start_date_obj_utc.strftime("%Y-%m-%d")
            end_date = new_date_obj_utc.strftime("%Y-%m-%d")

            new_date_str = new_date_obj_utc.strftime("%b %d, %Y")
            query = query_template.format(symbol=symbol)
    
            try:
                df = pd.read_sql_query(query,con, params=(start_date, end_date))
                if not df.empty:
                    change_percent = round((df['close'].iloc[1]/df['close'].iloc[0] -1)*100,2)
                else:
                    change_percent = '-'
            except Exception as e:
                change_percent = '-'

            res_list.append({'date': new_date_str, 'text': item['title'], 'changesPercentage': change_percent})
        with open(f"json/wiim/company/{symbol}.json", 'w') as file:
                ujson.dump(res_list, file)

        '''
        current_date = datetime.now(pytz.utc)
        date_difference = current_date - new_date_obj_utc
        if date_difference.days < 2:
            new_date_str = new_date_obj_utc.strftime("%b %d, %Y")
            formatted_data = {'wiim': res[0]['title'], 'updated': new_date_str}

            with open(f"json/wiim/{symbol}.json", 'w') as file:
                ujson.dump(formatted_data, file)
        '''


async def get_latest_wiim(session, stock_symbols, etf_symbols):
    url = "https://api.benzinga.com/api/v2/news"
    querystring = {"token": api_key,"channels":"WIIM","pageSize":"20","displayOutput":"full"}
    
    try:
        async with session.get(url, params=querystring, headers=headers) as response:
            res_list = []
            res = ujson.loads(await response.text())
            for item in res:
                for el in item['stocks']:
                    # Update the 'name' key to 'ticker'
                    if 'name' in el:
                        el['ticker'] = el.pop('name')
                        if el['ticker'] in stock_symbols:
                            el['assetType'] = 'stock'
                        elif el['ticker'] in etf_symbols:
                            el['assetType'] = 'etf'
                res_list.append({'date': item['created'], 'text': item['title'], 'stocks': item['stocks']})
            with open(f"json/wiim/rss-feed/data.json", 'w') as file:
                    ujson.dump(res_list, file)

    except Exception as e:
        #pass
        print(e)

async def run():
    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    etf_con = sqlite3.connect('etf.db')

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    
    async with aiohttp.ClientSession() as session:
        await get_latest_wiim(session, stock_symbols, etf_symbols)
        await asyncio.gather(*(get_endpoint(session, symbol, con) for symbol in stock_symbols))
        await asyncio.gather(*(get_endpoint(session, symbol, etf_con) for symbol in etf_symbols))

    con.close()
    etf_con.close()
try:
    asyncio.run(run())
except Exception as e:
    print(e)
