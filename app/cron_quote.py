import ujson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
import pytz

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')


async def get_quote_of_stocks(ticker_list):
    ticker_str = ','.join(ticker_list)
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v3/quote/{ticker_str}?apikey={api_key}" 
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {}

async def get_pre_post_quote_of_stocks(ticker_list):
    ticker_str = ','.join(ticker_list)
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v4/batch-pre-post-market/{ticker_str}?apikey={api_key}" 
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {}

async def save_quote_as_json(symbol, data):
    with open(f"json/quote/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

async def save_pre_post_quote_as_json(symbol, data):
    try:
        with open(f"json/quote/{symbol}.json", 'r') as file:
            previous_close = (ujson.load(file))['price']
            changes_percentage = round((data['ask']/previous_close-1)*100,2)
        with open(f"json/pre-post-quote/{symbol}.json", 'w') as file:
            res = {'symbol': symbol, 'price': round(data['ask'],2), 'changesPercentage': changes_percentage, 'time': data['timestamp']}
            ujson.dump(res, file)
    except:
        pass

async def run():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')
    crypto_con = sqlite3.connect('crypto.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol != ?", ('%5EGSPC',))
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    crypto_cursor = crypto_con.cursor()
    crypto_cursor.execute("PRAGMA journal_mode = wal")
    crypto_cursor.execute("SELECT DISTINCT symbol FROM cryptos")
    crypto_symbols = [row[0] for row in crypto_cursor.fetchall()]

    con.close()
    etf_con.close()
    crypto_con.close()

    new_york_tz = pytz.timezone('America/New_York')
    current_time_new_york = datetime.now(new_york_tz)
    is_market_closed = (current_time_new_york.hour < 9 or
                  (current_time_new_york.hour == 9 and current_time_new_york.minute < 30) or
                  current_time_new_york.hour >= 16)


    #Crypto Quotes
    latest_quote = await get_quote_of_stocks(crypto_symbols)
    for item in latest_quote:
        symbol = item['symbol']

        await save_quote_as_json(symbol, item)

    # Stock and ETF Quotes
    
    total_symbols = stocks_symbols+etf_symbols
    
    chunk_size = len(total_symbols) // 10  # Divide the list into 10 chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
    for chunk in chunks:
        if is_market_closed == False:
            latest_quote = await get_quote_of_stocks(chunk)
            for item in latest_quote:
                symbol = item['symbol']
                await save_quote_as_json(symbol, item)
                #print(f"Saved data for {symbol}.")

        if is_market_closed == True:
            latest_quote = await get_pre_post_quote_of_stocks(chunk)
            for item in latest_quote:
                symbol = item['symbol']
                await save_pre_post_quote_as_json(symbol, item)
                #print(f"Saved data for {symbol}.")


try:
    asyncio.run(run())
except Exception as e:
    print(e)