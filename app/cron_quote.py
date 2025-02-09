import orjson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
import pytz

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')

ny_timezone = pytz.timezone("America/New_York")


# Function to delete all files in a directory
def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


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
        #url = f"https://financialmodelingprep.com/api/v4/batch-pre-post-market/{ticker_str}?apikey={api_key}" 
        url = f"https://financialmodelingprep.com/api/v4/batch-pre-post-market-trade/{ticker_str}?apikey={api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {}

async def get_bid_ask_quote_of_stocks(ticker_list):
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
        file.write(orjson.dumps(data).decode())

async def save_pre_post_quote_as_json(symbol, data):
    try:
        with open(f"json/quote/{symbol}.json", 'r') as file:
            quote_data = orjson.loads(file.read())
            exchange = quote_data.get('exchange',None)
            previous_close = quote_data['price']
            changes_percentage = round((data['price']/previous_close-1)*100,2)
        if exchange in ['NASDAQ','AMEX','NYSE']:
            with open(f"json/pre-post-quote/{symbol}.json", 'w') as file:
                dt = datetime.fromtimestamp(data['timestamp']/1000, ny_timezone)
                formatted_date = dt.strftime("%b %d, %Y, %I:%M %p %Z")
                res = {'symbol': symbol, 'price': round(data['price'],2), 'changesPercentage': changes_percentage, 'time': formatted_date}
                file.write(orjson.dumps(res).decode())
    except Exception as e:
        pass

async def save_bid_ask_as_json(symbol, data):
    try:
        # Read previous close price and load existing quote data
        with open(f"json/quote/{symbol}.json", 'r') as file:
            quote_data = orjson.loads(file.read())

        # Update quote data with new price, ask, bid, changesPercentage, and timestamp
        quote_data.update({
            'ask': round(data['ask'], 2),  # Add ask price
            'bid': round(data['bid'], 2),   # Add bid price
        })

        # Save the updated quote data back to the same JSON file
        with open(f"json/quote/{symbol}.json", 'w') as file:
            file.write(orjson.dumps(quote_data).decode())
    except Exception as e:
        print(f"An error occurred: {e}")  # Print the error for debugging


async def run():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]


    index_symbols = ['^SPX','^VIX']

    con.close()
    etf_con.close()

    new_york_tz = pytz.timezone('America/New_York')
    current_time_new_york = datetime.now(new_york_tz)
    is_market_closed = (current_time_new_york.hour < 9 or
                  (current_time_new_york.hour == 9 and current_time_new_york.minute < 30) or
                  current_time_new_york.hour >= 16)


    #Crypto Quotes
    '''
    latest_quote = await get_quote_of_stocks(crypto_symbols)
    for item in latest_quote:
        symbol = item['symbol']

        await save_quote_as_json(symbol, item)
    '''
    # Stock and ETF Quotes
    
    total_symbols = stocks_symbols+etf_symbols+index_symbols
    
    chunk_size = len(total_symbols) // 20  # Divide the list into N chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
    delete_files_in_directory("json/pre-post-quote")
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
        #Always true
        bid_ask_quote = await get_bid_ask_quote_of_stocks(chunk)
        for item in bid_ask_quote:
            symbol = item['symbol']
            await save_bid_ask_as_json(symbol, item)

try:
    asyncio.run(run())
except Exception as e:
    print(e)