import orjson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
import pytz
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')

ny_timezone = pytz.timezone("America/New_York")

today = datetime.now().strftime("%Y-%m-%d")
today_str = datetime.now().strftime("%b %d, %Y")  # e.g. "Jul 10, 2025"

# Function to delete all files in a directory
def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path, 'r') as file:
                data = orjson.loads(file.read())
            file_time = data.get("time", "")
            # extract date part of "time"
            file_date = ", ".join(file_time.split(", ")[:2])  # "Jul 10, 2025"

            if file_date != today_str:
                os.remove(file_path)

        except:
            pass


async def get_quote_of_stocks(session, ticker_list):
    ticker_str = ','.join(ticker_list)
    
    url = f"https://financialmodelingprep.com/api/v3/quote/{ticker_str}?apikey={api_key}" 
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        else:
            return {}


async def get_pre_post_quote_of_stocks(ticker_list):
    ticker_str = ','.join(ticker_list)
    today_dt = datetime.now(ny_timezone).date()
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/stable/batch-aftermarket-trade?symbols={ticker_str}&apikey={api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                res = []
                for item in data:
                    try:
                        item_date = datetime.fromtimestamp(item['timestamp'] / 1000, ny_timezone).date()
                        if (today_dt - item_date).days <= 5:
                            item['date'] = item_date.strftime('%Y-%m-%d')
                            res.append(item)
                    except Exception as e:
                        print(f"Error parsing item: {e}")
                return res
            else:
                return []


async def get_bid_ask_quote_of_stocks(ticker_list):
    ticker_str = ','.join(ticker_list)
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/stable/batch-aftermarket-quote?symbols={ticker_str}&apikey={api_key}" 
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return []

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
        if exchange in ['NASDAQ','AMEX','NYSE','OTC']:
            with open(f"json/pre-post-quote/{symbol}.json", 'w') as file:
                dt = datetime.fromtimestamp(data['timestamp']/1000, ny_timezone)
                formatted_date = dt.strftime("%b %d, %Y, %I:%M %p %Z")

                #Bugfixing: pre-filter to fight against FMP bug that shows -30% drop sometimes in the premarket
                if abs(changes_percentage) <= 15:
                    res = {'symbol': symbol, 'price': round(data['price'],2), 'changesPercentage': changes_percentage, 'time': formatted_date}
                    file.write(orjson.dumps(res).decode())
    except Exception as e:
        pass

async def save_bid_ask_as_json(symbol, data):
    try:
        # Read previous close price and load existing quote data
        with open(f"json/quote/{symbol}.json", 'r') as file:
            quote_data = orjson.loads(file.read())

        ask_price = round(data['askPrice'], 2)
        bid_price = round(data['bidPrice'], 2)
        
        #Bugfixing: pre-filter to fight against FMP bug that shows -30% drop sometimes in the premarket
        
        mean_price = (ask_price+bid_price)/2
        current_price = quote_data['price']

        changes_percentage = (mean_price/current_price -1 ) *100

        if abs(changes_percentage) <= 15:
            quote_data.update({
                'ask': ask_price,  # Add ask price
                'bid': bid_price,   # Add bid price
            })

        # Save the updated quote data back to the same JSON file
        with open(f"json/quote/{symbol}.json", 'w') as file:
            file.write(orjson.dumps(quote_data).decode())
    except Exception as e:
        print(f"An error occurred: {e}")  # Print the error for debugging


async def run():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')
    index_con = sqlite3.connect('index.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    index_cursor = index_con.cursor()
    index_cursor.execute("PRAGMA journal_mode = wal")
    index_cursor.execute("SELECT DISTINCT symbol FROM indices")
    index_symbols = [row[0] for row in index_cursor.fetchall()]

    con.close()
    etf_con.close()
    index_con.close()

    new_york_tz = pytz.timezone('America/New_York')
    current_time_new_york = datetime.now(new_york_tz)
    is_market_closed = (current_time_new_york.hour < 9 or
                  (current_time_new_york.hour == 9 and current_time_new_york.minute < 30) or
                  current_time_new_york.hour >= 16)
    
    # Check if we're within 5 minutes after market close (4:00 PM - 4:05 PM ET)
    is_within_5min_after_close = (current_time_new_york.hour == 16 and current_time_new_york.minute <= 5)

    
    total_symbols = stocks_symbols+etf_symbols+index_symbols
    chunk_size = len(total_symbols) // 20  # Divide the list into N chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
    delete_files_in_directory("json/pre-post-quote")
    for chunk in tqdm(chunks):
        # Run get_quote_of_stocks during market hours OR within 15 minutes after close
        if is_market_closed == False or is_within_5min_after_close:
            print("Market Quote running...")
            async with aiohttp.ClientSession() as session:
                latest_quote = await get_quote_of_stocks(session, chunk)
                save_tasks = []
                for item in latest_quote:
                    try:
                        symbol = item['symbol']
                        await save_quote_as_json(symbol, item)
                        save_tasks.append(save_quote_as_json(symbol, item))
                    except:
                        pass
                await asyncio.gather(*save_tasks)

            #Always true
            bid_ask_quote = await get_bid_ask_quote_of_stocks(chunk)
            for item in bid_ask_quote:
                try:
                    symbol = item['symbol']
                    await save_bid_ask_as_json(symbol, item)
                except:
                    pass

        elif is_market_closed == True:
            print("Pre-Post Quote running...")
            latest_quote = await get_pre_post_quote_of_stocks(chunk)
            for item in latest_quote:
                symbol = item['symbol']
                await save_pre_post_quote_as_json(symbol, item)
        

try:
    asyncio.run(run())
except Exception as e:
    print(e)