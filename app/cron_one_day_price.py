import ujson
import orjson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime, timedelta, time
import pandas as pd
from GetStartEndDate import GetStartEndDate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

market_caps = {}

async def save_price_data(symbol, data):
    with open(f"json/one-day-price/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


async def fetch_and_save_symbols_data(symbols, semaphore):
    tasks = []
    for symbol in symbols:
        task = asyncio.create_task(get_todays_data(symbol, semaphore))
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    
    for symbol, response in zip(symbols, responses):
        if len(response) > 0:
            await save_price_data(symbol, response)

async def get_todays_data(ticker, semaphore):
    # Assuming GetStartEndDate().run() returns today's start and end datetime objects
    start_date_1d, end_date_1d = GetStartEndDate().run()
    
    # Format today's date as string "YYYY-MM-DD"
    today_str = start_date_1d.strftime("%Y-%m-%d")
    
    current_weekday = end_date_1d.weekday()
    start_date = start_date_1d.strftime("%Y-%m-%d")
    end_date = end_date_1d.strftime("%Y-%m-%d")
    
    # Make sure your URL is correctly constructed (note: query parameter concatenation may need adjustment)
    url = f"https://financialmodelingprep.com/stable/historical-chart/1min?symbol={ticker}&from={start_date}&to={end_date}&apikey={api_key}"
    
    df_1d = pd.DataFrame()
    current_date = start_date_1d
    target_time = time(9, 30)
    
    # Use semaphore to limit concurrent connections
    async with semaphore:
        # Async HTTP request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    try:
                        json_data = await response.json()
                        # Create DataFrame and reverse order if needed
                        df_1d = pd.DataFrame(json_data).iloc[::-1].reset_index(drop=True)
                        
                        # Filter out rows not matching today's date.
                        df_1d = df_1d[df_1d['date'].str.startswith(today_str)]
                        
                        # If you want to rename "date" to "time", do that after filtering:
                        df_1d = df_1d.drop(['volume'], axis=1)
                        df_1d = df_1d.round(2).rename(columns={"date": "time"})
                        
                        # Update the first row 'close' with previousClose from your stored json if available
                        try:
                            with open(f"json/quote/{ticker}.json", 'r') as file:
                                res = ujson.load(file)
                                df_1d.loc[df_1d.index[0], 'close'] = res['previousClose']
                        except Exception as e:
                            pass
            
                        # Convert DataFrame back to JSON list format
                        df_1d = ujson.loads(df_1d.to_json(orient="records"))
                    except Exception as e:
                        print(f"Error processing data for {ticker}: {e}")
                        df_1d = []
        except Exception as e:
            print(f"Connection error for {ticker}: {e}")
            df_1d = []
    
    return df_1d

async def run():
    # Create a semaphore to limit the number of concurrent connections
    # Adjust this number based on your system's limits
    connection_limit = 1000
    semaphore = asyncio.Semaphore(connection_limit)
    
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


    
    for symbol in stocks_symbols:
        try:
            with open(f"json/quote/{symbol}.json", "r") as file:
                quote_data = orjson.loads(file.read())
                # Get market cap; if it's None (or not a number), force 0
                raw_mcap = quote_data.get('marketCap')
                if not isinstance(raw_mcap, (int, float)):
                    raw_mcap = 0
                market_caps[symbol] = raw_mcap
        except FileNotFoundError:
            market_caps[symbol] = 0



    # Sort symbols by market cap in descending order (largest first)
    stocks_symbols = sorted(stocks_symbols, key=lambda s: market_caps[s], reverse=True)
    stocks_symbols = sorted(stocks_symbols, key=lambda x: '.' in x)

    total_symbols = stocks_symbols + etf_symbols + index_symbols

    # Reduce chunk size to avoid too many concurrent requests
    chunk_size = 250
    for i in range(0, len(total_symbols), chunk_size):
        symbols_chunk = total_symbols[i:i+chunk_size]
        await fetch_and_save_symbols_data(symbols_chunk, semaphore)
        print(f'Completed chunk {i//chunk_size + 1} of {(len(total_symbols) + chunk_size - 1) // chunk_size}')
        # No need to sleep as much since we're using a semaphore to control concurrency
        await asyncio.sleep(13)


try:
    asyncio.run(run())
except Exception as e:
    print(f"Main error: {e}")