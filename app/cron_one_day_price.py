import ujson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime, timedelta, time
import pandas as pd
from GetStartEndDate import GetStartEndDate
from dotenv import load_dotenv
import os
from utils.helper import check_market_hours

load_dotenv()
api_key = os.getenv('FMP_API_KEY')



async def save_price_data(symbol, data):
    with open(f"json/one-day-price/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


async def fetch_and_save_symbols_data(symbols):
    tasks = []
    for symbol in symbols:
        task = asyncio.create_task(get_todays_data(symbol))
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    
    for symbol, response in zip(symbols, responses):
        await save_price_data(symbol, response)

async def get_todays_data(ticker):

    start_date_1d, end_date_1d = GetStartEndDate().run()


    current_weekday = end_date_1d.weekday()

    start_date = start_date_1d.strftime("%Y-%m-%d")
    end_date = end_date_1d.strftime("%Y-%m-%d")


    url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{ticker}?from={start_date}&to={end_date}&apikey={api_key}"

    df_1d = pd.DataFrame()

    current_date = start_date_1d
    target_time = time(9,30)

    extract_date = current_date.strftime('%Y-%m-%d')

    async with aiohttp.ClientSession() as session:
        responses = await asyncio.gather(session.get(url))

        for response in responses:
            try:
                json_data = await response.json()
                df_1d = pd.DataFrame(json_data).iloc[::-1].reset_index(drop=True)
                df_1d = df_1d.drop(['volume'], axis=1)
                df_1d = df_1d.round(2).rename(columns={"date": "time"})
                try:
                    with open(f"json/quote/{ticker}.json", 'r') as file:
                        res = ujson.load(file)
                        df_1d.loc[df_1d.index[0], 'close'] = res['previousClose']
                except:
                    pass

                if current_weekday == 5 or current_weekday == 6:
                    pass
                else:
                    if current_date.time() < target_time:
                        pass                    
                    else:
                        end_time = pd.to_datetime(f'{extract_date} 16:00:00')
                        new_index = pd.date_range(start=df_1d['time'].iloc[-1], end=end_time, freq='1min')
                        
                        remaining_df = pd.DataFrame(index=new_index, columns=['open', 'high', 'low','close'])
                        remaining_df = remaining_df.reset_index().rename(columns={"index": "time"})
                        remaining_df['time'] = remaining_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        remainind_df = remaining_df.set_index('time')

                        df_1d = pd.concat([df_1d, remaining_df[1::]], ignore_index=True)
                        #To-do FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
    
                df_1d = ujson.loads(df_1d.to_json(orient="records"))
            except Exception as e:
                print(e)
                df_1d = []

    res = df_1d

    return res

async def run():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')
    crypto_con = sqlite3.connect('crypto.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
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

    total_symbols = stocks_symbols + etf_symbols + crypto_symbols

    market_open = check_market_hours()
    
    if market_open:
        chunk_size = 1000
        for i in range(0, len(total_symbols), chunk_size):
            symbols_chunk = total_symbols[i:i+chunk_size]
            await fetch_and_save_symbols_data(symbols_chunk)
            print('sleeping for 45 sec')
            await asyncio.sleep(45)  # Wait for 60 seconds between chunks
    else:
        print('Market Closed')

try:
    asyncio.run(run())
except Exception as e:
    print(e)