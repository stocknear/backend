import ujson
import asyncio
import aiohttp
import aiofiles
import sqlite3
from datetime import datetime, timedelta, time
import pytz
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')


async def fetch_and_save_symbols_data(symbols, etf_symbols, crypto_symbols, session):
    tasks = []
    for symbol in symbols:
        if symbol in etf_symbols:
            query_con = etf_con
        elif symbol in crypto_symbols:
            query_con = crypto_con
        else:
            query_con = con

        task = asyncio.create_task(get_historical_data(symbol, query_con, session))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    

async def get_historical_data(ticker, query_con, session):
    try:
        # Form API request URLs
        url_1w = f"https://financialmodelingprep.com/api/v3/historical-chart/30min/{ticker}?from={start_date_1w}&to={end_date}&apikey={api_key}"
        url_1m = f"https://financialmodelingprep.com/api/v3/historical-chart/1hour/{ticker}?from={start_date_1m}&to={end_date}&apikey={api_key}"
        
        async with session.get(url_1w) as response_1w, session.get(url_1m) as response_1m:
            data = []
            for response in [response_1w, response_1m]:
                json_data = await response.json()
                df = pd.DataFrame(json_data).iloc[::-1].reset_index(drop=True)
                try:
                    df = df.drop(['volume'], axis=1)
                except:
                    pass
                df = df.round(2).rename(columns={"date": "time"})
                data.append(df.to_json(orient="records"))

            # Database read for 6M, 1Y, MAX data
            query_template = """
                SELECT date, open,high,low,close
                FROM "{ticker}"
                WHERE date BETWEEN ? AND ?
            """
            query = query_template.format(ticker=ticker)
            df_6m = pd.read_sql_query(query, query_con, params=(start_date_6m, end_date)).round(2).rename(columns={"date": "time"})
            df_1y = pd.read_sql_query(query, query_con, params=(start_date_1y, end_date)).round(2).rename(columns={"date": "time"})
            df_max = pd.read_sql_query(query, query_con, params=(start_date_max, end_date)).round(2).rename(columns={"date": "time"})

            async with aiofiles.open(f"json/historical-price/one-week/{ticker}.json", 'w') as file:
                res = ujson.loads(data[0]) if data else []
                await file.write(ujson.dumps(res))

            async with aiofiles.open(f"json/historical-price/one-month/{ticker}.json", 'w') as file:
                res = ujson.loads(data[1]) if len(data) > 1 else []
                await file.write(ujson.dumps(res))

            async with aiofiles.open(f"json/historical-price/six-months/{ticker}.json", 'w') as file:
                res = ujson.loads(df_6m.to_json(orient="records"))
                await file.write(ujson.dumps(res))

            async with aiofiles.open(f"json/historical-price/one-year/{ticker}.json", 'w') as file:
                res = ujson.loads(df_1y.to_json(orient="records"))
                await file.write(ujson.dumps(res))

            async with aiofiles.open(f"json/historical-price/max/{ticker}.json", 'w') as file:
                res = ujson.loads(df_max.to_json(orient="records"))
                await file.write(ujson.dumps(res))

    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")

async def run():
    total_symbols = []
    try:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks")
        stock_symbols = [row[0] for row in cursor.fetchall()]

        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]

        crypto_cursor = crypto_con.cursor()
        crypto_cursor.execute("PRAGMA journal_mode = wal")
        crypto_cursor.execute("SELECT DISTINCT symbol FROM cryptos")
        crypto_symbols = [row[0] for row in crypto_cursor.fetchall()]

        total_symbols = stock_symbols + etf_symbols + crypto_symbols
    except Exception as e:
        print(f"Failed to fetch symbols: {e}")
        return

    try:
        connector = aiohttp.TCPConnector(limit=100)  # Adjust the limit as needed
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in range(0, len(total_symbols), chunk_size):
                symbols_chunk = total_symbols[i:i + chunk_size]
                await fetch_and_save_symbols_data(symbols_chunk, etf_symbols, crypto_symbols, session)
                print('sleeping for 60 sec')
                await asyncio.sleep(60)  # Wait for 60 seconds between chunks
    except Exception as e:
        print(f"Failed to run fetch and save data: {e}")

try:
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')
    crypto_con = sqlite3.connect('crypto.db')

    berlin_tz = pytz.timezone('Europe/Berlin')
    end_date = datetime.now(berlin_tz)
    start_date_1w = (end_date - timedelta(days=7)).strftime("%Y-%m-%d")
    start_date_1m = (end_date - timedelta(days=30)).strftime("%Y-%m-%d")
    start_date_6m = (end_date - timedelta(days=180)).strftime("%Y-%m-%d")
    start_date_1y = (end_date - timedelta(days=365)).strftime("%Y-%m-%d")
    start_date_max = datetime(1970, 1, 1).strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")


    chunk_size = 250
    asyncio.run(run())
    con.close()
    etf_con.close()
    crypto_con.close()
except Exception as e:
    print(e)

