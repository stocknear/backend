import ujson
import asyncio
import aiohttp
import aiofiles
import sqlite3
from datetime import datetime, timedelta
import pytz
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

current_year = datetime.today().year

# Helper to ensure directories exist and write JSON files asynchronously
async def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if data:
        async with aiofiles.open(path, 'w') as file:
            await file.write(ujson.dumps(data))

async def get_historical_data(ticker, query_con, session):
    try:
        # Form API request URLs
        url_1w = (f"https://financialmodelingprep.com/stable/historical-chart/5min?"
                  f"symbol={ticker}&from={start_date_1w}&to={end_date}&apikey={api_key}")
        url_1m = (f"https://financialmodelingprep.com/stable/historical-chart/1hour?"
                  f"symbol={ticker}&from={start_date_1m}&to={end_date}&apikey={api_key}")

        # Fetch both endpoints concurrently
        responses = await asyncio.gather(
            session.get(url_1w),
            session.get(url_1m),
            return_exceptions=True
        )

        data = []
        for resp in responses:
            try:
                if isinstance(resp, Exception):
                    print(f"Error fetching data for {ticker}: {resp}")
                    continue
                async with resp:
                    if resp.status != 200:
                        print(f"Non-200 response for {ticker}: {resp.status}")
                        continue
                    else:
                        json_data = await resp.json()
                        # Reverse rows so that oldest data comes first and reset the index
                        df = pd.DataFrame(json_data).iloc[::-1].reset_index(drop=True)
                        df = df.round(2).rename(columns={"date": "time"})
                        data.append(df.to_json(orient="records"))
            except:
                pass

        # Database queries for additional periods
        query_template = """
            SELECT date, open, high, low, close, volume
            FROM "{ticker}"
            WHERE date BETWEEN ? AND ?
        """
        query = query_template.format(ticker=ticker)
        df_6m = pd.read_sql_query(query, query_con, params=(start_date_6m, end_date))
        df_6m = df_6m.round(2).rename(columns={"date": "time"})
        df_1y = pd.read_sql_query(query, query_con, params=(start_date_1y, end_date))
        df_1y = df_1y.round(2).rename(columns={"date": "time"})
        df_5y = pd.read_sql_query(query, query_con, params=(start_date_5y, end_date))
        df_5y = df_5y.round(2).rename(columns={"date": "time"})
        df_max = pd.read_sql_query(query, query_con, params=(start_date_max, end_date))
        df_max = df_max.round(2).rename(columns={"date": "time"})

        max_list = ujson.loads(df_max.to_json(orient="records"))
        ytd_data = [entry for entry in max_list if datetime.strptime(entry["time"], "%Y-%m-%d").year == current_year]

        # Prepare file-writing tasks
        tasks = [
            write_json(f"json/historical-price/one-week/{ticker}.json", ujson.loads(data[0])),
            write_json(f"json/historical-price/one-month/{ticker}.json", ujson.loads(data[1])),
            write_json(f"json/historical-price/ytd/{ticker}.json", ytd_data),
            write_json(f"json/historical-price/six-months/{ticker}.json", ujson.loads(df_6m.to_json(orient="records"))),
            write_json(f"json/historical-price/one-year/{ticker}.json", ujson.loads(df_1y.to_json(orient="records"))),
            write_json(f"json/historical-price/five-years/{ticker}.json", ujson.loads(df_5y.to_json(orient="records"))),
            write_json(f"json/historical-price/max/{ticker}.json", ujson.loads(df_max.to_json(orient="records")))
        ]
        await asyncio.gather(*tasks)

    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")

async def fetch_and_save_symbols_data(symbols, etf_symbols, index_symbols, session):
    tasks = []
    for symbol in symbols:
        if symbol in etf_symbols:
            query_con = etf_con
        elif symbol in index_symbols:
            query_con = index_con
        else:
            query_con = con

        task = asyncio.create_task(get_historical_data(symbol, query_con, session))
        tasks.append(task)
    # Wait for all tasks in this chunk to complete
    await asyncio.gather(*tasks)

async def run():
    try:
        # Prepare symbols list
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks")
        stock_symbols = [row[0] for row in cursor.fetchall()]

        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]

        index_cursor = index_con.cursor()
        index_cursor.execute("PRAGMA journal_mode = wal")
        index_cursor.execute("SELECT DISTINCT symbol FROM indices")
        index_symbols = [row[0] for row in index_cursor.fetchall()]

        total_symbols = stock_symbols + etf_symbols + index_symbols
    except Exception as e:
        print(f"Failed to fetch symbols: {e}")
        return

    # Process symbols in chunks to avoid overwhelming the API
    chunk_size = 100
    try:
        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in range(0, len(total_symbols), chunk_size):
                symbols_chunk = total_symbols[i:i + chunk_size]
                await fetch_and_save_symbols_data(symbols_chunk, etf_symbols, index_symbols, session)
                print('Chunk processed; sleeping for 30 seconds...')
                await asyncio.sleep(30)
    except Exception as e:
        print(f"Failed to run fetch and save data: {e}")

if __name__ == "__main__":
    try:
        # Open SQLite connections
        con = sqlite3.connect('stocks.db')
        etf_con = sqlite3.connect('etf.db')
        index_con = sqlite3.connect('index.db')

        # Prepare date variables
        berlin_tz = pytz.timezone('Europe/Berlin')
        now = datetime.now(berlin_tz)
        end_date = now.strftime("%Y-%m-%d")
        start_date_1w = (now - timedelta(days=5)).strftime("%Y-%m-%d")
        start_date_1m = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        start_date_6m = (now - timedelta(days=180)).strftime("%Y-%m-%d")
        start_date_1y = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        start_date_5y = (now - timedelta(days=365*5)).strftime("%Y-%m-%d")
        start_date_max = datetime(1970, 1, 1).strftime("%Y-%m-%d")

        asyncio.run(run())

        con.close()
        etf_con.close()
        index_con.close()
    except Exception as e:
        print(e)
