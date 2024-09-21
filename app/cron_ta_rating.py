import ujson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
from rating import rating_model
import pandas as pd
from tqdm import tqdm

async def save_ta_rating(symbol, data):
    with open(f"json/ta-rating/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


async def run():
    start_date = "2022-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

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

    total_symbols = stocks_symbols + etf_symbols + crypto_symbols

    for symbol in tqdm(total_symbols):
        try:
            table_name = None
            if symbol in etf_symbols:  # Fixed variable name from symbols to symbol
                query_con = etf_con
            elif symbol in crypto_symbols:
                query_con = crypto_con
            elif symbol in stocks_symbols:
                query_con = con

            query_template = """
                    SELECT
                        date, open, high, low, close, volume
                    FROM
                        "{symbol}"
                    WHERE
                        date BETWEEN ? AND ?
                """
            query = query_template.format(symbol=symbol)
            df = pd.read_sql_query(query,query_con, params=(start_date, end_date))

            try:
                # Assuming rating_model and save_quote_as_json are defined elsewhere
                res_dict = rating_model(df).ta_rating()
                await save_ta_rating(symbol, res_dict)
            except Exception as e:
                print(e)
        except:
            pass
            

    con.close()
    etf_con.close()
    crypto_con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)