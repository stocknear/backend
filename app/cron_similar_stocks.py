import ujson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
from rating import rating_model
import pandas as pd
from tqdm import tqdm

async def save_similar_stocks(symbol, data):
    with open(f"json/similar-stocks/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


query_template = """
    SELECT 
        quote, stock_peers
    FROM 
        stocks 
    WHERE
        symbol = ?
"""

    

async def run():
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol != ?", ('%5EGSPC',))
    stocks_symbols = [row[0] for row in cursor.fetchall()]
    #stocks_symbols = ['AMD']
    for ticker in stocks_symbols:
        filtered_df = []
        df = pd.read_sql_query(query_template, con, params=(ticker,))
        try:
            df = ujson.loads(df['stock_peers'].iloc[0])
        except:
            df = []
        if len(df) > 0:
            df = [stock for stock in df if stock in stocks_symbols]
            for symbol in df:
                try:
                    df = pd.read_sql_query(query_template, con, params=(symbol,))
                    df_dict = df.to_dict()
                    quote_dict = eval(df_dict['quote'][0])[0]
                    filtered_df.append(quote_dict)  # Add the modified result to the combined list
                except:
                    pass

        filtered_df = [
            {
                "symbol": entry["symbol"],
                "name": entry["name"],
                "marketCap": entry["marketCap"],
                "avgVolume": entry["avgVolume"]
            }
            for entry in filtered_df
        ]

        sorted_df = sorted(filtered_df, key=lambda x: x['marketCap'], reverse=True)

        await save_similar_stocks(ticker, sorted_df)

try:
    con = sqlite3.connect('stocks.db')
    asyncio.run(run())
    con.close()
except Exception as e:
    print(e)