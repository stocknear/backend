import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm

import requests
import re



async def save_as_json(symbol, data):
    with open(f"json/shareholders/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


query_template = f"""
    SELECT 
        shareholders
    FROM 
        stocks
    WHERE
        symbol = ?
"""

async def get_data(ticker, con):

    try:
        df = pd.read_sql_query(query_template, con, params=(ticker,))
        shareholders_list = ujson.loads(df.to_dict()['shareholders'][0])
        # Keys to keep
        keys_to_keep = ["cik","filingDate","ownership", "investorName", "changeInSharesNumberPercentage", "weight", "sharesNumber", "marketValue"]

        # Create new list with only the specified keys
        shareholders_list = [
            {key: d[key] for key in keys_to_keep}
            for d in shareholders_list
        ]
    except Exception as e:
        #print(e)
        shareholders_list = []

    return shareholders_list


async def run():

    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    for ticker in tqdm(stock_symbols):
        shareholders_list = await get_data(ticker, con)
        if len(shareholders_list) > 0:
            await save_as_json(ticker, shareholders_list)

    con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)