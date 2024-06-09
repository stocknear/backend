import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm


query_template = """
    SELECT 
        analyst_estimates, income
    FROM 
        stocks
    WHERE
        symbol = ?
"""

async def save_as_json(symbol, data):
    with open(f"json/shareholders/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


async def get_data(ticker, etf_symbols, con, etf_con):
    if ticker in etf_symbols:
        table_name = 'etfs'
    else:
        table_name = 'stocks'

    query_template = f"""
        SELECT 
            shareholders
        FROM 
            {table_name} 
        WHERE
            symbol = ?
    """
    try:
        df = pd.read_sql_query(query_template, etf_con if table_name == 'etfs' else con, params=(ticker,))
        shareholders_list = ujson.loads(df.to_dict()['shareholders'][0])
        # Keys to keep
        keys_to_keep = ["cik","ownership", "investorName", "weight", "sharesNumber", "marketValue"]

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
    etf_con = sqlite3.connect('etf.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    total_symbols = stock_symbols + etf_symbols

    for ticker in tqdm(total_symbols):
        shareholders_list = await get_data(ticker, etf_symbols, con, etf_con)
        if len(shareholders_list) > 0:
            await save_as_json(ticker, shareholders_list)

    con.close()
    etf_con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)
