import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime



async def save_as_json(symbol, data):
    with open(f"json/share-statistics/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


query_template = f"""
    SELECT 
        historicalShares
    FROM 
        stocks
    WHERE
        symbol = ?
"""


def filter_data_quarterly(data):
    # Generate a range of quarter-end dates from the start to the end date
    start_date = data[0]['date']
    end_date = datetime.today().strftime('%Y-%m-%d')
    quarter_ends = pd.date_range(start=start_date, end=end_date, freq='QE').strftime('%Y-%m-%d').tolist()

    # Filter data to keep only entries with dates matching quarter-end dates
    filtered_data = [entry for entry in data if entry['date'] in quarter_ends]
    
    return filtered_data


async def get_data(ticker, con):

    try:
        df = pd.read_sql_query(query_template, con, params=(ticker,))
        shareholder_statistics = ujson.loads(df.to_dict()['historicalShares'][0])
        # Keys to keep
        keys_to_keep = ["date","floatShares", "outstandingShares"]

        # Create new list with only the specified keys and convert floatShares and outstandingShares to integers
        shareholder_statistics = [
            {key: int(d[key]) if key in ["floatShares", "outstandingShares"] else d[key] 
             for key in keys_to_keep}
            for d in shareholder_statistics
        ]

        shareholder_statistics = sorted(shareholder_statistics, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=False)
        # Filter out only quarter-end dates
        shareholder_statistics = filter_data_quarterly(shareholder_statistics)
    except Exception as e:
        #print(e)
        shareholder_statistics = []

    return shareholder_statistics


async def run():

    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    
    for ticker in tqdm(stock_symbols):
        shareholder_statistics = await get_data(ticker, con)
        if len(shareholder_statistics) > 0:
            await save_as_json(ticker, shareholder_statistics)
    
    con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)
