import ujson
import asyncio
from tqdm import tqdm
import pandas as pd
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


async def save_json_file(symbol, data):
    with open(f"json/top-etf-ticker-holder/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

# Fetch all ETF data in one go
def fetch_all_etf_data(etf_symbols):
    etf_data = {}
    for etf_ticker in etf_symbols:
        try:
            df = pd.read_sql_query(query_template, etf_con, params=(etf_ticker,))
            etf_data[etf_ticker] = df
        except Exception as e:
            print(f"Error fetching data for {etf_ticker}: {e}")
    return etf_data


def process_etf(etf_ticker, stock_ticker, df):
    etf_weight_percentages = []
    try:
        for index, row in df.iterrows():
            holdings = ujson.loads(row['holding'])
            total_assets = int(row['totalAssets'])
            name = row['name']
            for holding in holdings:
                if holding['asset'] == stock_ticker:
                    etf_weight_percentages.append({
                        'symbol': etf_ticker,
                        'name': name,
                        'totalAssets': total_assets,
                        'weightPercentage': holding['weightPercentage']
                    })
                    break  # No need to continue checking if found
    except Exception as e:
        print(e)
    return etf_weight_percentages

async def save_and_process(stock_ticker, etf_data):
    etf_weight_percentages = []
    with ThreadPoolExecutor(max_workers=14) as executor:
        futures = [executor.submit(process_etf, etf_ticker, stock_ticker, df) for etf_ticker, df in etf_data.items()]
        for future in as_completed(futures):
            etf_weight_percentages.extend(future.result())

    # Filter out only the ETFs where totalAssets > 0
    etf_weight_percentages = [etf for etf in etf_weight_percentages if etf['totalAssets'] > 0]

    data = sorted(etf_weight_percentages, key=lambda x: x['weightPercentage'], reverse=True)[:5]
    if len(data) > 0:
        await save_json_file(stock_ticker, data)


async def run():
    
    # Main loop
    etf_data = fetch_all_etf_data(etf_symbols)
    for stock_ticker in tqdm(stocks_symbols):
        await save_and_process(stock_ticker, etf_data)

try:
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    query_template = """
        SELECT 
            name, totalAssets, holding
        FROM 
            etfs 
        WHERE
            symbol = ?
    """

    asyncio.run(run())
    con.close()
    etf_con.close()
except Exception as e:
    print(e)