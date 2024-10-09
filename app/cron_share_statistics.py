import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import yfinance as yf
import time


async def save_as_json(symbol, forward_pe_dict, short_dict):
    with open(f"json/share-statistics/{symbol}.json", 'w') as file:
        ujson.dump(short_dict, file)
    with open(f"json/forward-pe/{symbol}.json", 'w') as file:
        ujson.dump(forward_pe_dict, file)


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

def get_yahoo_data(ticker, outstanding_shares, float_shares):
    try:
        data_dict = yf.Ticker(ticker).info
        forward_pe = round(data_dict['forwardPE'],2)
        short_outstanding_percent = round((data_dict['sharesShort']/outstanding_shares)*100,2)
        short_float_percent = round((data_dict['sharesShort']/float_shares)*100,2)
        return {'forwardPE': forward_pe}, {'sharesShort': data_dict['sharesShort'], 'shortRatio': data_dict['shortRatio'], 'sharesShortPriorMonth': data_dict['sharesShortPriorMonth'], 'shortOutStandingPercent': short_outstanding_percent, 'shortFloatPercent': short_float_percent}
    except Exception as e:
        print(e)
        return {'forwardPE': 0}, {'sharesShort': 0, 'shortRatio': 0, 'sharesShortPriorMonth': 0, 'shortOutStandingPercent': 0, 'shortFloatPercent': 0}


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
        
        latest_outstanding_shares = shareholder_statistics[-1]['outstandingShares']
        latest_float_shares = shareholder_statistics[-1]['floatShares']

        # Filter out only quarter-end dates
        historical_shares = filter_data_quarterly(shareholder_statistics)

        forward_pe_data, short_data = get_yahoo_data(ticker, latest_outstanding_shares, latest_float_shares)
        short_data = {**short_data, 'latestOutstandingShares': latest_outstanding_shares, 'latestFloatShares': latest_float_shares,'historicalShares': historical_shares}
    except Exception as e:
        print(e)
        short_data = {}
        forward_pe_data = {}

    return forward_pe_data, short_data


async def run():

    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    
    counter = 0

    for ticker in tqdm(stock_symbols):
        forward_pe_dict, short_dict = await get_data(ticker, con)
        if forward_pe_dict.keys() and short_dict.keys():
            await save_as_json(ticker, forward_pe_dict, short_dict)

        counter += 1
        if counter % 50 == 0:
            print(f"Processed {counter} tickers, waiting for 60 seconds...")
            await asyncio.sleep(60)

    con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)