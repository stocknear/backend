import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import yfinance as yf

# Constants
JSON_DIR = "json/"
QUARTERLY_FREQ = 'QE'

# SQL Query
QUERY_TEMPLATE = """
    SELECT historicalShares
    FROM stocks
    WHERE symbol = ?
"""

def filter_quarterly_data(data):
    """Filter data to keep only quarter-end dates."""
    quarter_ends = pd.date_range(start=data[0]['date'], end=datetime.now(), freq=QUARTERLY_FREQ).strftime('%Y-%m-%d').tolist()
    return [entry for entry in data if entry['date'] in quarter_ends]

def get_yahoo_finance_data(ticker, shares):
    """Fetch and process Yahoo Finance data."""
    try:
        info = yf.Ticker(ticker).info
        return {
            'forwardPE': round(info.get('forwardPE', 0), 2),
            'short': {
                'shares': info.get('sharesShort', 0),
                'ratio': info.get('shortRatio', 0),
                'priorMonth': info.get('sharesShortPriorMonth', 0),
                'outstandingPercent': round((info.get('sharesShort', 0) / shares['outstandingShares']) * 100, 2),
                'floatPercent': round((info.get('sharesShort', 0) / shares['floatShares']) * 100, 2)
            }
        }
    except Exception as e:
        #print(ticker)
        #print(e)
        #print("============")
        return {'forwardPE': 0, 'short': {k: 0 for k in ['shares', 'ratio', 'priorMonth', 'outstandingPercent', 'floatPercent']}}

async def save_json(symbol, data):
    """Save data to JSON files."""
    for key, path in [("forwardPE", f"{JSON_DIR}forward-pe/{symbol}.json"), ("short", f"{JSON_DIR}share-statistics/{symbol}.json")]:
        with open(path, 'w') as file:
            ujson.dump(data.get(key, {}), file)

async def process_ticker(ticker, con):
    """Process a single ticker."""
    try:
        df = pd.read_sql_query(QUERY_TEMPLATE, con, params=(ticker,))
        stats = ujson.loads(df.to_dict()['historicalShares'][0])
        
        # Filter and convert data
        filtered_stats = [
            {k: int(v) if k in ["floatShares", "outstandingShares"] else v 
             for k, v in d.items() if k in ["date", "floatShares", "outstandingShares"]}
            for d in sorted(stats, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
        ]
        
        latest_shares = filtered_stats[-1]
        
        quarterly_stats = filter_quarterly_data(filtered_stats)
        
        data = get_yahoo_finance_data(ticker, latest_shares)
        data['short'].update({
            'latestOutstandingShares': latest_shares['outstandingShares'],
            'latestFloatShares': latest_shares['floatShares'],
            'historicalShares': quarterly_stats
        })
        
        await save_json(ticker, data)
        return True
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return False

async def run():
    """Main function to process all tickers."""
    con = sqlite3.connect('stocks.db')
    con.execute("PRAGMA journal_mode = wal")
    
    with con:
        stock_symbols = [row[0] for row in con.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")]

    processed = 0
    for ticker in tqdm(stock_symbols):
        if await process_ticker(ticker, con):
            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed} tickers, waiting for 60 seconds...")
                await asyncio.sleep(60)

    con.close()

if __name__ == "__main__":
    asyncio.run(run())