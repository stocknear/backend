from datetime import timedelta
import os
import pandas as pd
import json
from pathlib import Path
import time
import ujson
import sqlite3
import orjson


def save_json(symbol, data):
    with open(f"json/fail-to-deliver/companies/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

def get_total_data(files_available, limit=24):
    """
    Combine all the 1/2 monthly csv into 1 large csv file.
    """
    combined_df = pd.DataFrame(columns=["SETTLEMENT DATE", "SYMBOL", "QUANTITY (FAILS)", "PRICE"])
    for file in files_available[:limit]:
        print(f"Processing file: {file}")
        try:
            # Read the CSV file with appropriate parameters
            df = pd.read_csv(file, sep='|', quotechar='"', engine='python')
            
            # Safely remove columns if they exist
            if 'CUSIP' in df.columns:
                del df['CUSIP']
            if 'DESCRIPTION' in df.columns:
                del df['DESCRIPTION']
            
            # Safely convert SETTLEMENT DATE
            if 'SETTLEMENT DATE' in df.columns:
                df['SETTLEMENT DATE'] = pd.to_datetime(df['SETTLEMENT DATE'], format='%Y%m%d', errors='coerce')
            
            combined_df = pd.concat([combined_df, df]).drop_duplicates()
        
        except pd.errors.ParserError as e:
            print(f"Error reading {file}: {e}")
        except Exception as e:
            print(f"Unexpected error with {file}: {e}")
    
    combined_df["SETTLEMENT DATE"] = combined_df["SETTLEMENT DATE"].astype(str)
    combined_df.rename(columns={
        "SETTLEMENT DATE": "date",
        "SYMBOL": "Ticker",
        "QUANTITY (FAILS)": "failToDeliver",
        "PRICE": "price"
    }, inplace=True)
    
    combined_df["T+35 Date"] = pd.to_datetime(combined_df['date'], format='%Y-%m-%d', errors="coerce") + timedelta(days=35)
    combined_df["T+35 Date"] = combined_df["T+35 Date"].astype(str)

    combined_df["failToDeliver"] = pd.to_numeric(combined_df["failToDeliver"], errors='coerce')
    combined_df["failToDeliver"] = combined_df["failToDeliver"].fillna(0).astype(int)
    
    combined_df = combined_df[~combined_df["Ticker"].isna()]
    combined_df.sort_values(by="date", inplace=True)
    
    print(combined_df)
    return combined_df

def filter_by_ticker(combined_df):
    # Group by 'Ticker' column
    grouped = combined_df.groupby('Ticker')
    
    # Dictionary to store DataFrames for each ticker
    ticker_dfs = {}
    
    # Iterate over groups
    for ticker, group in grouped:
        # Store each group (DataFrame) in the dictionary
        ticker_dfs[ticker] = group.copy()  # Use .copy() to avoid modifying original DataFrame
    
    return ticker_dfs

if __name__ == '__main__':

    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    con.close()
    etf_con.close()

    total_symbols = stock_symbols + etf_symbols

    # Specify your directory path
    directory_path = 'json/fail-to-deliver/csv'
    
    # List CSV files sorted by modification time
    files_available = sorted(Path(directory_path).iterdir(), key=os.path.getmtime)
    #files_available = ['json/fail-to-deliver/csv/cnsfails202507a.csv']

    combined_df = get_total_data(files_available, limit=1000)

    
    ticker_dataframes = filter_by_ticker(combined_df)
    
    # Example usage: print or access dataframes for each ticker
    for ticker, df in ticker_dataframes.items():
        try:
            if ticker in total_symbols:
                data= [{k: v for k, v in d.items() if k not in ['Ticker', 'T+35 Date']} for d in df.to_dict('records')]

                with open(f"json/historical-price/adj/{ticker}.json", "rb") as file:
                    historical_price = orjson.loads(file.read())

                # Create a lookup dictionary from historical_price
                close_lookup = {item['date']: item['adjClose'] for item in historical_price}

                # Replace 'price' in data if the date exists in the lookup
                for entry in data:
                    try:
                        if entry['date'] in close_lookup:
                            entry['price'] = close_lookup[entry['date']]
                    except:
                        pass

                save_json(ticker, data)
        except:
            pass
    