import ujson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
from ml_models.prophet_model import PricePredictor
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import orjson
import os

def convert_symbols(symbol_list):
    converted_symbols = []
    for symbol in symbol_list:
        # Determine the base and quote currencies
        base_currency = symbol[:-3]
        quote_currency = symbol[-3:]
        
        # Construct the new symbol in the desired format
        new_symbol = f"{base_currency}-{quote_currency}"
        converted_symbols.append(new_symbol)
    
    return converted_symbols

async def save_json(symbol, data):
    path = "json/price-analysis"
    os.makedirs(path, exist_ok=True)  # Ensure directory exists
    with open(f"{path}/{symbol}.json", 'w') as file:
        ujson.dump(data, file)
        
async def download_data(ticker: str):
    try:
        with open(f"json/historical-price/adj/{ticker}.json", "r") as file:
            data = orjson.loads(file.read())

        df = pd.DataFrame(data)
        # Rename columns to ensure consistency
        df = df.rename(columns={"date": "ds", "adjClose": "y",})

        # Ensure correct data types
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)

        # Convert start_date and end_date from string to datetime

        # Apply filtering logic if enough data exists
        if len(df) > 252 * 2:  # At least 2 years of history is necessary
            q_high = df["y"].quantile(0.99)
            q_low = df["y"].quantile(0.1)
            df = df[(df["y"] > q_low) & (df["y"] < q_high)]

        # Calculate Simple Moving Average (SMA)
        #df["y"] = df["y"].rolling(window=50).mean()  # 50-day SMA

        return df.dropna()  # Drop initial NaN values due to rolling window
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

async def process_symbol(ticker):
    file_path = f"json/price-analysis/{ticker}.json"  # Declare early to avoid UnboundLocalError
    try:
        df = await download_data(ticker)
        data = PricePredictor().run(df)

        if data and data['lowPriceTarget'] > 0:
            print(data)
            await save_json(ticker, data)
        else:
            await asyncio.to_thread(os.remove, file_path)
    
    except Exception as e:
        print(f"Error in process_symbol for {ticker}: {e}")
        try:
            await asyncio.to_thread(os.remove, file_path)
        except FileNotFoundError:
            pass  # The file might not exist, so we ignore the error



async def run():
    con = sqlite3.connect('stocks.db')
    
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    total_symbols = stock_symbols
    print(f"Total tickers: {len(total_symbols)}")
    start_date = datetime(2015, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")

    #df_sp500 = await download_data('SPY')
    #df_sp500 = df_sp500.rename(columns={"y": "sp500"})
    #print(df_sp500)

    chunk_size = len(total_symbols) // 70  # Divide the list into N chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
    #chunks = [['AAPL']]
    for chunk in chunks:
        try:
            tasks = []
            for ticker in tqdm(chunk):
                try:
                    tasks.append(process_symbol(ticker))
                except:
                    pass
            await asyncio.gather(*tasks)
        except:
            pass

try:
    asyncio.run(run())
except Exception as e:
    print(e)
