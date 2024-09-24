import ujson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
from ml_models.classification import TrendPredictor
import yfinance as yf
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import re
import subprocess

async def save_json(symbol, data):
    with open(f"json/trend-analysis/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

async def download_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        df = df.rename(columns={'Adj Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume', 'Date': 'date'})
        return df
    except Exception as e:
        print(e)

def convert_symbols(symbol_list):
    """
    Converts the symbols in the given list from 'BTCUSD' and 'USDTUSD' format to 'BTC-USD' and 'USDT-USD' format.
    
    Args:
        symbol_list (list): A list of strings representing the symbols to be converted.
    
    Returns:
        list: A new list with the symbols converted to the desired format.
    """
    converted_symbols = []
    for symbol in symbol_list:
        # Determine the base and quote currencies
        base_currency = symbol[:-3]
        quote_currency = symbol[-3:]
        
        # Construct the new symbol in the desired format
        new_symbol = f"{base_currency}-{quote_currency}"
        converted_symbols.append(new_symbol)
    
    return converted_symbols

async def process_symbol(ticker, start_date, end_date, crypto_symbols):
    try:
        best_features = ['close','williams','fi','emv','adi','cmf','bb_hband','bb_lband','vpt','stoch','stoch_rsi','rsi','nvi','macd','mfi','cci','obv','adx','adx_pos','adx_neg']
        test_size = 0.2
        df = await download_data(ticker, start_date, end_date)

        async def process_nth_day(nth_day):
            try:
                predictor = TrendPredictor(nth_day=nth_day, path="ml_models/weights")
                df_copy = df.copy()
                df_copy["Target"] = ((df_copy["close"].shift(-nth_day) > df_copy["close"])).astype(int)
                predictors = predictor.generate_features(df_copy)
                df_copy = df_copy.dropna(subset=df_copy.columns[df_copy.columns != "nth_day"])
                split_size = int(len(df_copy) * (1-test_size))
                test_data = df_copy.iloc[split_size:]

                res_dict = predictor.evaluate_model(test_data[best_features], test_data['Target'])

                if nth_day == 5:
                    time_period = 'oneWeek'
                elif nth_day == 20:
                    time_period = 'oneMonth'
                elif nth_day == 60:
                    time_period = 'threeMonth'

                return {'label': time_period, **res_dict}

            except Exception as e:
                print(e)
                return None

        tasks = [process_nth_day(nth_day) for nth_day in [5, 20, 60]]
        results = await asyncio.gather(*tasks)
        res_list = [r for r in results if r is not None]

        if ticker in crypto_symbols:
            ticker = ticker.replace('-','') #convert back from BTC-USD to BTCUSD

        await save_json(ticker, res_list)

    except Exception as e:
        print(e)

async def run():

    #Train first model
    try:
        print('training...')
        subprocess.run(["python3", "ml_models/classification.py", "--train"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running classification.py: {e}")

    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')
    crypto_con = sqlite3.connect('crypto.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    #cursor.execute("SELECT DISTINCT symbol FROM stocks")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap > 1E9 AND symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs WHERE totalAssets > 5E9")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    crypto_cursor = crypto_con.cursor()
    crypto_cursor.execute("PRAGMA journal_mode = wal")
    crypto_cursor.execute("SELECT DISTINCT symbol FROM cryptos")
    crypto_symbols = [row[0] for row in crypto_cursor.fetchall()]
    crypto_symbols = convert_symbols(crypto_symbols)

    con.close()
    etf_con.close()
    crypto_con.close()

    total_symbols = stock_symbols + etf_symbols + crypto_symbols
    print(f"Total tickers: {len(total_symbols)}")
    start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")

    chunk_size = len(total_symbols) // 70  # Divide the list into N chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
    for chunk in chunks:
        tasks = []
        for ticker in tqdm(chunk):
            tasks.append(process_symbol(ticker, start_date, end_date, crypto_symbols))

        await asyncio.gather(*tasks)

try:
    asyncio.run(run())
except Exception as e:
    print(e)
