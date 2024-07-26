import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import concurrent.futures
import json
from tqdm import tqdm
import warnings
import numpy as np
import os

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

def get_stock_prices(ticker, cursor, start_date, end_date):
    query = f"""
        SELECT date, close, volume
        FROM "{ticker}"
        WHERE date BETWEEN ? AND ?
    """
    cursor.execute(query, (start_date, end_date))
    return pd.DataFrame(cursor.fetchall(), columns=['date', 'close', 'volume'])

def process_symbol(op_symbol, symbols, start_date, end_date, query_fundamental):
    with sqlite3.connect('etf.db') as con:
        con.execute("PRAGMA journal_mode = WAL")
        cursor = con.cursor()
        
        op_df = get_stock_prices(op_symbol, cursor, start_date, end_date)
        avg_volume = op_df['volume'].mean() * 0.5
        correlations = {}

        for symbol in symbols:
            if symbol != op_symbol:
                try:
                    stock_df = get_stock_prices(symbol, cursor, start_date, end_date)
                    if stock_df['volume'].mean() > avg_volume:
                        correlation = np.corrcoef(op_df['close'], stock_df['close'])[0, 1]
                        correlations[symbol] = correlation
                except Exception:
                    pass

        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        most_least_list = sorted_correlations[:5] + sorted_correlations[-5:]
        
        res_list = []
        for symbol, correlation in most_least_list:
            cursor.execute(query_fundamental, (symbol,))
            fundamental_data = cursor.fetchone()
            if correlation is not None and not np.isnan(correlation):
                res_list.append({
                    'symbol': symbol,
                    'name': fundamental_data[0],
                    'marketCap': int(fundamental_data[1]),
                    'value': round(correlation, 3)
                })

        sorted_res = sorted(res_list, key=lambda x: x['value'], reverse=True)
        res_list = list({d['symbol']: d for d in sorted_res}.values())
        
        if res_list:
            os.makedirs("json/correlation/companies", exist_ok=True)
            with open(f"json/correlation/companies/{op_symbol}.json", 'w') as file:
                json.dump(res_list, file)

def main():
    query_fundamental = "SELECT name, marketCap FROM etfs WHERE symbol = ?"
    
    with sqlite3.connect('etf.db') as con:
        con.execute("PRAGMA journal_mode = WAL")
        cursor = con.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM etfs")
        symbols = [row[0] for row in cursor.fetchall()]

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)  # 12 months

    num_processes = 14  # As specified in your original code
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_symbol, symbol, symbols, start_date, end_date, query_fundamental) for symbol in symbols]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(symbols), desc="Processing"):
            pass

if __name__ == "__main__":
    main()