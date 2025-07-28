import os
import pandas as pd
import numpy as np
import orjson
from dotenv import load_dotenv
import sqlite3
from datetime import datetime, timedelta
import pytz
from typing import List, Dict
import sqlite3
from tqdm import tqdm
import time
from collections import defaultdict
from utils.helper import check_market_hours

utc = pytz.utc
ny_tz = pytz.timezone("America/New_York")

def save_json(data, symbol):
    def convert_numpy(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Convert numpy scalar to Python scalar
        raise TypeError(f"Type is not JSON serializable: {type(obj)}")

    directory = "json/dark-pool/price-level"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory}/{symbol}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data, default=convert_numpy))

def get_last_N_weekdays():
    today = datetime.today()
    weekdays = []
    
    while len(weekdays) < 3:
        if today.weekday() < 5:  # Monday to Friday are weekdays (0-4)
            weekdays.append(today)
        today -= timedelta(days=1)
    
    weekdays = [item.strftime("%Y-%m-%d") for item in weekdays]
    return weekdays


def analyze_dark_pool_levels(trades: List[Dict], 
                             size_threshold: float = 0, 
                             price_grouping: float = 1.0) -> Dict:
    if not trades or not isinstance(trades, list):
        return {}
    try:
        df = pd.DataFrame(trades)
        if df.empty:
            return {}

        # Ensure necessary columns exist
        if 'premium' not in df or 'price' not in df or 'size' not in df:
            return {}

        # Convert premium strings to float values
        df['premium'] = df['premium'].apply(lambda x: float(str(x).replace(',', '')))
        df['price_level'] = (df['price'] / price_grouping).round(1) * price_grouping

        size_by_price = df.groupby('price_level').agg({
            'size': 'sum',
            'premium': 'sum'
        }).reset_index()

        min_size = size_by_price['size'].quantile(size_threshold)
        significant_levels = size_by_price[size_by_price['size'] >= min_size]
        significant_levels = significant_levels.sort_values('size', ascending=False)

        current_price = df['price'].iloc[-1]
        support_levels = significant_levels[significant_levels['price_level'] < current_price].to_dict('records')
        resistance_levels = significant_levels[significant_levels['price_level'] > current_price].to_dict('records')

        metrics = {
            'avgTradeSize': round(df['size'].mean(), 2),
            'totalPrem': round(df['premium'].sum(), 2),
            'avgPremTrade': round(df['premium'].mean(), 2)
        }

        price_level = support_levels + resistance_levels
        price_level = sorted(price_level, key=lambda x: float(x['price_level']))

        return {
            'price_level': price_level,
            'metrics': metrics,
        }
    except Exception as e:
        print(f"Error analyzing dark pool levels: {e}")
        return {}


def today_trend(data):
    filtered_list = []
    result = []
    for item in data:
        try:
            # Convert date to NY timezone
            dt_utc = datetime.fromisoformat(item['date'][:-6]).replace(tzinfo=utc)
            dt_ny = dt_utc.astimezone(ny_tz)
            
            # Define trading hours (9:30 AM - 4:00 PM NY time)
            market_open = dt_ny.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = dt_ny.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if market_open <= dt_ny <= market_close:  # Filter valid times
                filtered_list.append({
                    'size': item['size'],
                    'date': dt_ny.strftime("%Y-%m-%d %H:%M")  # Format as HH:MM
                })
        except:
            pass

    filtered_list.sort(key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d %H:%M"))
    
    summed_data = defaultdict(float)
    for entry in filtered_list:
        try:
            summed_data[entry['date']] += entry['size']
        except:
            pass

    result = [{'date': date, 'totalSize': size} for date, size in summed_data.items()]

    return result

def run():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    con.close()
    etf_con.close()

    total_symbols = stocks_symbols+ etf_symbols
    with open(f"json/dark-pool/feed/data.json", "r") as file:
        raw_data = orjson.loads(file.read())

    market_status = check_market_hours()
    if market_status:
        for symbol in tqdm(total_symbols):
            try:
                res_list = [item for item in raw_data if isinstance(item, dict) and item['ticker'] == symbol]
                
                trend_list = today_trend(res_list)
                
                dark_pool_levels = analyze_dark_pool_levels(
                    trades=res_list,
                    size_threshold=0,
                    price_grouping=1.0
                )
                
                if dark_pool_levels.get('price_level'):  # Ensure there are valid levels
                    top_N_elements = [
                        {k: v for k, v in item.items() if k not in ['ticker', 'sector', 'assetType']}
                        for item in sorted(res_list, key=lambda x: float(x.get('premium', 0)), reverse=True)[:20]
                    ]

                    # Add rank to each item
                    for rank, item in enumerate(top_N_elements, 1):
                        item['rank'] = rank

                    data_to_save = {
                        'hottestTrades': top_N_elements,
                        'priceLevel': dark_pool_levels['price_level'],
                        'trend': trend_list,
                        'metrics': dark_pool_levels['metrics']
                    }

                    save_json(data_to_save, symbol)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")



if __name__ == "__main__":
    run()