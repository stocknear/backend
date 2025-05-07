import os
import pandas as pd
import orjson
from dotenv import load_dotenv
import sqlite3
from datetime import datetime, timezone
import pytz
import requests  # Add missing import
from dateutil.parser import isoparse
from utils.helper import load_latest_json
import time
import hashlib
import requests
from tqdm import tqdm

load_dotenv()
api_key = os.getenv('INTRINIO_API_KEY')

today_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

start_date = today_date
end_date = today_date
with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

quote_cache = {}



def get_quote_data(symbol):
    """Get quote data for a symbol from JSON file"""
    if symbol in quote_cache:
        return quote_cache[symbol]
    try:
        with open(f"json/quote/{symbol}.json") as file:
            quote_data = orjson.loads(file.read())
            quote_cache[symbol] = quote_data  # Cache the loaded data
            return quote_data
    except FileNotFoundError:
        return None


def save_json(data):
    directory = "json/dark-pool/feed"
    try:
        # Create a set to track unique entries based on a combination of 'ticker' and 'trackingID'
        seen = set()
        unique_data = []

        for item in data:
            identifier = f"{item['trackingID']}"
            if identifier not in seen:
                seen.add(identifier)
                unique_data.append(item)

        # Sort the data by date
        latest_data = sorted(unique_data, key=lambda x: datetime.fromisoformat(x['date'].replace('Z', '+00:00')), reverse=True)

        # Use the date from the first element of sorted data
        if latest_data:
            first_date = datetime.fromisoformat(latest_data[0]['date'].replace('Z', '+00:00')).strftime('%Y-%m-%d')
        else:
            first_date = datetime.now().strftime('%Y-%m-%d')  # Fallback in case data is empty


        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Save the data to the dated JSON file
        with open(directory+"/data.json", 'wb') as file:
            file.write(orjson.dumps(latest_data))

        print(f"Saved {len(latest_data)} unique datapoints")
    except Exception as e:
        print(f"An error occurred while saving data: {e}")



def get_data():
    try:
        unique_trades = {}
        sources = ['utp_delayed', 'cta_a_delayed', 'cta_b_delayed']
        page_size = 50000
        min_size = 2000
        threshold = 1E5  # Define threshold

        for source in tqdm(sources):
            try:
                next_page = ''  # Reset for each source
                while True:
                    # Build the URL with the current page (if available)
                    url = (
                        f"https://api-v2.intrinio.com/securities/trades?"
                        f"timezone=UTC&source={source}&start_date={start_date}&end_date={end_date}"
                        f"&page_size={page_size}&min_size={min_size}"
                    )
                    if next_page:
                        url += f"&next_page={next_page}"
                    url += f"&darkpool_only=true&api_key={api_key}"

                    response = requests.get(url)
                    if response.status_code == 200:
                        output = response.json()
                        trades = output.get("trades", [])

                        # Process each trade and maintain uniqueness
                        for trade in trades:
                            price = trade.get("price", 0)
                            size = trade.get("size", 0)
                            
                            if price * size > threshold:  # Apply filtering condition
                                unique_key = (
                                    f"{trade.get('symbol')}_"
                                    f"{trade.get('timestamp')}_"
                                    f"{trade.get('price')}_"
                                    f"{trade.get('total_volume')}"
                                )
                                if unique_key not in unique_trades:
                                    unique_trades[unique_key] = trade

                        # Check if there's a next page; if not, exit the loop
                        next_page = output.get("next_page")
                        print(next_page)
                        if not next_page:
                            break
                    else:
                        print(f"Error fetching data from source {source}: {response.status_code} - {response.text}")
                        break
            except Exception as e:
                print(f"Error processing source {source}: {e}")
                pass
            time.sleep(1)

        return list(unique_trades.values())

    except Exception as e:
        print(f"Error fetching data: {e}")
        return []



def get_symbols(db_path, table_name):
    con = sqlite3.connect(db_path)
    cursor = con.cursor()
    cursor.execute(f"SELECT DISTINCT symbol FROM {table_name}")
    symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    return symbols


def main():

    stock_symbols = get_symbols('stocks.db', 'stocks')
    etf_symbols = get_symbols('etf.db', 'etfs')
    total_symbols = stock_symbols + etf_symbols
    print(f"Total Symbols: {len(total_symbols)}")

    data = get_data()
    print(len(data))
    res = []

    for item in data:
        try:
            symbol = item['symbol']
            # Adjust ticker formatting for BRK-A/BRK-B if needed
            ticker = item['symbol']
            if symbol.lower() == 'brk.b':
                ticker = 'BRK-B'
            elif symbol.lower() == 'brk.a':
                ticker = 'BRK-A'
            # Use the datetime 'timestamp' to extract the executed date
            timestamp_dt = datetime.fromisoformat(item['timestamp'])
            executed_date = timestamp_dt.strftime('%Y-%m-%d')
            
            # Create a unique trackingID using hashlib (MD5)
            raw_tracking_string = f"{symbol}_{timestamp_dt.isoformat()}"
            tracking_id = hashlib.md5(raw_tracking_string.encode('utf-8')).hexdigest()[:10]

            if executed_date == today_date:
                sector = stock_screener_data_dict.get(symbol, {}).get('sector', "")
                volume = float(item['total_volume'])
                size = float(item['size'])
                price = round(float(item['price']), 2)
                quote_data = get_quote_data(symbol) or {}
                size_volume_ratio = round((size / volume) * 100, 2) if volume else 0
                size_avg_volume_ratio = round((size / quote_data.get('avgVolume', 1)) * 100, 2)
                
                if ticker in total_symbols:
                    res.append({
                        'ticker': ticker,
                        'date': item['timestamp'],
                        'price': price,
                        'size': size,
                        'volume': volume,
                        'premium': round(size*price,2),  # default to 0 if premium isn't provided
                        'sector': sector,
                        'assetType': 'Stock' if ticker in stock_symbols else 'ETF',
                        'sizeVolRatio': size_volume_ratio,
                        'sizeAvgVolRatio': size_avg_volume_ratio,
                        'trackingID': tracking_id
                    })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Combine new data with existing data
    if res:
        # Save the combined data to a daily file
        save_json(res)


if __name__ == '__main__':
    main()