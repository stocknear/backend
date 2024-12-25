import os
import pandas as pd
import orjson
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import pytz
import requests  # Add missing import
from dateutil.parser import isoparse

load_dotenv()
api_key = os.getenv('UNUSUAL_WHALES_API_KEY')

querystring = {"limit": "200"}
url = "https://api.unusualwhales.com/api/darkpool/recent"
headers = {
    "Accept": "application/json, text/plain",
    "Authorization": api_key
}

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

def load_json(file_path):
    """Load existing JSON data from file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return orjson.loads(file.read())
        except (ValueError, IOError):
            print(f"Warning: Could not read or parse {file_path}. Starting with an empty list.")
    return []

def save_latest_ratings(combined_data, json_file_path, limit=2000):
    try:
        # Create a set to track unique entries based on a combination of 'ticker' and 'date'
        seen = set()
        unique_data = []

        for item in combined_data:
            identifier = f"{item['trackingID']}"
            if identifier not in seen:
                seen.add(identifier)
                unique_data.append(item)

        # Sort the data by date
        sorted_data = sorted(unique_data, key=lambda x: datetime.fromisoformat(x['date'].replace('Z', '+00:00')), reverse=True)

        # Keep only the latest `limit` entries
        latest_data = sorted_data[:limit]

        # Save the trimmed and deduplicated data to the JSON file
        with open(json_file_path, 'wb') as file:
            file.write(orjson.dumps(latest_data))

        print(f"Saved {len(latest_data)} unique and latest ratings to {json_file_path}.")
    except Exception as e:
        print(f"An error occurred while saving data: {e}")



def get_data():
    try:
        response = requests.get(url, headers=headers, params=querystring)
        return response.json().get('data', [])
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def main():
    # Load environment variables
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]
    total_symbols = stock_symbols + etf_symbols
    con.close()
    etf_con.close()

    json_file_path = 'json/dark-pool/feed/data.json'
    existing_data = load_json(json_file_path)
    # Transform existing data into a set of unique trackingIDs
    existing_keys = {item.get('trackingID',None) for item in existing_data}
    data = get_data()

    # Prepare results with only new data
    res = []
    for item in data:
        symbol = item['ticker']
        if symbol.lower() == 'brk.b':
            item['ticker'] = 'BRK-B'
            symbol = item['ticker']
        if symbol.lower() == 'brk.a':
            item['ticker'] = 'BRK-A'
            symbol = item['ticker']
        if symbol in total_symbols:
            quote_data = get_quote_data(symbol)
            if symbol in stock_symbols:
                asset_type = 'Stock'
            else:
                asset_type = 'ETF'

            try:
                # Check if the data is already in the file
                if item['tracking_id'] not in existing_keys:
                    try:
                        sector = stock_screener_data_dict[symbol].get('sector', None)
                    except:
                        sector = ""

                    volume = float(item['volume'])
                    size = float(item['size'])

                    size_volume_ratio = round((size / volume) * 100, 2)
                    size_avg_volume_ratio = round((size / quote_data.get('avgVolume', 1)) * 100, 2)
                    res.append({
                        'ticker': item['ticker'],
                        'date': item['executed_at'],
                        'price': round(float(item['price']),2),
                        'size': item['size'],
                        'volume': volume,
                        'premium': item['premium'],
                        'sector': sector,
                        'assetType': asset_type,
                        'sizeVolRatio': size_volume_ratio,
                        'sizeAvgVolRatio': size_avg_volume_ratio,
                        'trackingID': item['tracking_id']
                    })
            except Exception as e:
                print(f"Error processing {symbol}: {e}")

    # Append new data to existing data and combine
    combined_data = existing_data + res
    # Save the updated data
    save_latest_ratings(combined_data, json_file_path)


if __name__ == '__main__':
    main()
