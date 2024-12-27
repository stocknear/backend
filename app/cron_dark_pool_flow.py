import os
import pandas as pd
import orjson
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import pytz
import requests  # Add missing import
from dateutil.parser import isoparse
from utils.helper import load_latest_json


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


def save_to_daily_file(data, directory):
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

        json_file_path = os.path.join(directory, f"{first_date}.json")

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Save the data to the dated JSON file
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
    # Directory for saving daily historical flow data
    historical_directory = 'json/dark-pool/historical-flow'

    # Load the latest JSON file from the directory
    existing_data = load_latest_json(historical_directory)
    existing_keys = {item.get('trackingID', None) for item in existing_data}

    # Fetch new data from the API
    data = get_data()

    print(data[0])

    res = []
    for item in data:
        symbol = item['ticker']
        if symbol.lower() == 'brk.b':
            item['ticker'] = 'BRK-B'
        if symbol.lower() == 'brk.a':
            item['ticker'] = 'BRK-A'
        try:
            if item['tracking_id'] not in existing_keys:
                sector = stock_screener_data_dict.get(symbol, {}).get('sector', "")
                volume = float(item['volume'])
                size = float(item['size'])
                quote_data = get_quote_data(symbol) or {}
                size_volume_ratio = round((size / volume) * 100, 2)
                size_avg_volume_ratio = round((size / quote_data.get('avgVolume', 1)) * 100, 2)
                res.append({
                    'ticker': item['ticker'],
                    'date': item['executed_at'],
                    'price': round(float(item['price']), 2),
                    'size': item['size'],
                    'volume': volume,
                    'premium': item['premium'],
                    'sector': sector,
                    'assetType': 'Stock' if symbol in stock_screener_data_dict else 'ETF',
                    'sizeVolRatio': size_volume_ratio,
                    'sizeAvgVolRatio': size_avg_volume_ratio,
                    'trackingID': item['tracking_id']
                })
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Combine new data with existing data
    combined_data = existing_data + res

    # Save the combined data to a daily file
    save_to_daily_file(combined_data, historical_directory)

if __name__ == '__main__':
    main()
