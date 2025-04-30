import time
from benzinga import financial_data
import ujson
import orjson
import numpy as np
import sqlite3
import asyncio
from datetime import datetime, timedelta, date
import concurrent.futures
import os
from GetStartEndDate import GetStartEndDate
from dotenv import load_dotenv
from utils.helper import compute_option_return


# Load API key from .env
load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')
fin = financial_data.Benzinga(api_key)

# Connect to databases and fetch symbols
stock_con = sqlite3.connect('stocks.db')
stock_cursor = stock_con.cursor()
stock_cursor.execute("SELECT DISTINCT symbol FROM stocks")
stock_symbols = [row[0] for row in stock_cursor.fetchall()]

etf_con = sqlite3.connect('etf.db')
etf_cursor = etf_con.cursor()
etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
etf_symbols = [row[0] for row in etf_cursor.fetchall()]

# Close the database connections
stock_con.close()
etf_con.close()



# Define start and end dates for historical data
start_date = datetime.strptime('2023-01-01', '%Y-%m-%d')
end_date = datetime.now()

today = date.today()

# Directory to save the JSON files
output_dir = "json/options-historical-data/flow-data/"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


quote_cache = {}
historical_cache = {}


def get_quote_data(symbol):
    """Get quote data for a symbol from JSON file"""
    if symbol in quote_cache:
        return quote_cache[symbol]
    else:
        try:
            with open(f"json/quote/{symbol}.json") as file:
                quote_data = orjson.loads(file.read())
                quote_cache[symbol] = quote_data  # Cache the loaded data
                return quote_data
        except:
            return None


def get_current_price(item):
    """
    Given an item dict with 'ticker' and 'date_expiration' as 'YYYY-mm-dd' string,
    returns the appropriate price: latest quote if expiration >= today,
    otherwise the historical close price on date_expiration.
    """
    # Parse expiration date
    exp_str = item.get('date_expiration')
    try:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
    except (TypeError, ValueError):
        raise ValueError(f"Invalid date_expiration format: {exp_str}")

    symbol = item['ticker']

    # Future or today → live quote
    if exp_date >= today:
        quote_data = get_quote_data(symbol)
        current_price = quote_data.get('price')
        if current_price is None:
            raise KeyError(f"Price not found in quote data for {symbol}")
        return current_price

    # Past date → historical
    # 1) Load from cache or file
    if symbol in historical_cache:
        historical_list = historical_cache[symbol]
    else:
        hist_path = f"json/historical-price/max/{symbol}.json"
        try:
            with open(hist_path, 'rb') as f:
                historical_list = orjson.loads(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"Historical file not found: {hist_path}")
        historical_cache[symbol] = historical_list

    # 2) Find exact match
    exp_iso = exp_date.isoformat()
    match = next((rec for rec in historical_list if rec.get('time') == exp_iso), None)

    # 3) If no exact match, pick the most recent prior
    if match is None:
        previous = [rec for rec in historical_list if rec.get('time') < exp_iso]
        if not previous:
            raise ValueError(f"No historical data available on or before {exp_iso}")
        match = max(previous, key=lambda rec: rec.get('time'))

    # 4) Extract close price
    current_price = match.get('close')
    
    if current_price is None:
        raise KeyError(f"Close price missing for {symbol} on {exp_iso}")

    return current_price
# Function to fetch options activity data for a specific day
def process_page(page, date):
    try:
        data = fin.options_activity(date_from=date, date_to=date, page=page, pagesize=1000)
        data = ujson.loads(fin.output(data))['option_activity']
        return data
    except Exception as e:
        print(e)
        return []

# Process the data for each day
def process_day(date_str):
    # Check if the file for this date already exists
    file_path = os.path.join(output_dir, f"{date_str}.json")
    #if os.path.exists(file_path):
        #print(f"File for {date_str} already exists. Skipping...")
    #    return

    res_list = []
    max_workers = 6  # Adjust max_workers to control parallelism

    # Fetch pages concurrently for the given day
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {executor.submit(process_page, page, date_str): page for page in range(15)}
        for future in concurrent.futures.as_completed(future_to_page):
            page = future_to_page[future]
            try:
                page_list = future.result()
                res_list += page_list
            except Exception as e:
                print(f"Exception occurred: {e}")
                break

    # Filter and clean the data
    res_list = [{key: value for key, value in item.items() if key not in ['description_extended', 'updated']} for item in res_list]
    filtered_list = []
    
    for item in res_list:
        try:
            if item['underlying_price'] != '':
                ticker = item['ticker']
                if ticker == 'BRK.A':
                    ticker = 'BRK-A'
                elif ticker == 'BRK.B':
                    ticker = 'BRK-B'

                put_call = 'Calls' if item['put_call'] == 'CALL' else 'Puts'

                asset_type = 'stock' if ticker in stock_symbols else ('etf' if ticker in etf_symbols else '')

                item['underlying_type'] = asset_type.lower()
                item['put_call'] = put_call
                item['ticker'] = ticker
                item['price'] = round(float(item['price']), 2)
                item['strike_price'] = round(float(item['strike_price']), 2)
                item['cost_basis'] = round(float(item['cost_basis']), 2)
                item['underlying_price'] = round(float(item['underlying_price']), 2)
                item['option_activity_type'] = item['option_activity_type'].capitalize()
                item['sentiment'] = item['sentiment'].capitalize()
                item['execution_estimate'] = item['execution_estimate'].replace('_', ' ').title()
                item['tradeCount'] = item['trade_count']
                item['size'] = int(float(item['cost_basis'])/(float(item['price'])*100))

                filtered_list.append(item)
        except:
            pass

    # Sort the list by time in reverse order
    filtered_list = sorted(filtered_list, key=lambda x: x['time'], reverse=True)

    for item in filtered_list:
        try:
            current_price = get_current_price(item)
            if current_price:
                roi = compute_option_return(item, current_price)
            item['roi'] = roi
        except:
            item['roi'] = None

    # Save the data to a JSON file named after the date
    if len(filtered_list) > 0:
        with open(file_path, 'w') as file:
            ujson.dump(filtered_list, file)

    #print(f"Data saved for {date_str}")

# Iterate through each weekday from the start_date to today
current_date = start_date
while current_date <= end_date:
    try:
        # Check if it's a weekday (Monday=0, Sunday=6)
        if current_date.weekday() < 5:  # Monday to Friday
            date_str = current_date.strftime("%Y-%m-%d")
            process_day(date_str)
        
        # Move to the next day
        current_date += timedelta(days=1)
    except:
        pass

