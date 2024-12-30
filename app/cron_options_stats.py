import requests
import orjson
from dotenv import load_dotenv
import os
import sqlite3
import time
load_dotenv()

api_key = os.getenv('UNUSUAL_WHALES_API_KEY')

# Connect to the databases
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

# Combine the lists of stock and ETF symbols
total_symbols = stocks_symbols + etf_symbols



def save_json(data, symbol):
    directory = "json/options-stats/companies"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory}/{symbol}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))

def safe_round(value):
    """Attempt to convert a value to float and round it. Return the original value if not possible."""
    try:
        return round(float(value), 2)
    except (ValueError, TypeError):
        return value

def calculate_neutral_premium(data_item):
    """Calculate the neutral premium for a data item."""
    call_premium = float(data_item['call_premium'])
    put_premium = float(data_item['put_premium'])
    bearish_premium = float(data_item['bearish_premium'])
    bullish_premium = float(data_item['bullish_premium'])
    
    total_premiums = bearish_premium + bullish_premium
    observed_premiums = call_premium + put_premium
    neutral_premium = observed_premiums - total_premiums
    
    return safe_round(neutral_premium)


def prepare_data(data):
    for item in data:
        symbol = item['ticker']
        bearish_premium = float(item['bearish_premium'])
        bullish_premium = float(item['bullish_premium'])
        neutral_premium = calculate_neutral_premium(item)

        new_item = {
            key: safe_round(value)
            for key, value in item.items()
            if key != 'in_out_flow'
        }
    

        new_item['premium_ratio'] = [
            safe_round(bearish_premium),
            neutral_premium,
            safe_round(bullish_premium)
        ]
        try:
            new_item['open_interest_change'] = new_item['total_open_interest'] - (new_item.get('prev_call_oi',0) + new_item.get('prev_put_oi',0))
        except:
            new_item['open_interest_change'] = None

        if len(new_item) > 0:
            save_json(new_item, symbol)

def chunk_symbols(symbols, chunk_size=50):
    for i in range(0, len(symbols), chunk_size):
        yield symbols[i:i + chunk_size]


chunks = chunk_symbols(total_symbols)
chunk_counter = 0  # To keep track of how many chunks have been processed

for chunk in chunks:
    try:
        chunk_str = ",".join(chunk)
        print(chunk_str)
        
        url = "https://api.unusualwhales.com/api/screener/stocks"
        querystring = {"ticker": chunk_str}
        
        headers = {
            "Accept": "application/json, text/plain",
            "Authorization": api_key
        }
        
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            data = response.json()['data']
            prepare_data(data)
            print(f"Chunk processed. Number of results: {len(data)}")
        else:
            print(f"Error fetching data for chunk {chunk_str}: {response.status_code}")
        
        # Increment the chunk counter
        chunk_counter += 1
        
        # If 50 chunks have been processed, sleep for 60 seconds
        if chunk_counter == 50:
            print("Processed 50 chunks. Sleeping for 60 seconds...")
            time.sleep(60)  # Sleep for 60 seconds
            chunk_counter = 0  # Reset the chunk counter after sleep
        
    except Exception as e:
        print(f"Error processing chunk {chunk_str}: {e}")
