import time
from datetime import datetime
from GetStartEndDate import GetStartEndDate
from tqdm import tqdm
import concurrent.futures

import intrinio_sdk as intrinio
import ujson
import sqlite3
import pytz

from dotenv import load_dotenv
import os
from threading import Lock

ny_tz = pytz.timezone('America/New_York')

load_dotenv()
api_key = os.getenv('INTRINIO_API_KEY')

intrinio.ApiClient().set_api_key(api_key)
intrinio.ApiClient().allow_retries(True)

def save_json(data):
    with open(f"json/dark-pool/flow/data.json", 'w') as file:
        ujson.dump(data, file)

source = 'cta_a_delayed'
start_date = ''
end_date = ''
start_time = ''
end_time = ''
timezone = 'UTC'
page_size = 100
darkpool_only = True
min_size = 100
next_page = ''

api_call_counter = 0
lock = Lock()

def get_data(symbol):
    global api_call_counter
    try:
        response = intrinio.SecurityApi().get_security_trades_by_symbol(
            identifier=symbol, source=source, start_date=start_date, start_time=start_time, 
            end_date=end_date, end_time=end_time, timezone=timezone, page_size=page_size, 
            darkpool_only=darkpool_only, min_size=min_size, next_page=next_page
        )
        data = response.trades
        
        with lock:
            api_call_counter += 1
            if api_call_counter % 1600 == 0:
                #print("API call limit reached, sleeping for 60 seconds...")
                time.sleep(60)

    except:
        pass

def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("SELECT DISTINCT symbol, name FROM stocks")
    stocks = cursor.fetchall()
    con.close()

    symbol_name_map = {row[0]: row[1] for row in stocks}
    stock_symbols = list(symbol_name_map.keys())

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(get_data, stock_symbols), total=len(stock_symbols)))

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}")
