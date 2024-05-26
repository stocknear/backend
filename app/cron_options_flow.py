import time 
from benzinga import financial_data
import ujson
import numpy as np
import sqlite3
import asyncio
from datetime import datetime, timedelta
import concurrent.futures
from GetStartEndDate import GetStartEndDate

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

fin = financial_data.Benzinga(api_key)

stock_con = sqlite3.connect('stocks.db')
stock_cursor = stock_con.cursor()
stock_cursor.execute("SELECT DISTINCT symbol FROM stocks")
stock_symbols = [row[0] for row in stock_cursor.fetchall()]

etf_con = sqlite3.connect('etf.db')
etf_cursor = etf_con.cursor()
etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
etf_symbols = [row[0] for row in etf_cursor.fetchall()]

start_date_1d, end_date_1d = GetStartEndDate().run()
start_date = start_date_1d.strftime("%Y-%m-%d")
end_date = end_date_1d.strftime("%Y-%m-%d")

#print(start_date,end_date)

def process_page(page):
    try:
        data = fin.options_activity(date_from=start_date, date_to=end_date, page=page, pagesize=1000)
        data = ujson.loads(fin.output(data))['option_activity']
        filtered_data = [{key: value for key, value in item.items() if key in ['ticker','time', 'id','sentiment','underlying_price', 'cost_basis', 'underlying_price','option_activity_type','date', 'date_expiration', 'open_interest','price', 'put_call','strike_price', 'volume']} for item in data]
        time.sleep(1)
        page_list = []
        for item in filtered_data:
            if item['underlying_price'] != '':
                ticker = item['ticker']
                if ticker == 'BRK.A':
                    ticker = 'BRK-A'
                elif ticker == 'BRK.B':
                    ticker = 'BRK-B'

                put_call = 'Calls' if item['put_call'] == 'CALL' else 'Puts'

                asset_type = 'stock' if ticker in stock_symbols else ('etf' if ticker in etf_symbols else '')

                item['assetType'] = asset_type
                item['put_call'] = put_call
                item['ticker'] = ticker
                item['price'] = round(float(item['price']), 2)
                item['strike_price'] = round(float(item['strike_price']), 2)
                item['cost_basis'] = round(float(item['cost_basis']), 2)
                item['underlying_price'] = round(float(item['underlying_price']), 2)
                item['type'] = item['option_activity_type'].capitalize()
                item['sentiment'] = item['sentiment'].capitalize()

                page_list.append(item)

        return page_list
    except Exception as e:
        print(f"Error processing page {page}: {e}")
        return []


# Assuming fin, stock_symbols, and etf_symbols are defined elsewhere
res_list = []

# Adjust max_workers to control the degree of parallelism
max_workers = 6

# Fetch pages concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_page = {executor.submit(process_page, page): page for page in range(20)}
    for future in concurrent.futures.as_completed(future_to_page):
        page = future_to_page[future]
        try:
            page_list = future.result()
            res_list += page_list
        except Exception as e:
            print(f"Exception occurred: {e}")
            break

# res_list now contains the aggregated results from all pages
#print(res_list)
def custom_key(item):
    return item['time']

res_list = sorted(res_list, key=custom_key, reverse =True)

with open(f"json/options-flow/feed/data.json", 'w') as file:
    ujson.dump(res_list, file)

stock_con.close()
etf_con.close()