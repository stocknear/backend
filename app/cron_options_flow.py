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

# Load environment variables
load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

# Initialize Benzinga API client
fin = financial_data.Benzinga(api_key)

# Database connection and fetching stock/ETF symbols
def get_symbols(db_path, table_name):
    con = sqlite3.connect(db_path)
    cursor = con.cursor()
    cursor.execute(f"SELECT DISTINCT symbol FROM {table_name}")
    symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    return symbols

stock_symbols = get_symbols('stocks.db', 'stocks')
etf_symbols = get_symbols('etf.db', 'etfs')

# Get start and end dates
start_date_1d, end_date_1d = GetStartEndDate().run()
start_date = start_date_1d.strftime("%Y-%m-%d")
end_date = end_date_1d.strftime("%Y-%m-%d")

# Process a page of option activity
def process_page(page):
    try:
        data = fin.options_activity(date_from=start_date, date_to=end_date, page=page, pagesize=1000)
        data = ujson.loads(fin.output(data))['option_activity']
        return data
    except Exception as e:
        print(f"Error on page {page}: {e}")
        return []

# Fetch and process pages concurrently
def fetch_options_data(max_pages=130, max_workers=6):
    res_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {executor.submit(process_page, page): page for page in range(max_pages)}
        for future in concurrent.futures.as_completed(future_to_page):
            page = future_to_page[future]
            try:
                page_data = future.result()
                res_list.extend(page_data)
            except Exception as e:
                print(f"Exception on page {page}: {e}")
                break
    return res_list

# Clean and filter the fetched data
def clean_and_filter_data(res_list):
    filtered_list = []
    for item in res_list:
        try:
            if item.get('underlying_price', ''):
                ticker = item['ticker']
                ticker = 'BRK-A' if ticker == 'BRK.A' else 'BRK-B' if ticker == 'BRK.B' else ticker

                asset_type = 'stock' if ticker in stock_symbols else 'etf' if ticker in etf_symbols else ''
                if not asset_type:
                    continue

                # Standardize item fields
                item.update({
                    'underlying_type': asset_type.lower(),
                    'put_call': 'Calls' if item['put_call'] == 'CALL' else 'Puts',
                    'ticker': ticker,
                    'price': round(float(item['price']), 2),
                    'strike_price': round(float(item['strike_price']), 2),
                    'cost_basis': round(float(item['cost_basis']), 2),
                    'underlying_price': round(float(item['underlying_price']), 2),
                    'option_activity_type': item['option_activity_type'].capitalize(),
                    'sentiment': item['sentiment'].capitalize(),
                    'execution_estimate': item['execution_estimate'].replace('_', ' ').title(),
                    'tradeCount': item.get('trade_count', 0)
                })

                filtered_list.append({key: value for key, value in item.items() if key not in ['description_extended', 'updated']})
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    return filtered_list

# Main execution flow
if __name__ == "__main__":
    # Fetch and process option data
    options_data = fetch_options_data()

    # Clean and filter the data
    filtered_data = clean_and_filter_data(options_data)

    # Sort the data by time
    sorted_data = sorted(filtered_data, key=lambda x: x['time'], reverse=True)

    # Write the final data to a JSON file
    output_file = "json/options-flow/feed/data.json"
    with open(output_file, 'w') as file:
        ujson.dump(sorted_data, file)

    print(f"Data successfully written to {output_file}")
