import asyncio
import orjson
import sqlite3
import os
from datetime import datetime
from GetStartEndDate import GetStartEndDate
from dotenv import load_dotenv
from benzinga import financial_data
from utils.helper import check_market_hours, compute_option_return


# Load environment variables
load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

# Initialize Benzinga API client
fin = financial_data.Benzinga(api_key)
quote_cache = {}


async def get_quote_data(symbol):
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

# Asynchronous wrapper for fin.options_activity
async def fetch_options_activity(page):
    try:
        data = await asyncio.to_thread(fin.options_activity, date_from=start_date, date_to=end_date, page=page, pagesize=1000)
        return orjson.loads(fin.output(data))['option_activity']
    except:
        return []

# Asynchronous function to fetch multiple pages
async def fetch_all_pages(max_pages=50):
    tasks = [fetch_options_activity(page) for page in range(max_pages)]
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]

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
                    'tradeCount': item.get('trade_count', 0),
                    'size': int(float(item['cost_basis'])/(float(item['price'])*100))
                })

                filtered_list.append({key: value for key, value in item.items() if key not in ['description_extended', 'updated']})
        except Exception as e:
            print(f"Error processing item: {e}")
            pass

    return filtered_list

# Main execution flow
async def main():
    # Fetch and process option data
    options_data = await fetch_all_pages()

    # Clean and filter the data
    filtered_data = clean_and_filter_data(options_data)
    
    # Sort the data by time
    sorted_data = sorted(filtered_data, key=lambda x: x['time'], reverse=True)

    # Write the final data to a JSON file
    output_file = "json/options-flow/feed/data.json"
    if len(sorted_data) > 0:
        with open(output_file, 'wb') as file:
            file.write(orjson.dumps(sorted_data))

    print(f"Data successfully written to {output_file}")

# Run the async event loop
if __name__ == "__main__":
    market_open = check_market_hours()
    asyncio.run(main())
    if market_open:
        asyncio.run(main())
    else:
        print('market closed')
