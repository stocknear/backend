
from datetime import timedelta
from GetStartEndDate import GetStartEndDate
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import intrinio_sdk as intrinio
import ujson
import sqlite3

from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv('INTRINIO_API_KEY')

intrinio.ApiClient().set_api_key(api_key)
intrinio.ApiClient().allow_retries(True)

def save_json(data):
    with open(f"json/dark-pool/flow/data.json", 'w') as file:
        ujson.dump(data, file)


identifier = 'GME'
source = 'cta_a_delayed'
start_date, end_date = GetStartEndDate().run()
start_date = start_date.strftime("%Y-%m-%d")
end_date = end_date.strftime("%Y-%m-%d")
start_time = ''
end_time = ''
timezone = 'UTC'
page_size = 1000
darkpool_only = True
min_size = 100
count = 0


def get_data():
	data = []
	count = 0
	while True:
		if count == 0:
			next_page = ''
		try:
			response = intrinio.SecurityApi().get_security_trades_by_symbol(identifier, source, start_date=start_date, start_time=start_time, end_date=end_date, end_time=end_time, timezone=timezone, page_size=page_size, darkpool_only=darkpool_only, min_size=min_size, next_page=next_page)
			
			filtered_entries = [
                entry.__dict__ for entry in response.trades 
                if int(entry._price * entry._total_volume) >= 2E9
            ]

			data.extend(filtered_entries)
			next_page = response.next_page
			
			if not next_page:
				break
			count +=1
			print(f'Current length {len(data)}')

		except Exception as e:
			print(e)
			break

	return data

def run():
	con = sqlite3.connect('stocks.db')
	cursor = con.cursor()
	cursor.execute("SELECT DISTINCT symbol, name FROM stocks")
	stocks = cursor.fetchall()
	con.close()

	symbol_name_map = {row[0]: row[1] for row in stocks}
	stock_symbols = list(symbol_name_map.keys())
	data = get_data()
	print(data)

	# Filter the data
	filtered_data = [entry for entry in data if entry['_symbol'] in stock_symbols]
	res = [
	    {
	        'symbol': entry['_symbol'],
	        'name': symbol_name_map[entry['_symbol']],
	        'date': (entry['_timestamp']-timedelta(hours=4)).isoformat(),
	        'price': entry['_price'],
	        'volume': entry['_total_volume'],
	        'size': entry['_size']
	    }
	    for entry in filtered_data
	]


	if len(res) > 0:
		save_json(res)


if __name__ == "__main__":
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(run)
    try:
        # Wait for the result with a timeout of 300 seconds (5 minutes)
        future.result(timeout=1000)
    except TimeoutError:
        print("The operation timed out.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        executor.shutdown()