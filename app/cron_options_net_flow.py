import sqlite3
from datetime import datetime, timedelta, date
import ujson
import os
import numpy as np
from dotenv import load_dotenv
from benzinga import financial_data
from collections import defaultdict
from tqdm import tqdm

load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

fin = financial_data.Benzinga(api_key)


def save_json(symbol, data):
    with open(f"json/options-net-flow/companies/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

def calculate_moving_average(data, window_size):
    data = np.array(data, dtype=float)
    cumsum = np.cumsum(data)
    moving_avg = (cumsum[window_size - 1:] - np.concatenate(([0], cumsum[:-window_size]))) / window_size
    return moving_avg.tolist()

def calculate_net_flow(data, window_size=100):
    date_data = defaultdict(lambda: {'price': [], 'netCall': 0, 'netPut': 0})

    for item in data:
        date = item['date']
        premium = float(item['cost_basis'])
        #volume = int(item['volume'])

        date_data[date]['price'].append(float(item['underlying_price']))
        #date_data[date]['volume'] += volume

        if item['put_call'] == 'CALL':
            if item['execution_estimate'] == 'AT_ASK':
                date_data[date]['netCall'] += premium
            elif item['execution_estimate'] == 'AT_BID':
                date_data[date]['netCall'] -= premium
        elif item['put_call'] == 'PUT':
            if item['execution_estimate'] == 'AT_ASK':
                date_data[date]['netPut'] -= premium
            elif item['execution_estimate'] == 'AT_BID':
                date_data[date]['netPut'] += premium

    # Calculate average underlying price and format the results
    result = []
    for date, values in date_data.items():
        avg_price = sum(values['price']) / len(values['price'])
        #volume = values['volume']

        # Change sign of volume if netPut > netCall
        #if values['netPut'] > values['netCall']:
        #    volume = -volume

        result.append({
            'date': date,
            'price': round(avg_price, 2),
            'netCall': int(values['netCall']),
            'netPut': int(values['netPut']),
            #'volume': int(volume)
        })
    sorted_data = sorted(result, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
    

    # Calculate moving averages
    netCall_values = [item['netCall'] for item in sorted_data]
    netPut_values = [item['netPut'] for item in sorted_data]
    
    netCall_ma = calculate_moving_average(netCall_values, window_size)
    netPut_ma = calculate_moving_average(netPut_values, window_size)
    
    # Add moving averages to the result and remove None values
    filtered_data = []

    # Add moving averages to the result
    filtered_data = []
    for i, item in enumerate(sorted_data):
        if i >= window_size - 1:
            item['netCall'] = int(netCall_ma[i - window_size + 1])
            item['netPut'] = int(netPut_ma[i - window_size + 1])
            filtered_data.append(item)

    return filtered_data



def get_data(symbol):
    try:
        end_date = date.today()
        start_date = end_date - timedelta(300)

        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')

        res_list = []
        for page in range(0, 100):
            try:
                data = fin.options_activity(company_tickers=symbol, page=page, pagesize=1000, date_from=start_date_str, date_to=end_date_str)
                data = ujson.loads(fin.output(data))['option_activity']
                res_list += data
            except:
                break

        res_filtered = [{key: value for key, value in item.items() if key in ['ticker','date','execution_estimate', 'underlying_price', 'put_call', 'cost_basis', 'volume']} for item in res_list]

        
        #Save raw data for each ticker for options page stack bar chart
        ticker_filtered_data = [entry for entry in res_filtered if entry['ticker'] == symbol]
        if len(ticker_filtered_data) > 100:
            net_flow_data = calculate_net_flow(ticker_filtered_data, window_size=100)
            if len(net_flow_data) > 0:
                save_json(symbol, net_flow_data)
     

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(e)

try:
    stock_con = sqlite3.connect('stocks.db')
    stock_cursor = stock_con.cursor()
    stock_cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap > 500E6 AND symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in stock_cursor.fetchall()]

    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    stock_con.close()
    etf_con.close()
    
    total_symbols = stock_symbols #+ etf_symbols

    for symbol in tqdm(total_symbols):
        get_data(symbol)

except Exception as e:
    print(e)

