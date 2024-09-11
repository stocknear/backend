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


def calculate_net_flow(data):
    date_data = defaultdict(lambda: {'price': [], 'netCall': 0, 'netPut': 0})
    for item in data:
        date_str = item['date']
        time_str = item['time']
        datetime_str = f"{date_str} {time_str}"
        
        # Parse the combined date and time into a datetime object
        date_time = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        
        try:
            premium = float(item['cost_basis'])
            date_data[date_time]['price'].append(round(float(item['underlying_price']), 2))
            if item['put_call'] == 'CALL':
                if item['execution_estimate'] == 'AT_ASK':
                    date_data[date_time]['netCall'] += premium
                elif item['execution_estimate'] == 'AT_BID':
                    date_data[date_time]['netCall'] -= premium
            elif item['put_call'] == 'PUT':
                if item['execution_estimate'] == 'AT_ASK':
                    date_data[date_time]['netPut'] -= premium
                elif item['execution_estimate'] == 'AT_BID':
                    date_data[date_time]['netPut'] += premium
        except:
            pass

    # Calculate average underlying price and format the results
    result = []
    for date_time, values in date_data.items():
        result.append({
            'date': date_time.strftime('%Y-%m-%d %H:%M:%S'),
            'price': sum(values['price']) / len(values['price']) if values['price'] else 0,
            'netCall': int(values['netCall']),
            'netPut': int(values['netPut']),
        })

    sorted_data = sorted(result, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d %H:%M:%S'))

    # Compute 30-minute interval averages
    interval_data = defaultdict(lambda: {'price': [], 'netCall': [], 'netPut': []})
    for item in sorted_data:
        date_time = datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S')
        interval_start = date_time.replace(minute=date_time.minute // 120 * 120, second=0)
        
        interval_data[interval_start]['price'].append(item['price'])
        interval_data[interval_start]['netCall'].append(item['netCall'])
        interval_data[interval_start]['netPut'].append(item['netPut'])

    # Calculate averages for each 30-minute interval
    averaged_data = []
    for interval_start, values in interval_data.items():
        if values['price']:
            averaged_data.append({
                'date': interval_start.strftime('%Y-%m-%d %H:%M:%S'),
                #'price': sum(values['price']) / len(values['price']) ,
                'netCall': sum(values['netCall']) if values['netCall'] else 0,
                'netPut': sum(values['netPut']) if values['netPut'] else 0,
            })

    # Sort the averaged data by interval start time
    averaged_data.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d %H:%M:%S'))

    return averaged_data

def get_data(symbol):
    try:
        end_date = date.today()
        start_date = end_date - timedelta(10)

        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')

        res_list = []
        for page in range(0, 1000):
            try:
                data = fin.options_activity(company_tickers=symbol, page=page, pagesize=1000, date_from=start_date_str, date_to=end_date_str)
                data = ujson.loads(fin.output(data))['option_activity']
                res_list += data
            except:
                break

        res_filtered = [{key: value for key, value in item.items() if key in ['ticker','time','date','execution_estimate', 'underlying_price', 'put_call', 'cost_basis']} for item in res_list]
        
        #Save raw data for each ticker for options page stack bar chart
        ticker_filtered_data = [entry for entry in res_filtered if entry['ticker'] == symbol]
        if len(ticker_filtered_data) > 100:
            net_flow_data = calculate_net_flow(ticker_filtered_data)
            if len(net_flow_data) > 0:
                save_json(symbol, net_flow_data)
     

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(e)

try:
    stock_con = sqlite3.connect('stocks.db')
    stock_cursor = stock_con.cursor()
    stock_cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >500E6 AND symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in stock_cursor.fetchall()]

    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    stock_con.close()
    etf_con.close()
    
    total_symbols = stock_symbols + etf_symbols

    for symbol in tqdm(total_symbols):
        get_data(symbol)

except Exception as e:
    print(e)

