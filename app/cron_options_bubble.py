import sqlite3
from datetime import datetime, timedelta, date
import ujson
import asyncio
import os
from dotenv import load_dotenv
from benzinga import financial_data
import time


load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

fin = financial_data.Benzinga(api_key)


def calculate_dte(date_expiration):
    expiration_date = datetime.strptime(date_expiration, "%Y-%m-%d")
    return (expiration_date - datetime.today()).days

def calculate_avg_dte(data):
    active_options = [entry for entry in data if calculate_dte(entry['date_expiration']) >= 0]
    
    if active_options:
        total_dte = sum(entry['dte'] for entry in active_options)
        return int(total_dte / len(active_options))
    else:
        return 0

def calculate_put_call_volumes(data):
    put_volume = sum(int(entry['volume']) for entry in data if entry['put_call'] == 'PUT')
    call_volume = sum(int(entry['volume']) for entry in data if entry['put_call'] == 'CALL')
    return put_volume, call_volume

def options_bubble_data(chunk):
    try:
        company_tickers = ','.join(chunk)
        end_date = date.today()
        start_date = end_date - timedelta(90)

        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')

        res_list = []
        for page in range(0, 100):
            try:
                data = fin.options_activity(company_tickers=company_tickers, page=page, pagesize=500, date_from=start_date_str, date_to=end_date_str)
                data = ujson.loads(fin.output(data))['option_activity']
                res_list += data
            except:
                break

        res_filtered = [{key: value for key, value in item.items() if key in ['ticker','date', 'date_expiration', 'put_call', 'volume', 'open_interest']} for item in res_list]

        for option_type in ['CALL', 'PUT']:
            for item in res_filtered:
                if item['put_call'].upper() == option_type:
                    item['dte'] = calculate_dte(item['date_expiration'])
                    if item['ticker'] in ['BRK.A', 'BRK.B']:
                        item['ticker'] = f"BRK-{item['ticker'][-1]}"


        #Save raw data for each ticker for options page stack bar chart
        for ticker in chunk:
            ticker_filtered_data = [entry for entry in res_filtered if entry['ticker'] == ticker]
            if len(ticker_filtered_data) != 0:
                #sum up calls and puts for each day for the plot
                summed_data = {}
                for entry in ticker_filtered_data:
                    volume = int(entry['volume'])
                    open_interest = int(entry['open_interest'])
                    put_call = entry['put_call']
                    
                    if entry['date'] not in summed_data:
                        summed_data[entry['date']] = {'CALL': {'volume': 0, 'open_interest': 0}, 'PUT': {'volume': 0, 'open_interest': 0}}
                    
                    summed_data[entry['date']][put_call]['volume'] += volume
                    summed_data[entry['date']][put_call]['open_interest'] += open_interest

                result_list = [{'date': date, 'CALL': summed_data[date]['CALL'], 'PUT': summed_data[date]['PUT']} for date in summed_data]
                #reverse the list
                result_list = result_list[::-1]
                with open(f"json/options-flow/company/{ticker}.json", 'w') as file:
                    ujson.dump(result_list, file)

        #Save bubble data for each ticker for overview page
        for ticker in chunk:

            bubble_data = {}
            for time_period, days in {'oneDay': 1, 'oneWeek': 7, 'oneMonth': 30, 'threeMonth': 90}.items():
                start_date = end_date - timedelta(days=days) #end_date is today

                filtered_data = [item for item in res_filtered if start_date <= datetime.strptime(item['date'], '%Y-%m-%d').date() <= end_date]


                ticker_filtered_data = [entry for entry in filtered_data if entry['ticker'] == ticker]
                put_volume, call_volume = calculate_put_call_volumes(ticker_filtered_data)
                avg_dte = calculate_avg_dte(ticker_filtered_data)
                bubble_data[time_period] = {'putVolume': put_volume, 'callVolume': call_volume, 'avgDTE': avg_dte}

            if all(all(value == 0 for value in data.values()) for data in bubble_data.values()):
                bubble_data = {}
                #don't save the json
            else:
                with open(f"json/options-bubble/{ticker}.json", 'w') as file:
                    ujson.dump(bubble_data, file)
    

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(e)

try:
    stock_con = sqlite3.connect('stocks.db')
    stock_cursor = stock_con.cursor()
    stock_cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in stock_cursor.fetchall()]

    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    stock_con.close()
    etf_con.close()
    
    total_symbols = stock_symbols + etf_symbols
    total_symbols = [item.replace("BRK-B", "BRK.B") for item in total_symbols]

    chunk_size = len(total_symbols) // 20  # Divide the list into N chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
    
    for chunk in chunks:
        options_bubble_data(chunk)

except Exception as e:
    print(e)

