import requests
import orjson
import re
from datetime import datetime,timedelta
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
import time
from tqdm import tqdm
from collections import defaultdict



#today = datetime.today()
#N_days_ago = today - timedelta(days=90)

query_template = """
    SELECT date, close, change_percent
    FROM "{ticker}"
    WHERE date BETWEEN ? AND ?
"""



def save_json(data, symbol):
    directory_path = f"json/options-historical-data/companies"
    os.makedirs(directory_path, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory_path}/{symbol}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))

def prepare_data(data, symbol):
    res_list = []

    data = [entry for entry in data if entry['call_volume'] != 0 or entry['put_volume'] != 0]
    
    data = sorted(data, key=lambda x: x['date'])
    for i in range(1, len(data)):
        try:
            current_open_interest = data[i]['total_open_interest']
            previous_open_interest = data[i-1]['total_open_interest']
            changes_percentage_oi = round((current_open_interest/previous_open_interest -1)*100,2)
            data[i]['changesPercentageOI'] = changes_percentage_oi
            data[i]['changeOI'] = current_open_interest-previous_open_interest
        except:
            data[i]['changesPercentageOI'] = None
            data[i]['changeOI'] = None

    data = sorted(data, key=lambda x: x['date'], reverse=True)
    
    if data:
        save_json(data,symbol)
    '''
    
    start_date_str = data[-1]['date']
    end_date_str = data[0]['date']

    query = query_template.format(ticker=symbol)
    df_price = pd.read_sql_query(query, con if symbol in stocks_symbols else etf_con, params=(start_date_str, end_date_str)).round(2)
    df_price = df_price.rename(columns={"change_percent": "changesPercentage"})

    # Convert the DataFrame to a dictionary for quick lookups by date
    df_change_dict = df_price.set_index('date')['changesPercentage'].to_dict()
    df_close_dict = df_price.set_index('date')['close'].to_dict()

    for item in data:
        try:
            # Round numerical and numerical-string values
            new_item = {
                key: safe_round(value) if isinstance(value, (int, float, str)) else value
                for key, value in item.items()
            }

            # Add parsed fields
            new_item['volume'] = round(new_item['call_volume'] + new_item['put_volume'], 2)
            new_item['putCallRatio'] = round(new_item['put_volume']/new_item['call_volume'],2)
            new_item['avgVolumeRatio'] = round(new_item['volume'] / (round(new_item['avg_30_day_call_volume'] + new_item['avg_30_day_put_volume'], 2)), 2)
            new_item['total_premium'] = round(new_item['call_premium'] + new_item['put_premium'], 2)
            new_item['net_premium'] = round(new_item['net_call_premium'] - new_item['net_put_premium'],2)
            new_item['total_open_interest'] = round(new_item['call_open_interest'] + new_item['put_open_interest'], 2)
            
            bearish_premium = float(item['bearish_premium'])
            bullish_premium = float(item['bullish_premium'])
            neutral_premium = calculate_neutral_premium(item)

            new_item['premium_ratio'] = [
                safe_round(bearish_premium),
                neutral_premium,
                safe_round(bullish_premium)
            ]


            # Add changesPercentage if the date exists in df_change_dict
            if item['date'] in df_change_dict:
                new_item['changesPercentage'] = df_change_dict[item['date']]
            if item['date'] in df_close_dict:
                new_item['price'] = df_close_dict[item['date']]

            res_list.append(new_item)
        except:
            pass
    
    res_list = sorted(res_list, key=lambda x: x['date'])
    for i in range(1, len(res_list)):
        try:
            current_open_interest = res_list[i]['total_open_interest']
            previous_open_interest = res_list[i-1]['total_open_interest']
            changes_percentage_oi = round((current_open_interest/previous_open_interest -1)*100,2)
            res_list[i]['changesPercentageOI'] = changes_percentage_oi
        except:
            res_list[i]['changesPercentageOI'] = None

    res_list = sorted(res_list, key=lambda x: x['date'],reverse=True)

    if res_list:
        save_json(res_list, symbol)
    '''


def get_contracts_from_directory(directory: str):
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        
        # Get all tickers from filenames
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



def get_contracts_from_directory(directory):
    """Retrieve a list of contract files from a directory."""
    return [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.json')]


def aggregate_data_by_date(total_symbols):
    data_by_date = defaultdict(lambda: {
        "date": "",  # Add date field to the dictionary
        "call_volume": 0,
        "put_volume": 0,
        "call_open_interest": 0,
        "put_open_interest": 0,
        "call_premium": 0,
        "call_net_premium": 0,
        "put_premium": 0,
        "put_net_premium": 0,
    })
    
    for symbol in tqdm(total_symbols):
        try:
            contract_dir = f"json/all-options-contracts/{symbol}"
            if not os.path.exists(contract_dir):
                print(f"Directory does not exist: {contract_dir}")
                continue
            
            contract_list = get_contracts_from_directory(contract_dir)
            
            for item in tqdm(contract_list, desc=f"Processing {symbol} contracts", leave=False):
                try:
                    file_path = os.path.join(contract_dir, f"{item}.json")
                    with open(file_path, "r") as file:
                        data = orjson.loads(file.read())
                    option_type = data.get('optionType', None)
                    if option_type not in ['call', 'put']:
                        continue
                    for entry in data.get('history', []):
                        date = entry.get('date')
                        volume = entry.get('volume',0)
                        open_interest = entry.get('open_interest',0)
                        total_premium = entry.get('total_premium',0)


                        if volume is None:
                            volume = 0
                        if open_interest is None:
                            open_interest = 0
                        if total_premium is None:
                            total_premium = 0
                      

                        if date:
                            data_by_date[date]["date"] = date  # Store the date in the dictionary
                            if option_type == 'call':
                                if volume is not None:
                                    data_by_date[date]["call_volume"] += int(volume)
                                if open_interest is not None:
                                    data_by_date[date]["call_open_interest"] += int(open_interest)
                                if total_premium is not None:
                                    data_by_date[date]["call_premium"] += int(total_premium)

                            elif option_type == 'put':
                                if volume is not None:
                                    data_by_date[date]["put_volume"] += int(volume)
                                if open_interest is not None:
                                    data_by_date[date]["put_open_interest"] += int(open_interest)
                                if total_premium is not None:
                                    data_by_date[date]["put_premium"] += int(total_premium)
                            try:
                                data_by_date[date]["putCallRatio"] = round(data_by_date[date]["put_volume"]/data_by_date[date]["call_volume"],2)
                            except:
                                data_by_date[date]["putCallRatio"] = None

                            data_by_date[date]["volume"] = data_by_date[date]["call_volume"] + data_by_date[date]["put_volume"]
                            data_by_date[date]["total_open_interest"] = data_by_date[date]["call_open_interest"] + data_by_date[date]["put_open_interest"]

            
                except Exception as e:
                    print(f"Error processing contract {item} for {symbol}: {e}")
                    continue
        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
            continue
            
        # Convert to list of dictionaries and sort by date
        data = list(data_by_date.values())

        data = prepare_data(data,symbol)



if __name__ == '__main__':
    
    # Connect to the databases
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    #cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND marketCap > 1E9")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    #etf_cursor.execute("SELECT DISTINCT symbol FROM etfs WHERE marketCap > 1E9")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]
    total_symbols = stocks_symbols + etf_symbols


    total_symbols = ['AA']
    data = aggregate_data_by_date(total_symbols)
    
    con.close()
    etf_con.close()