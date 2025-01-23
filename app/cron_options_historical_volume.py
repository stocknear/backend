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

def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value


def calculate_iv_rank_for_all(data):
    # Extract all IV values
    iv_values = [entry['iv'] for entry in data if 'iv' in entry]

    if not iv_values:
        return None  # No IV data available

    # Compute highest and lowest IV
    highest_iv = max(iv_values)
    lowest_iv = min(iv_values)

    # Calculate IV Rank for each entry
    for entry in data:
        if 'iv' in entry:
            iv = entry['iv']
            if highest_iv == lowest_iv:
                entry['iv_rank'] = 100.0  # If all IVs are the same, rank is 100%
            else:
                entry['iv_rank'] = round(((iv - lowest_iv) / (highest_iv - lowest_iv)) * 100,2)
        else:
            entry['iv_rank'] = None  # Handle missing IV

    return data


def prepare_data(data, symbol):
    
    data = [entry for entry in data if entry['call_volume'] != 0 or entry['put_volume'] != 0]
    
    start_date_str = data[-1]['date']
    end_date_str = data[0]['date']

    query = query_template.format(ticker=symbol)
    df_price = pd.read_sql_query(query, con if symbol in stocks_symbols else etf_con, params=(start_date_str, end_date_str)).round(2)
    df_price = df_price.rename(columns={"change_percent": "changesPercentage"})

    # Convert the DataFrame to a dictionary for quick lookups by date
    df_change_dict = df_price.set_index('date')['changesPercentage'].to_dict()
    df_close_dict = df_price.set_index('date')['close'].to_dict()

    res_list = []

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
            #new_item['avgVolumeRatio'] = round(new_item['volume'] / (round(new_item['avg_30_day_call_volume'] + new_item['avg_30_day_put_volume'], 2)), 2)
            new_item['total_premium'] = round(new_item['call_premium'] + new_item['put_premium'], 2)
            #new_item['net_premium'] = round(new_item['net_call_premium'] - new_item['net_put_premium'],2)
            new_item['total_open_interest'] = round(new_item['call_open_interest'] + new_item['put_open_interest'], 2)

            
            #bearish_premium = float(item['bearish_premium'])
            #bullish_premium = float(item['bullish_premium'])
            #neutral_premium = calculate_neutral_premium(item)
            '''
            new_item['premium_ratio'] = [
                safe_round(bearish_premium),
                neutral_premium,
                safe_round(bullish_premium)
            ]
            '''

            # Add changesPercentage if the date exists in df_change_dict
            if item['date'] in df_change_dict:
                new_item['changesPercentage'] = float(df_change_dict[item['date']])
            else:
                new_item['changesPercentage'] = None

            if item['date'] in df_close_dict:
                new_item['price'] = float(df_close_dict[item['date']])
            else:
                new_item['price'] = None

            res_list.append(new_item)
        except Exception as e:
            print(e)
    

    res_list = sorted(res_list, key=lambda x: x['date'])

    for i in range(1, len(res_list)):
        try:
            current_open_interest = res_list[i]['total_open_interest']
            previous_open_interest = res_list[i-1]['total_open_interest']
            changes_percentage_oi = round((current_open_interest/previous_open_interest -1)*100,2)
            res_list[i]['changesPercentageOI'] = changes_percentage_oi
            res_list[i]['changeOI'] = current_open_interest-previous_open_interest
        except:
            res_list[i]['changesPercentageOI'] = None
            res_list[i]['changeOI'] = None

    res_list = sorted(res_list, key=lambda x: x['date'],reverse=True)

    if res_list:
        save_json(res_list, symbol)


def get_contracts_from_directory(directory: str):
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            return []
        
        # Get all tickers from filenames
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    
    except Exception as e:
        print(e)
        return []


def aggregate_data_by_date(symbol):
    data_by_date = defaultdict(lambda: {
        "date": "",
        "call_volume": 0,
        "put_volume": 0,
        "call_open_interest": 0,
        "put_open_interest": 0,
        "call_premium": 0,
        "call_net_premium": 0,
        "put_premium": 0,
        "put_net_premium": 0,
        "iv": 0,  # Sum of implied volatilities
        "iv_count": 0,  # Count of entries for IV
    })
    
    contract_dir = f"json/all-options-contracts/{symbol}"
    contract_list = get_contracts_from_directory(contract_dir)

    if len(contract_list) > 0:
    
        for item in contract_list:
            try:
                file_path = os.path.join(contract_dir, f"{item}.json")
                with open(file_path, "r") as file:
                    data = orjson.loads(file.read())
                
                option_type = data.get('optionType', None)
                if option_type not in ['call', 'put']:
                    continue
                
                for entry in data.get('history', []):
                    date = entry.get('date')
                    volume = entry.get('volume', 0) or 0
                    open_interest = entry.get('open_interest', 0) or 0
                    total_premium = entry.get('total_premium', 0) or 0
                    implied_volatility = entry.get('implied_volatility', 0) or 0
                    
                    if date:
                        daily_data = data_by_date[date]
                        daily_data["date"] = date
                        
                        if option_type == 'call':
                            daily_data["call_volume"] += int(volume)
                            daily_data["call_open_interest"] += int(open_interest)
                            daily_data["call_premium"] += int(total_premium)
                        elif option_type == 'put':
                            daily_data["put_volume"] += int(volume)
                            daily_data["put_open_interest"] += int(open_interest)
                            daily_data["put_premium"] += int(total_premium)
                            daily_data["iv"] += round(implied_volatility, 2)
                            daily_data["iv_count"] += 1
                        
                        try:
                            daily_data["putCallRatio"] = round(daily_data["put_volume"] / daily_data["call_volume"], 2)
                        except ZeroDivisionError:
                            daily_data["putCallRatio"] = None
        
            except:
                pass
      
        # Convert to list of dictionaries and sort by date
        data = list(data_by_date.values())
        for daily_data in data:
            try:
                if daily_data["iv_count"] > 0:
                    daily_data["iv"] = round(daily_data["iv"] / daily_data["iv_count"], 2)
                else:
                    daily_data["iv"] = None  # Or set it to 0 if you prefer
            except:
                daily_data["iv"] = None
        
        data = sorted(data, key=lambda x: x['date'], reverse=True)
        data = calculate_iv_rank_for_all(data)
        
        return data
    else:
        return []




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


for symbol in tqdm(total_symbols):
    try:
        data = aggregate_data_by_date(symbol)
        data = prepare_data(data, symbol)
    except:
        pass

con.close()
etf_con.close()