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


def aggregate_data_by_date(symbol):
    data_by_date = defaultdict(lambda: {
        "date": "",
        "call_volume": 0,
        "put_volume": 0,
        "call_open_interest": 0,
        "put_open_interest": 0,
        "call_premium": 0,
        "put_premium": 0,
        "call_gex": 0,
        "put_gex": 0,
        "call_dex": 0,
        "put_dex": 0,
        "iv": 0.0,  # Sum of implied volatilities
        "iv_count": 0,  # Count of entries for IV
    })
    
    # Calculate cutoff date (1 year ago)
    today = datetime.today().date()
    one_year_ago = today - timedelta(days=365)
    one_year_ago_str = one_year_ago.strftime('%Y-%m-%d')
    
    contract_dir = f"json/all-options-contracts/{symbol}"
    contract_list = get_contracts_from_directory(contract_dir)

    with open(f"json/historical-price/max/{symbol}.json","r") as file:
        price_list = orjson.loads(file.read())

    if len(contract_list) > 0:
        for item in tqdm(contract_list):
            try:
                file_path = os.path.join(contract_dir, f"{item}.json")
                with open(file_path, "r") as file:
                    data = orjson.loads(file.read())
                
                option_type = data.get('optionType', None)
                if option_type not in ['call', 'put']:
                    continue
                
                for entry in data.get('history', []):
                    date = entry.get('date')

                    # Skip entries older than one year
                    if date < one_year_ago_str:
                        continue
                    
                    volume = entry.get('volume', 0) or 0
                    open_interest = entry.get('open_interest', 0) or 0
                    total_premium = entry.get('total_premium', 0) or 0
                    implied_volatility = entry.get('implied_volatility', 0) or 0
                    gamma = entry.get('gamma',0) or 0
                    delta = entry.get('delta',0) or 0

                    # Find the matching date in price_list
                    matching_price = next((p for p in price_list if p.get('time') == date), 0)

                    if matching_price:
                        spot_price = matching_price['close']
                    else:
                        spot_price = 0  # Or some default value

                    gex = open_interest * gamma * spot_price
                    dex = open_interest * delta * spot_price


                    daily_data = data_by_date[date]
                    daily_data["date"] = date
                    
                    if option_type == 'call':
                        daily_data["call_volume"] += int(volume)
                        daily_data["call_open_interest"] += int(open_interest)
                        daily_data["call_premium"] += int(total_premium)
                        daily_data["call_gex"] += round(gex,2)
                        daily_data["call_dex"] += round(dex,2)
                    elif option_type == 'put':
                        daily_data["put_volume"] += int(volume)
                        daily_data["put_open_interest"] += int(open_interest)
                        daily_data["put_premium"] += int(total_premium)
                        daily_data["put_gex"] += round(gex,2)
                        daily_data["put_dex"] += round(dex,2)
                    
                    # Aggregate IV for both calls and puts
                    daily_data["iv"] += round(implied_volatility, 2)
                    daily_data["iv_count"] += 1
                    
                    # Calculate put/call ratio
                    try:
                        daily_data["putCallRatio"] = round(daily_data["put_volume"] / daily_data["call_volume"], 2)
                    except ZeroDivisionError:
                        daily_data["putCallRatio"] = None
            
            except Exception as e:
                print(f"Error processing {item}: {e}")
                continue
        
        # Convert to list and calculate average IV
        data = []
        for date, daily in data_by_date.items():
            if daily['iv_count'] > 0:
                daily['iv'] = round(daily['iv'] / daily['iv_count'], 2)
            else:
                daily['iv'] = None
            data.append(daily)
        
        # Sort and calculate IV Rank
        data = sorted(data, key=lambda x: x['date'])
        data = calculate_iv_rank_for_all(data)
        data = sorted(data, key=lambda x: x['date'], reverse=True)

        return data
    else:
        return []

def calculate_iv_rank_for_all(data):
    if not data:
        return []
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Check if 'iv' exists and filter out entries without IV
    if 'iv' not in df.columns or df['iv'].isnull().all():
        for entry in data:
            entry['iv_rank'] = None
        return data
    
    # Convert date to datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    # Calculate rolling 365-day min and max for IV
    df.set_index('date', inplace=True)
    rolling_min = df['iv'].rolling('365D', min_periods=1).min()
    rolling_max = df['iv'].rolling('365D', min_periods=1).max()
    
    # Merge back into DataFrame
    df['rolling_min'] = rolling_min
    df['rolling_max'] = rolling_max
    
    # Calculate IV Rank
    df['iv_rank'] = ((df['iv'] - df['rolling_min']) / (df['rolling_max'] - df['rolling_min'])) * 100
    df['iv_rank'] = df['iv_rank'].round(2)
    
    # Handle cases where max == min
    df.loc[df['rolling_max'] == df['rolling_min'], 'iv_rank'] = 100.0
    
    # Replace NaN with None
    df['iv_rank'] = df['iv_rank'].where(pd.notnull(df['iv_rank']), None)
    
    # Drop temporary columns
    df.drop(['rolling_min', 'rolling_max'], axis=1, inplace=True)
    
    # Convert back to list of dicts
    df.reset_index(inplace=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    result = df.to_dict('records')
    
    # Sort in reverse chronological order
    result = sorted(result, key=lambda x: x['date'], reverse=True)
    
    return result


def prepare_data(data, symbol):
    
    data = [entry for entry in data if entry['call_volume'] != 0 or entry['put_volume'] != 0]
    
    start_date_str = data[-1]['date']
    end_date_str = data[0]['date']

    query = query_template.format(ticker=symbol)
    if symbol in stocks_symbols:
        query_con = con
    elif symbol in etf_symbols:
        query_con = etf_con
    else:
        query_con = index_con

    df_price = pd.read_sql_query(query, query_con, params=(start_date_str, end_date_str)).round(2)
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
        except:
            pass
    

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





# Connect to the databases
con = sqlite3.connect('stocks.db')
etf_con = sqlite3.connect('etf.db')
index_con = sqlite3.connect("index.db")

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

index_symbols =["^SPX","^VIX"]

total_symbols = stocks_symbols + etf_symbols + index_symbols


for symbol in tqdm(total_symbols):
    try:
        data = aggregate_data_by_date(symbol)
        data = prepare_data(data, symbol)
    except:
        pass

con.close()
etf_con.close()
index_con.close()