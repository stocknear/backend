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

load_dotenv()

api_key = os.getenv('UNUSUAL_WHALES_API_KEY')

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

#today = datetime.today()
#N_days_ago = today - timedelta(days=90)

query_template = """
    SELECT date, close, change_percent
    FROM "{ticker}"
    WHERE date BETWEEN ? AND ?
"""

def get_tickers_from_directory(directory: str):
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        
        # Get all tickers from filenames
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

directory_path = "json/options-historical-data/companies"
total_symbols = get_tickers_from_directory(directory_path)

if len(total_symbols) < 100:
    total_symbols = stocks_symbols+etf_symbols

print(len(total_symbols))


def save_json(data, symbol):
    os.makedirs(directory_path, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory_path}/{symbol}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))


def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value


def calculate_neutral_premium(data_item):
    """Calculate the neutral premium for a data item."""
    call_premium = float(data_item['call_premium'])
    put_premium = float(data_item['put_premium'])
    bearish_premium = float(data_item['bearish_premium'])
    bullish_premium = float(data_item['bullish_premium'])
    
    total_premiums = bearish_premium + bullish_premium
    observed_premiums = call_premium + put_premium
    neutral_premium = observed_premiums - total_premiums
    
    return safe_round(neutral_premium)


def prepare_data(data, symbol):
    res_list = []
    #data = [entry for entry in data if datetime.strptime(entry['date'], "%Y-%m-%d") >= N_days_ago]
    
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



querystring = {"limit":"300"}
headers = {
    "Accept": "application/json, text/plain",
    "Authorization": api_key
}

total_symbols = ['NVDA']

counter = 0
for symbol in tqdm(total_symbols):
    try:
        
        url = f"https://api.unusualwhales.com/api/stock/{symbol}/options-volume"
        
        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code == 200:
            data = response.json()['data']
            prepare_data(data, symbol)
        
        counter +=1
        # If 50 chunks have been processed, sleep for 60 seconds
        if counter == 260:
            print("Sleeping...")
            time.sleep(60)  # Sleep for 60 seconds
            counter = 0
        
    except Exception as e:
        print(f"Error for {symbol}:{e}")


con.close()
etf_con.close()