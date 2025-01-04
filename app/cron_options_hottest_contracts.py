import requests
import orjson
import re
from datetime import datetime
from dotenv import load_dotenv
import os
import sqlite3
import time
from tqdm import tqdm

load_dotenv()

api_key = os.getenv('UNUSUAL_WHALES_API_KEY')
headers = {"Accept": "application/json, text/plain", "Authorization": api_key}

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

con.close()
etf_con.close()


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

directory_path = "json/hottest-contracts/companies"
total_symbols = get_tickers_from_directory(directory_path)

if len(total_symbols) < 100:
    total_symbols = stocks_symbols+etf_symbols

def save_json(data, symbol,directory="json/hottest-contracts/companies"):
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory}/{symbol}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))


def parse_option_symbol(option_symbol):
    # Define regex pattern to match the symbol structure
    match = re.match(r"([A-Z]+)(\d{6})([CP])(\d+)", option_symbol)
    if not match:
        raise ValueError(f"Invalid option_symbol format: {option_symbol}")
    
    ticker, expiration, option_type, strike_price = match.groups()
    
    # Convert expiration to datetime
    date_expiration = datetime.strptime(expiration, "%y%m%d").date()
    
    # Convert strike price to float
    strike_price = int(strike_price) / 1000

    return date_expiration, option_type, strike_price

def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value


def prepare_data(data, symbol):

    res_list = []
    for item in data:
        try:
            if float(item['volume']) > 0:
                # Parse option_symbol
                date_expiration, option_type, strike_price = parse_option_symbol(item['option_symbol'])

                # Round numerical and numerical-string values
                new_item = {
                    key: safe_round(value) if isinstance(value, (int, float, str)) else value
                    for key, value in item.items()
                }

                # Add parsed fields
                new_item['date_expiration'] = date_expiration
                new_item['option_type'] = option_type
                new_item['strike_price'] = strike_price

                # Calculate open_interest_change
                new_item['open_interest_change'] = safe_round(
                    new_item.get('open_interest', 0) - new_item.get('prev_oi', 0)
                )

                res_list.append(new_item)
        except:
            pass

    if res_list:
        highest_volume = sorted(res_list, key=lambda x: x['volume'], reverse=True)[:10]
        highest_open_interest = sorted(res_list, key=lambda x: x['open_interest'], reverse=True)[:10]
        res_dict = {'volume': highest_volume, 'openInterest': highest_open_interest}
        save_json(res_dict, symbol,"json/hottest-contracts/companies")


def get_hottest_contracts():
    counter = 0
    for symbol in tqdm(total_symbols):
        try:
            
            url = f"https://api.unusualwhales.com/api/stock/{symbol}/option-contracts"
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()['data']
                prepare_data(data, symbol)
                counter +=1
            # If 50 chunks have been processed, sleep for 60 seconds
            if counter == 100:
                print("Sleeping...")
                time.sleep(30)  # Sleep for 60 seconds
                counter = 0
            
        except Exception as e:
            print(f"Error for {symbol}:{e}")


def get_single_contract_historical_data(contract_id):
    keys_to_remove = {'high_price', 'low_price', 'iv_low', 'iv_high', 'last_tape_time'}

    url = f"https://api.unusualwhales.com/api/option-contract/{contract_id}/historic"
    response = requests.get(url, headers=headers)
    data = response.json()['chains']
    data = sorted(data, key=lambda x: datetime.strptime(x.get('date', ''), '%Y-%m-%d'))
    res_list = []
    for i, item in enumerate(data):
        new_item = {
            key: safe_round(value) if isinstance(value, (int, float, str)) else value
            for key, value in item.items()
        }
        
        # Compute open interest change and percent if not the first item
        if i > 0:
            previous_open_interest = safe_round(data[i-1].get('open_interest', 0))
            open_interest = safe_round(item.get('open_interest', 0))

            if previous_open_interest > 0:
                new_item['open_interest_change'] = safe_round(open_interest - previous_open_interest)
                new_item['open_interest_change_percent'] = safe_round((open_interest / previous_open_interest - 1) * 100)
            else:
                new_item['open_interest_change'] = 0
                new_item['open_interest_change_percent'] = 0

        res_list.append(new_item)

    if res_list:
        res_list = [{key: value for key, value in item.items() if key not in keys_to_remove} for item in res_list]
        res_list = sorted(res_list, key=lambda x: datetime.strptime(x.get('date', ''), '%Y-%m-%d'), reverse=True)

        save_json(res_list, contract_id,"json/hottest-contracts/contracts")


if __name__ == '__main__':
    get_hottest_contracts()

    '''
    total_symbols = get_tickers_from_directory(directory_path)

    contract_id_set = set()  # Use a set to ensure uniqueness
    for symbol in total_symbols:
        try:
            with open(f"json/hottest-contracts/companies/{symbol}.json", "r") as file:
                data = orjson.loads(file.read())
                for item in data:
                    try:
                        contract_id_set.add(item['option_symbol'])  # Add to the set
                    except KeyError:
                        pass  # Handle missing 'option_symbol' keys gracefully
        except FileNotFoundError:
            pass  # Handle missing files gracefully

    # Convert the set to a list if needed
    contract_id_list = list(contract_id_set)
    
    print(len(contract_id_list))
    print(contract_id_list[0])

    get_single_contract_historical_data('GME250117C00125000')
    '''