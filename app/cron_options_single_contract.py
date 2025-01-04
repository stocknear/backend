import requests
import orjson
import re
from datetime import datetime
from dotenv import load_dotenv
import os
import time
import asyncio
import aiohttp
from tqdm import tqdm

load_dotenv()

api_key = os.getenv('UNUSUAL_WHALES_API_KEY')
headers = {"Accept": "application/json, text/plain", "Authorization": api_key}
keys_to_remove = {'high_price', 'low_price', 'iv_low', 'iv_high', 'last_tape_time'}

def save_json(data, filename, directory):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{filename}.json")
    with open(filepath, 'wb') as file:
        file.write(orjson.dumps(data))

def get_tickers_from_directory(directory: str):
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value


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
    directory_path = "json/hottest-contracts/companies"

    total_symbols = get_tickers_from_directory(directory_path)

    contract_id_set = set()  # Use a set to ensure uniqueness
    for symbol in total_symbols:
        try:
            with open(f"json/hottest-contracts/companies/{symbol}.json", "r") as file:
                data = orjson.loads(file.read())
                volume_list = data.get('volume',[])
                open_interest_list = data.get('openInterest',[])
                if len(volume_list) > 0:
                    for item in volume_list:
                        try:
                            contract_id_set.add(item['option_symbol'])  # Add to the set
                        except KeyError:
                            pass  # Handle missing 'option_symbol' keys gracefully

                if len(open_interest_list) > 0:
                    for item in open_interest_list:
                        try:
                            contract_id_set.add(item['option_symbol'])  # Add to the set
                        except KeyError:
                            pass  # Handle missing 'option_symbol' keys gracefully
        except FileNotFoundError:
            pass  # Handle missing files gracefully

    # Convert the set to a list if needed
    contract_id_list = list(contract_id_set)
    
    print("Number of contract chains:", len(contract_id_list))
    
    counter = 0
    for item in tqdm(contract_id_list):
        try:
            get_single_contract_historical_data(item)
        except:
            pass
        # If 50 chunks have been processed, sleep for 60 seconds
        counter +=1
        if counter == 260:
            print("Sleeping...")
            time.sleep(60)
            counter = 0