import requests
import orjson
import re
from datetime import datetime
from dotenv import load_dotenv
import os
import sqlite3
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio
import aiohttp

today = datetime.today().date()

load_dotenv()



def get_tickers_from_directory(directory: str):
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_contracts_from_directory(symbol):
    directory = f"json/all-options-contracts/{symbol}/"
    try:
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    except:
        return []

directory_path = "json/options-historical-data/companies"
total_symbols = get_tickers_from_directory(directory_path)


def save_json(data, symbol, directory="json/hottest-contracts/companies"):
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))



def parse_option_symbol(option_symbol):
    match = re.match(r"(\^?[A-Z]+)(\d{6})([CP])(\d+)", option_symbol)
    if not match:
        raise ValueError(f"Invalid option_symbol format: {option_symbol}")
    ticker, expiration, option_type, strike_price = match.groups()
    date_expiration = datetime.strptime(expiration, "%y%m%d").date()
    strike_price = int(strike_price) / 1000
    return date_expiration, option_type, strike_price

def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value

def process_contract(item, symbol):
    try:
        with open(f"json/all-options-contracts/{symbol}/{item['contract_id']}.json", "r") as file:
            data = orjson.loads(file.read())
        history = data['history']
        latest_entry = history[-1]
        return {
            'option_symbol': item['contract_id'],
            'date_expiration': data['expiration'],
            'option_type': data['optionType'].replace('call', 'C').replace('put', 'P'),
            'strike_price': data['strike'],
            'volume': item['peak_volume'],
            'open_interest': latest_entry['open_interest'],
            'changeOI': latest_entry['open_interest'] - history[-2]['open_interest'],
            'total_premium': int(latest_entry['mark'] * latest_entry['open_interest'] * 100),
            'iv': round(latest_entry['implied_volatility'] * 100, 2),
            'last': latest_entry['close'],
            'low': latest_entry['low'],
            'high': latest_entry['high']
        }
    except:
        return None

def prepare_data(highest_volume_list, highest_oi_list, symbol):
    highest_volume = []
    for item in highest_volume_list:
        result = process_contract(item, symbol)
        if result is not None:
            highest_volume.append(result)
    highest_oi = []
    for item in highest_oi_list:
        result = process_contract(item, symbol)
        if result is not None:
            highest_oi.append(result)
    res_dict = {'volume': highest_volume, 'openInterest': highest_oi}
    if len(highest_volume) > 0 or len(highest_oi) > 0:
        save_json(res_dict, symbol, "json/hottest-contracts/companies")
    return res_dict

def find_peak_volume(history_data):
    peak_volume = 0
    for entry in history_data:
        volume = entry['volume']
        if volume is not None and volume > peak_volume:
            peak_volume = volume
    return peak_volume

def find_latest_open_interest(history_data):
    sorted_history = sorted(history_data, key=lambda x: x['date'], reverse=True)
    for entry in sorted_history:
        open_interest = entry.get('open_interest')
        if open_interest is not None:
            return open_interest
    return 0

def get_hottest_contracts(base_dir="json/all-options-contracts"):
    for symbol in tqdm(total_symbols):
        symbol_dir = os.path.join(base_dir, symbol)
        if not os.path.exists(symbol_dir):
            continue

        top_by_volume = []
        top_by_open_interest = []

        for contract_file in os.listdir(symbol_dir):
            if not contract_file.endswith('.json'):
                continue
            try:
                file_path = os.path.join(symbol_dir, contract_file)
                with open(file_path, 'rb') as f:
                    data = orjson.loads(f.read())

                #only consider contracts that didn't expire yet
                expiration_date, _, _ = parse_option_symbol(contract_file.replace(".json",""))
                    
                # Check if the contract is expired
                if expiration_date < today:
                    continue

                if 'history' not in data:
                    continue

                peak_volume = find_peak_volume(data['history'])
                latest_open_interest = find_latest_open_interest(data['history'])
                contract_info = {
                    'contract_id': os.path.splitext(contract_file)[0],
                    'peak_volume': peak_volume
                }

                top_by_volume.append((peak_volume, contract_info))
                top_by_open_interest.append((latest_open_interest, contract_info))
            except:
                pass

        # Sort and select top 10 for volume
        top_by_volume.sort(key=lambda x: x[0], reverse=True)
        top_volume_contracts = [contract_info for (pv, contract_info) in top_by_volume[:20]]

        # Sort and select top 10 for open interest
        top_by_open_interest.sort(key=lambda x: x[0], reverse=True)
        top_oi_contracts = [contract_info for (loi, contract_info) in top_by_open_interest[:20]]

        prepare_data(top_volume_contracts, top_oi_contracts, symbol)

if __name__ == "__main__":
    get_hottest_contracts()