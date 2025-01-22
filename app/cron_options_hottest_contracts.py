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

def get_contracts_from_directory(symbol):
    directory = f"json/all-options-contracts/{symbol}/"
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        # Get all tickers from filenames
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    
    except:
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


def process_contract(item, symbol):
    """Process a single contract and return its processed data."""
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
    except Exception as e:
        print(e)
        return None

def prepare_data(highest_volume_list, highest_oi_list, symbol):
    """Prepare and save options data for highest volume and open interest contracts."""
    # Process both lists in parallel using list comprehension
    highest_volume = [
        result for item in highest_volume_list 
        if (result := process_contract(item, symbol)) is not None
    ]
    
    highest_oi = [
        result for item in highest_oi_list 
        if (result := process_contract(item, symbol)) is not None
    ]
    
    res_dict = {'volume': highest_volume, 'openInterest': highest_oi}
    
    if res_dict:
        save_json(res_dict, symbol, "json/hottest-contracts/companies")
    
    return res_dict



def get_hottest_contracts(base_dir="json/all-options-contracts"):
    """
    Ranks option contracts by highest peak volume and highest latest open interest,
    and returns the top 10 contracts for each.
    """
    top_by_volume = []  # Will store (peak_volume, contract_info) tuples
    top_by_open_interest = []  # Will store (latest_open_interest, contract_info) tuples
    
    def find_peak_volume(history_data):
        """Find the highest single-day volume from history data, handling None values"""
        peak_volume = 0
        peak_date = None
        
        for entry in history_data:
            volume = entry['volume']
            if volume is not None and volume > peak_volume:
                peak_volume = volume
                peak_date = entry['date']
                
        return peak_volume
    
    def find_latest_open_interest(history_data):
        """Find the most recent non-None open interest value from history data"""
        latest_open_interest = 0
        
        # Sort history data by date to ensure we get the latest value
        sorted_history = sorted(history_data, key=lambda x: x['date'], reverse=True)
        
        # Find the first non-None open interest value
        for entry in sorted_history:
            open_interest = entry.get('open_interest')
            if open_interest is not None:
                latest_open_interest = open_interest
                break
                
        return latest_open_interest
    
    def process_symbol(symbol):
        symbol_dir = os.path.join(base_dir, symbol)
        if not os.path.exists(symbol_dir):
            return
        
        # Process each contract file for this symbol
        for contract_file in os.listdir(symbol_dir):
            if not contract_file.endswith('.json'):
                continue
                
            try:
                file_path = os.path.join(symbol_dir, contract_file)
                with open(file_path, 'rb') as f:
                    data = orjson.loads(f.read())
                    
                if 'history' not in data:
                    continue
                
                # Find peak volume
                peak_volume = find_peak_volume(data['history'])
                
                # Find latest open interest
                latest_open_interest = find_latest_open_interest(data['history'])
                
                contract_info = {'contract_id': os.path.splitext(contract_file)[0], 'peak_volume': peak_volume}
                
                # Add to top by volume list if it qualifies
                top_by_volume.append((peak_volume, contract_info))
                top_by_volume.sort(key=lambda x: x[0], reverse=True)  # Sort by peak volume
                
                # Keep only top 10 by volume
                if len(top_by_volume) > 10:
                    top_by_volume.pop()
                
                # Add to top by latest open interest list if it qualifies
                top_by_open_interest.append((latest_open_interest, contract_info))
                top_by_open_interest.sort(key=lambda x: x[0], reverse=True)  # Sort by latest open interest
                
                # Keep only top 10 by open interest
                if len(top_by_open_interest) > 10:
                    top_by_open_interest.pop()
                    
            except Exception as e:
                print(f"Error processing {contract_file}: {e}")
    
    # Process each symbol directory
    total_symbols = ['AA']
    for symbol in tqdm(total_symbols):
        try:
            process_symbol(symbol)
        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
    
        top_by_volume_contracts = [contract_info for _, contract_info in top_by_volume]
        top_by_open_interest_contracts = [contract_info for _, contract_info in top_by_open_interest]

        prepare_data(top_by_volume_contracts, top_by_open_interest_contracts, symbol)
        
# Example usage
if __name__ == "__main__":
    get_hottest_contracts()