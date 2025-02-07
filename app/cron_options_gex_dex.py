import requests
import orjson
import re
from datetime import datetime
from dotenv import load_dotenv
import os
import sqlite3
import time
from tqdm import tqdm
from collections import defaultdict
import numpy as np

load_dotenv()

today = datetime.today().date()

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


def save_json(data, symbol, directory_path):
    os.makedirs(directory_path, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory_path}/{symbol}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))


def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value


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


def get_expiration_date(option_symbol):
    # Define regex pattern to match the symbol structure
    match = re.match(r"([A-Z]+)(\d{6})([CP])(\d+)", option_symbol)
    if not match:
        raise ValueError(f"Invalid option_symbol format: {option_symbol}")
    
    ticker, expiration, option_type, strike_price = match.groups()
    
    # Convert expiration to datetime
    date_expiration = datetime.strptime(expiration, "%y%m%d").date()
    return date_expiration

def aggregate_data_by_strike(symbol):
    data_by_date = defaultdict(lambda: defaultdict(lambda: {
        "strike": 0,
        "call_gex": 0.0,
        "put_gex": 0.0,
        "call_dex": 0.0,
        "put_dex": 0.0,
    }))
    
    contract_dir = f"json/all-options-contracts/{symbol}"
    contract_list = get_contracts_from_directory(contract_dir)
    
    # Only consider contracts that haven't expired
    contract_list = [item for item in contract_list if get_expiration_date(item) >= today]

    # Load historical price data
    with open(f"json/historical-price/max/{symbol}.json", "r") as file:
        price_list = orjson.loads(file.read())

    if contract_list:
        for item in contract_list:
            try:
                file_path = os.path.join(contract_dir, f"{item}.json")
                with open(file_path, "r") as file:
                    data = orjson.loads(file.read())

                option_type = data.get("optionType")
                if option_type not in ["call", "put"]:
                    continue

                strike = float(data.get("strike", 0))

                # Get only the last element from 'history' (latest entry)
                if not data.get("history"):
                    continue

                latest_entry = data["history"][-1]  # Latest data point
                date = latest_entry.get("date")

                open_interest = latest_entry.get("open_interest", 0) or 0
                gamma = latest_entry.get("gamma", 0) or 0
                delta = latest_entry.get("delta", 0) or 0

                # Find the matching spot price for the date
                matching_price = next((p for p in price_list if p.get("time") == date), None)
                spot_price = matching_price["close"] if matching_price else 0

                gex = open_interest * gamma * spot_price
                dex = open_interest * delta * spot_price

                # Aggregate data by date and strike
                daily_data = data_by_date[date][strike]
                daily_data["strike"] = strike

                if option_type == "call":
                    daily_data["call_gex"] += round(gex, 2)
                    daily_data["call_dex"] += round(dex, 2)
                elif option_type == "put":
                    daily_data["put_gex"] += round(gex, 2)
                    daily_data["put_dex"] += round(dex, 2)

            except:
                continue

        # Convert defaultdict to sorted list format
        final_output = []
        for date, strikes in sorted(data_by_date.items()):
            for strike, data in sorted(strikes.items(), key=lambda x: x[0]):
                final_output.append(data)

        return final_output

    return []


def aggregate_data_by_expiration(symbol):
    data_by_date = defaultdict(lambda: defaultdict(lambda: {
        "expiration": "",
        "call_gex": 0.0,
        "put_gex": 0.0,
        "call_dex": 0.0,
        "put_dex": 0.0,
    }))
    
    contract_dir = f"json/all-options-contracts/{symbol}"
    contract_list = get_contracts_from_directory(contract_dir)
    
    # Only consider contracts that haven't expired
    contract_list = [item for item in contract_list if get_expiration_date(item) >= today]

    # Load historical price data
    with open(f"json/historical-price/max/{symbol}.json", "r") as file:
        price_list = orjson.loads(file.read())

    if contract_list:
        for item in contract_list:
            try:
                file_path = os.path.join(contract_dir, f"{item}.json")
                with open(file_path, "r") as file:
                    data = orjson.loads(file.read())

                option_type = data.get("optionType")
                if option_type not in ["call", "put"]:
                    continue

                expiration_date = data.get("expiration")
                if not expiration_date:
                    continue  # Skip if expiration date is missing

                # Get only the last element from 'history' (latest entry)
                if not data.get("history"):
                    continue

                latest_entry = data["history"][-1]  # Latest data point
                date = latest_entry.get("date")

                open_interest = latest_entry.get("open_interest", 0) or 0
                gamma = latest_entry.get("gamma", 0) or 0
                delta = latest_entry.get("delta", 0) or 0

                # Find the matching spot price for the date
                matching_price = next((p for p in price_list if p.get("time") == date), None)
                spot_price = matching_price["close"] if matching_price else 0

                gex = open_interest * gamma * spot_price
                dex = open_interest * delta * spot_price

                # Aggregate data by expiration date
                daily_data = data_by_date[date][expiration_date]
                daily_data["expiry"] = expiration_date

                if option_type == "call":
                    daily_data["call_gex"] += round(gex, 2)
                    daily_data["call_dex"] += round(dex, 2)
                elif option_type == "put":
                    daily_data["put_gex"] += round(gex, 2)
                    daily_data["put_dex"] += round(dex, 2)

            except:
                continue

        # Convert defaultdict to sorted list format and filter out empty GEX entries
        final_output = []
        for date, expirations in sorted(data_by_date.items()):
            for expiration_date, data in sorted(expirations.items(), key=lambda x: x[0]):
                final_output.append(data)

        return final_output

    return []

def get_overview_data():
    directory_path = "json/gex-dex/overview"
    total_symbols = stocks_symbols+etf_symbols

    counter = 0
    
    #Test mode
    #total_symbols = ['TSLA']
    for symbol in tqdm(total_symbols):
        try:
            with open(f"json/options-historical-data/companies/{symbol}.json","r") as file:
                data = orjson.loads(file.read())

            filtered_data = [{k: d[k] for k in ['date','call_gex', 'call_dex', 'put_gex', 'put_dex']} for d in data]
            
            for item in filtered_data:
                try:
                    item['netGex'] = item['call_gex'] + item['put_gex']
                    item['netDex'] = item['call_dex'] + item['put_dex']
                except:
                    pass
            
            if filtered_data:
                filtered_data = sorted(filtered_data, key=lambda x: x['date'], reverse=True)
                save_json(filtered_data, symbol, directory_path)

            #prepare_data(data, symbol, directory_path)
            
        except:
            pass

def get_strike_data():
    directory_path = "json/gex-dex/strike/"
    total_symbols = stocks_symbols+etf_symbols

    #Test mode
    #total_symbols = ['TSLA']
    for symbol in tqdm(total_symbols):
        try:
            data = aggregate_data_by_strike(symbol)
            for key_element in ['gex','dex']:
                val_sums = [item[f"call_{key_element}"] + item[f"put_{key_element}"] for item in data]
                threshold = np.percentile(val_sums, 90)
                filtered_data = [item for item in data if (item[f"call_{key_element}"] + item[f"put_{key_element}"]) >= threshold]
                filtered_data = sorted(filtered_data, key=lambda x: x['strike'], reverse=True)
                if filtered_data:
                    save_json(filtered_data, symbol, directory_path+key_element)
        except Exception as e:
            print(e)

def get_expiry_data():
    directory_path = "json/gex-dex/expiry/"
    total_symbols = stocks_symbols+etf_symbols

    #Test mode
    #total_symbols = ['TSLA']
    for symbol in tqdm(total_symbols):
        try:
            data = aggregate_data_by_expiration(symbol)
            for key_element in ['gex','dex']:
                val_sums = [item[f"call_{key_element}"] + item[f"put_{key_element}"] for item in data]
                threshold = np.percentile(val_sums, 90)
                filtered_data = [item for item in data if (item[f"call_{key_element}"] + item[f"put_{key_element}"]) >= threshold]
                filtered_data = sorted(filtered_data, key=lambda x: x['expiry'], reverse=True)
                if filtered_data:
                    save_json(filtered_data, symbol, directory_path+key_element)

        except Exception as e:
            print(e)

if __name__ == '__main__':
    get_overview_data()
    get_strike_data()
    get_expiry_data()
    