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




def save_json(data, symbol, directory_path):
    os.makedirs(directory_path, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory_path}/{symbol}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))


def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value


def prepare_data(data, symbol, directory_path, sort_by = "date"):
    data = [{k: v for k, v in item.items() if "charm" not in k and "vanna" not in k} for item in data]
    res_list = []
    for item in data:
        try:
            new_item = {
                key: safe_round(value) if isinstance(value, (int, float, str)) else value
                for key, value in item.items()
            }

            res_list.append(new_item)
        except:
            pass

    if res_list:
        res_list = sorted(res_list, key=lambda x: x[sort_by], reverse=True)
        save_json(res_list, symbol, directory_path)


def get_overview_data():
    print("Starting to download overview data...")
    directory_path = "json/gex-dex/overview"
    total_symbols = get_tickers_from_directory(directory_path)
    if len(total_symbols) < 100:
        total_symbols = stocks_symbols+etf_symbols

    counter = 0
    total_symbols = ['GME']
    for symbol in tqdm(total_symbols):
        try:
            url = f"https://api.unusualwhales.com/api/stock/{symbol}/greek-exposure"
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()['data']
                prepare_data(data, symbol, directory_path)
            
            counter +=1
            
            # If 50 chunks have been processed, sleep for 60 seconds
            if counter == 260:
                print("Sleeping...")
                time.sleep(60)
                counter = 0
            
        except Exception as e:
            print(f"Error for {symbol}:{e}")



def get_strike_data():
    print("Starting to download strike data...")
    directory_path = "json/gex-dex/strike"
    total_symbols = get_tickers_from_directory(directory_path)
    if len(total_symbols) < 100:
        total_symbols = stocks_symbols+etf_symbols

    counter = 0
    total_symbols = ['GME']
    for symbol in tqdm(total_symbols):
        try:
            url = f"https://api.unusualwhales.com/api/stock/{symbol}/greek-exposure/strike"
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()['data']
                prepare_data(data, symbol, directory_path, sort_by = 'strike')
            
            counter +=1
            
            # If 50 chunks have been processed, sleep for 60 seconds
            if counter == 260:
                print("Sleeping...")
                time.sleep(60)
                counter = 0
            
        except Exception as e:
            print(f"Error for {symbol}:{e}")

def get_expiry_data():
    print("Starting to download expiry data...")
    directory_path = "json/gex-dex/expiry"
    total_symbols = get_tickers_from_directory(directory_path)
    if len(total_symbols) < 100:
        total_symbols = stocks_symbols+etf_symbols

    counter = 0
    total_symbols = ['GME']
    for symbol in tqdm(total_symbols):
        try:
            url = f"https://api.unusualwhales.com/api/stock/{symbol}/greek-exposure/expiry"
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()['data']
                prepare_data(data, symbol, directory_path)
            
            counter +=1
            
            # If 50 chunks have been processed, sleep for 60 seconds
            if counter == 260:
                print("Sleeping...")
                time.sleep(60)
                counter = 0
            
        except Exception as e:
            print(f"Error for {symbol}:{e}")


if __name__ == '__main__':
    get_overview_data()
    get_strike_data()
    get_expiry_data()

