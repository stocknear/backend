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

directory_path = "json/options-contract-lookup/companies"

def get_contracts_from_directory(symbol):
    directory = f"json/all-options-contracts/{symbol}/"
    try:
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    except Exception as e:
        return []

def save_json(data, symbol):
    os.makedirs(directory_path, exist_ok=True)
    with open(f"{directory_path}/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))

def parse_option_symbol(option_symbol):
    match = re.match(r"(\^?[A-Z]+)(\d{6})([CP])(\d+)", option_symbol)
    if not match:
        raise ValueError(f"Invalid option_symbol format: {option_symbol}")
    ticker, expiration, option_type, strike_price = match.groups()
    date_expiration = datetime.strptime(expiration, "%y%m%d").date()
    strike_price = int(strike_price) / 1000
    return date_expiration, option_type, strike_price

if __name__ == "__main__":

    # Connect to databases and fetch symbols
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')
    index_con = sqlite3.connect('index.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    index_cursor = index_con.cursor()
    index_cursor.execute("PRAGMA journal_mode = wal")
    index_cursor.execute("SELECT DISTINCT symbol FROM indices")
    index_symbols = [row[0] for row in index_cursor.fetchall()]


    con.close()
    etf_con.close()
    index_con.close()


    total_symbols = stocks_symbols + etf_symbols + index_symbols

    # For testing
    #total_symbols = ['NVDA']


    for symbol in tqdm(total_symbols):
        res = {"Call": {}, "Put": {}}

        options_contracts = get_contracts_from_directory(symbol)
        if options_contracts:
            for option_symbol in options_contracts:
                try:
                    date_expiration, option_type, strike_price = parse_option_symbol(option_symbol)
                except ValueError as e:
                    print(e)
                    continue  # Skip symbols with incorrect format

                # Only include options whose expiration date is today or in the future
                if date_expiration >= today:
                    date_str = date_expiration.strftime("%Y-%m-%d")
                    if option_type == "C":
                        if date_str not in res["Call"]:
                            res["Call"][date_str] = []
                        res["Call"][date_str].append(strike_price)
                    elif option_type == "P":
                        if date_str not in res["Put"]:
                            res["Put"][date_str] = []
                        res["Put"][date_str].append(strike_price)
                    else:
                        print(f"Unknown option type {option_type} for {option_symbol}")

            # Sort the strike prices for each expiration date
            for option_type in res:
                for date_str in res[option_type]:
                    res[option_type][date_str].sort()

            # Sort the dictionary by expiration date
            res["Call"] = dict(sorted(res["Call"].items()))
            res["Put"] = dict(sorted(res["Put"].items()))

            if res and res != {"Call": {}, "Put": {}}:
                save_json(res, symbol)


    
