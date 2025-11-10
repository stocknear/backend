import os
import sqlite3
import orjson
import numpy as np

from datetime import datetime, date
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm
import time

import asyncio
import aiofiles
import sys


load_dotenv()
today = date.today()


def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value


def save_json(data):
    directory_path = "json/screener"
    os.makedirs(directory_path, exist_ok=True)  # Create directory if it doesn't exist
    filepath = os.path.join(directory_path, "options-screener.json")
    with open(filepath, "wb") as file:
        file.write(orjson.dumps(data))


def compute_iv_rank(option_history):
    if len(option_history) == 0:
        return None

    ivs = [
        entry['implied_volatility']
        for entry in option_history
        if entry.get('implied_volatility') is not None and entry['implied_volatility'] > 0
    ]
    
    if len(ivs) < 2:
        # need at least two points to define a range
        return None

    iv_min = min(ivs)
    iv_max = max(ivs)
    
    # avoid divide-by-zero if all IVs are identical
    if iv_max == iv_min:
        return 0  # or 100.0, depending on your convention

    iv_current = ivs[-1]  # assume last in list is "today"
    iv_rank = round((iv_current - iv_min) / (iv_max - iv_min) * 100,2)
    return iv_rank

def get_unique_expirations(options_list):
    return sorted(set(option['expiration'] for option in options_list))

def get_contracts_from_directory(directory: str):
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith(".json")]


def get_screener(symbol: str, name:str, current_stock_price: float, asset_type:str):
    base_dir = os.path.join("json/all-options-contracts", symbol)
    contract_files = get_contracts_from_directory(base_dir)
    results = []
    # Load and bucket by expiration and strike
    for filepath in contract_files:
        try:
            with open(filepath, "rb") as f:
                data = orjson.loads(f.read())
            exp_str = data.get("expiration")
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            if exp_date < today:
                continue

            option_symbol = filepath.split('/')[-1].replace('.json', '')
            latest_data = data.get("history", [])[-1]
            strike = float(data.get("strike", 0))
            oi = latest_data.get("open_interest", None)
            opt_type = data.get("optionType", "").capitalize()
            implied_volatility = round(latest_data.get("implied_volatility", None) * 100,2)
            change_oi = latest_data.get("changeOI", None)
            change_oi_percentage = latest_data.get('changesPercentageOI',None)
            delta = latest_data.get("delta", None)
            gamma = latest_data.get("gamma", None)
            theta = latest_data.get("theta", None)
            vega = latest_data.get("vega", None)
            #mark = latest_data.get('mark',None)
            close = latest_data.get('close',None)
            volume = latest_data.get('volume',None)
            if opt_type == 'Call':
                moneyness = round((current_stock_price/strike -1)*100,2)
            elif opt_type == 'Put':
                moneyness = round((strike / current_stock_price - 1) * 100, 2)

            total_prem = latest_data.get('total_premium',0)

            iv_rank = compute_iv_rank(data.get('history', []))

            if implied_volatility and oi and delta and gamma and theta and vega and volume and iv_rank:
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "optionSymbol": option_symbol,
                    "assetType": asset_type,
                    "expiration": exp_str,
                    "strike": strike,
                    "oi": oi,
                    "optionType": opt_type,
                    "iv": implied_volatility,
                    "ivRank": iv_rank,
                    #"changeOI": change_oi,
                    "changesPercentageOI": change_oi_percentage,
                    #"mark": mark,
                    "volume": volume,
                    "close": close,
                    "delta": delta,
                    "gamma": gamma,
                    "theta": theta,
                    "vega": vega,
                    "moneynessPercentage": moneyness,
                    "totalPrem": total_prem,
                })
                
        except Exception as e:
            print(e)
    
    return results


def load_symbol_list():
    symbol_dict = {
        "stocks": [],
        "etfs": [],
        "indices": []
    }

    db_configs = [
        ("stocks", "stocks.db", "SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')"),
        ("etfs", "etf.db", "SELECT DISTINCT symbol FROM etfs"),
        ("indices", "index.db", "SELECT DISTINCT symbol FROM indices")
    ]

    for key, db_file, query in db_configs:
        try:
            with sqlite3.connect(db_file) as con:
                cur = con.cursor()
                cur.execute(query)
                symbol_dict[key] = [r[0] for r in cur.fetchall()]
        except Exception as e:
            # Optionally print or log the error if needed
            symbol_dict[key] = []

    return symbol_dict


def is_fully_defined(item):
    """
    Returns True if all values in the item are not None.
    If a value is a dict, ensures all its sub-values are not None.
    """
    for v in item.values():
        if v is None:
            return False
        if isinstance(v, dict):
            for sv in v.values():
                if sv is None:
                    return False
    return True

def create_dataset():

    symbols_dict = load_symbol_list()
    stock_symbols = symbols_dict.get('stocks')
    etf_symbols = symbols_dict.get('etfs')
    index_symbols = symbols_dict.get('indices')
    total_symbols = stock_symbols + etf_symbols + index_symbols

    #total_symbols = ['BLDR']  # override for testing

    res = []
    
    for symbol in tqdm(total_symbols, desc="Options Screener"):
        try:
            with open(f"json/quote/{symbol}.json","rb") as file:
                quote_data = orjson.loads(file.read())
                current_stock_price = quote_data.get('price', None)
                name = quote_data.get('name')
                if symbol in stock_symbols:
                    asset_type = 'Stock'
                elif symbol in etf_symbols:
                    asset_type = 'ETF'
                else:
                    asset_type = 'Index'

            if current_stock_price:
                res += get_screener(symbol, name, current_stock_price, asset_type)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if res:
        sorted_res = sorted(res,key=lambda x: datetime.strptime(x['expiration'], "%Y-%m-%d"))
        
        #unique_expirations = get_unique_expirations(sorted_res)
        #filtered_res = [item for item in sorted_res if item['expiration'] == unique_expirations[0]]

        clean_data = [item for item in sorted_res if is_fully_defined(item)]

        save_json(clean_data)


async def load_quote_data_async(symbol):
    """Asynchronously load quote data for a single symbol"""
    try:
        quote_path = f"json/quote/{symbol}.json"
        if os.path.exists(quote_path):
            async with aiofiles.open(quote_path, 'rb') as file:
                content = await file.read()
                quote_data = orjson.loads(content)
                return symbol, quote_data.get('price')
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
    return symbol, None

async def update_dataset():
    # Load symbols once
    symbols_dict = load_symbol_list()
    total_symbols = (symbols_dict.get('stocks', []) + 
                    symbols_dict.get('etfs', []) + 
                    symbols_dict.get('indices', []))
    
    # Load options screener data once
    async with aiofiles.open("json/screener/options-screener.json", 'rb') as file:
        content = await file.read()
        options_screener_data = orjson.loads(content)
    
    # Create semaphore to limit concurrent file operations
    semaphore = asyncio.Semaphore(50)  # Adjust based on system limits
    
    async def bounded_load_quote(symbol):
        async with semaphore:
            return await load_quote_data_async(symbol)
    
    # Load all quotes asynchronously
    print("Loading quote data...")
    tasks = [bounded_load_quote(symbol) for symbol in total_symbols]
    quote_results = await asyncio.gather(*tasks)
    
    # Create price lookup dictionary
    symbol_prices = {symbol: price for symbol, price in quote_results if price is not None}
    
    # Update moneyness calculations
    print("Calculating moneyness...")
    for item in options_screener_data:
        try:
            symbol = item.get('symbol')  # Assuming options have symbol field
            opt_type = item.get('optionType')
            if symbol in symbol_prices:
                current_stock_price = symbol_prices[symbol]
                strike = item.get('strike')
                if current_stock_price and strike:
                    if opt_type == 'Call':
                        moneyness = round((current_stock_price/strike -1)*100,2)
                    elif opt_type == 'Put':
                        moneyness = round((strike / current_stock_price - 1) * 100, 2)
                    item['moneynessPercentage'] = moneyness
        except:
            pass
    
    # Filter fully defined items
    clean_data = [item for item in options_screener_data if is_fully_defined(item)]
    print(f"Processed {len(clean_data)} valid options")
    print(clean_data[0])
    return clean_data


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'update':
        print("Updating screener")
        asyncio.run(update_dataset())
    else:
        print("Create options screener from scratch:")
        create_dataset()
