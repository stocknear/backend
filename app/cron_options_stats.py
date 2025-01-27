from __future__ import print_function
import asyncio
import aiohttp
import time
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
from datetime import datetime, timedelta
import ast
import orjson
from tqdm import tqdm
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from dotenv import load_dotenv
import os
import re

load_dotenv()

api_key = os.getenv('INTRINIO_API_KEY')

intrinio.ApiClient().set_api_key(api_key)
#intrinio.ApiClient().allow_retries(True)

current_date = datetime.now().date()

source = ''
show_stats = ''
stock_price_source = ''
model = ''
show_extended_price = ''


after = datetime.today().strftime('%Y-%m-%d')
before = '2100-12-31'
include_related_symbols = False
page_size = 5000
MAX_CONCURRENT_REQUESTS = 50  # Adjust based on API rate limits
BATCH_SIZE = 1500



def get_expiration_date(option_symbol):
    # Define regex pattern to match the symbol structure
    match = re.match(r"([A-Z]+)(\d{6})([CP])(\d+)", option_symbol)
    if not match:
        raise ValueError(f"Invalid option_symbol format: {option_symbol}")
    
    ticker, expiration, option_type, strike_price = match.groups()
    
    # Convert expiration to datetime
    date_expiration = datetime.strptime(expiration, "%y%m%d").date()
    return date_expiration
    

# Database connection and symbol retrieval
def get_total_symbols():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
        stocks_symbols = [row[0] for row in cursor.fetchall()]

    with sqlite3.connect('etf.db') as etf_con:
        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    return stocks_symbols + etf_symbols


def get_tickers_from_directory():
    directory = "json/options-historical-data/companies"
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
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    except:
        return []

def save_json(data, symbol):
    directory = "json/options-stats/companies"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))


def safe_round(value):
    try:
        return round(float(value), 2)
    except (ValueError, TypeError):
        return value


async def get_price_batch_realtime(symbol, contract_list):
    # API Configuration
    api_url = "https://api-v2.intrinio.com/options/prices/realtime/batch"
    headers = {
        "Authorization": f"Bearer {api_key}"  # Replace with your actual API key
    }
    params = {
        "source": source,
        "show_stats": show_stats,
        "stock_price_source": stock_price_source,
        "model": model,
        "show_extended_price": show_extended_price
    }
    body = {
        "contracts": contract_list
    }

    # Make API request
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, params=params, json=body) as response:
            response_data = await response.json()
    
    contracts_data = response_data.get('contracts', [])
    
    res_dict = {
        'total_premium': 0, 'call_premium': 0, 'put_premium': 0,
        'volume': 0, 'call_volume': 0, 'put_volume': 0, 
        'gex': 0, 'dex': 0,
        'total_open_interest': 0, 'call_open_interest': 0, 'put_open_interest': 0,
        'iv_list': [],
        'time': None
    }

    for item in contracts_data:
        try:
            price_data = item.get('price', {})
            stats_data = item.get('stats', {})
            option_data = item.get('option', {})
            
            option_type = option_data.get('type', '').lower()
            volume = int(price_data.get('volume', 0)) if price_data.get('volume') is not None else 0
            open_interest = int(price_data.get('open_interest', 0)) if price_data.get('open_interest') is not None else 0
            last_price = price_data.get('last', 0) or 0
            premium = int(volume * last_price * 100)
            
            implied_volatility = stats_data.get('implied_volatility')
            gamma = stats_data.get('gamma', 0) or 0
            delta = stats_data.get('delta', 0) or 0

            # Update metrics
            res_dict['gex'] += gamma * open_interest * 100
            res_dict['dex'] += delta * open_interest * 100
            res_dict['total_premium'] += premium
            res_dict['volume'] += volume
            res_dict['total_open_interest'] += open_interest

            if option_type == 'call':
                res_dict['call_premium'] += premium
                res_dict['call_volume'] += volume
                res_dict['call_open_interest'] += open_interest
            else:
                res_dict['put_premium'] += premium
                res_dict['put_volume'] += volume
                res_dict['put_open_interest'] += open_interest

            if implied_volatility is not None:
                res_dict['iv_list'].append(implied_volatility)
            
            # Handle timestamp
            if 'ask_timestamp' in price_data and price_data['ask_timestamp']:
                timestamp_str = price_data['ask_timestamp']
                try:
                    dt = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    res_dict['time'] = dt.strftime("%Y-%m-%d")
                except:
                    res_dict['time'] = timestamp_str[:10]  # Fallback to string slicing
        except Exception as e:
            print(f"Error processing contract: {e}")
            continue

    return res_dict


async def main():
    total_symbols = get_tickers_from_directory()
    if len(total_symbols) < 3000:
        total_symbols = get_total_symbols()
    print(f"Number of tickers: {len(total_symbols)}")

    for symbol in tqdm(total_symbols):
        try:
            contract_list = get_contracts_from_directory(symbol)
            if len(contract_list) > 0:
                # Initialize aggregated results dictionary
                aggregated_results = {
                    'total_premium': 0, 'call_premium': 0, 'put_premium': 0,
                    'volume': 0, 'call_volume': 0, 'put_volume': 0, 
                    'gex': 0, 'dex': 0,
                    'total_open_interest': 0, 'call_open_interest': 0, 'put_open_interest': 0,
                    'iv_list': [],
                    'time': None
                }

                # Process batches of 250 contracts
                for i in range(0, len(contract_list), 250):
                    batch = contract_list[i:i+250]
                    batch_results = await get_price_batch_realtime(symbol, batch)
                    
                    # Aggregate results
                    for key in ['total_premium', 'call_premium', 'put_premium', 
                                'volume', 'call_volume', 'put_volume', 
                                'gex', 'dex', 
                                'total_open_interest', 'call_open_interest', 'put_open_interest']:
                        aggregated_results[key] += batch_results[key]
                    
                    aggregated_results['iv_list'].extend(batch_results['iv_list'])
                    aggregated_results['time'] = batch_results['time']

                # Calculate final metrics
                aggregated_results['iv'] = round((sum(aggregated_results['iv_list']) / len(aggregated_results['iv_list'])*100), 2) if aggregated_results['iv_list'] else 0
                aggregated_results['putCallRatio'] = round(aggregated_results['put_volume'] / aggregated_results['call_volume'], 2) if aggregated_results['call_volume'] > 0 else 0

                # Load previous data and calculate changes
                with open(f"json/options-historical-data/companies/{symbol}.json", "r") as file:
                    past_data = orjson.loads(file.read())
                    index = next((i for i, item in enumerate(past_data) if item['date'] == aggregated_results['time']), 0)
                    previous_open_interest = past_data[index]['total_open_interest']

                aggregated_results['changesPercentageOI'] = round((aggregated_results['total_open_interest']/previous_open_interest-1)*100, 2)
                aggregated_results['changeOI'] = aggregated_results['total_open_interest'] - previous_open_interest

                # Remove the temporary iv_list before saving
                del aggregated_results['iv_list']

                # Save aggregated results
                save_json(aggregated_results, symbol)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
