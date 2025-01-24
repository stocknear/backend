from __future__ import print_function
import asyncio
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



def get_expiration_date(contract_id):
    # Extract the date part (YYMMDD) from the contract ID
    date_str = contract_id[2:8]
    # Convert to datetime object
    return datetime.strptime(date_str, "%y%m%d").date()


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


def get_all_expirations(symbol):
    response = intrinio.OptionsApi().get_options_expirations_eod(
        symbol, 
        after=after, 
        before=before, 
        include_related_symbols=include_related_symbols
    )
    data = (response.__dict__).get('_expirations')
    return data

async def get_options_chain(symbol, expiration, semaphore):
    async with semaphore:
        try:
            # Run the synchronous API call in a thread pool since intrinio doesn't support async
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                response = await loop.run_in_executor(
                    pool,
                    lambda: intrinio.OptionsApi().get_options_chain_eod(
                        symbol,
                        expiration,
                        include_related_symbols=include_related_symbols
                    )
                )
            contracts = set()
            for item in response.chain:
                try:
                    contracts.add(item.option.code)
                except Exception as e:
                    print(f"Error processing contract in {expiration}: {e}")
            return contracts
            
        except:
            return set()



async def get_price_batch_realtime(symbol, contract_list):
    body = {
      "contracts": contract_list
    }
    response = intrinio.OptionsApi().get_options_prices_batch_realtime(body, source=source, show_stats=show_stats, stock_price_source=stock_price_source, model=model, show_extended_price=show_extended_price)
    data = response.__dict__
    data = data['_contracts']
    
    res_dict = {
        'total_premium': 0, 'call_premium': 0, 'put_premium': 0,
        'volume': 0, 'call_volume': 0, 'put_volume': 0, 
        'gex': 0, 'dex': 0,
        'total_open_interest': 0, 'call_open_interest': 0, 'put_open_interest': 0,
        'iv_list': [],
        'time': None
    }

    for item in data:
        try:
            price_data = (item.__dict__)['_price'].__dict__
            stats_data = (item.__dict__)['_stats'].__dict__
            option_type = ((item.__dict__)['_option'].__dict__)['_type']
            
            volume = int(price_data['_volume']) if price_data['_volume'] != None else 0

            total_open_interest = int(price_data['_open_interest']) if price_data['_open_interest'] != None else 0
            last_price = price_data['_last'] if price_data['_last'] != None else 0
            premium = int(volume * last_price * 100)
            implied_volatility = stats_data['_implied_volatility']
            gamma = stats_data['_gamma'] if stats_data['_gamma'] != None else 0
            delta = stats_data['_delta'] if stats_data['_delta'] != None else 0

            res_dict['gex'] += gamma * total_open_interest * 100
            res_dict['dex'] += delta * total_open_interest * 100
            res_dict['total_premium'] += premium
            res_dict['volume'] += volume
            res_dict['total_open_interest'] += total_open_interest

            if option_type == 'call':
                res_dict['call_premium'] += premium
                res_dict['call_volume'] += volume
                res_dict['call_open_interest'] += total_open_interest
            else:
                res_dict['put_premium'] += premium
                res_dict['put_volume'] += volume
                res_dict['put_open_interest'] += total_open_interest

            res_dict['iv_list'].append(implied_volatility)
            res_dict['time'] = price_data['_ask_timestamp'].strftime("%Y-%m-%d")
        except:
            pass

    return res_dict


async def prepare_dataset(symbol):
    expiration_list = get_all_expirations(symbol)
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Create tasks for all expirations
    tasks = [get_options_chain(symbol, expiration, semaphore) for expiration in expiration_list]
    # Show progress bar for completed tasks
    contract_sets = set()
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing expirations"):
        contracts = await task
        contract_sets.update(contracts)
    
    # Convert final set to list
    contract_list = list(contract_sets)


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
