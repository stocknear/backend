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
import re


load_dotenv()
api_key = os.getenv('INTRINIO_API_KEY')

directory_path = "json/all-options-contracts"
current_date = datetime.now().date()


with sqlite3.connect('index.db') as index_con:
    index_cursor = index_con.cursor()
    index_cursor.execute("PRAGMA journal_mode = wal")
    index_cursor.execute("SELECT DISTINCT symbol FROM indices")
    #important: don't add ^ since intrino doesn't add it to the symbol
    index_symbols = [row[0].replace("^","") for row in index_cursor.fetchall()]



async def save_json(data, symbol, contract_id):
    # Additional safety check (should already be filtered, but just in case)
    if not is_valid_expiration(contract_id):
        print(f"Skipping save for {contract_id}: invalid expiration date")
        return
    
    if symbol in index_symbols:
        symbol = "^"+symbol
        contract_id = "^"+contract_id

    directory_path = f"json/all-options-contracts/{symbol}"
    os.makedirs(directory_path, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory_path}/{contract_id}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))

def safe_round(value):
    try:
        return round(float(value), 4)
    except (ValueError, TypeError):
        return value

class OptionsResponse:
    @property
    def chain(self):
        return self._chain
        
class ChainItem:
    @property
    def prices(self):
        return self._prices



intrinio.ApiClient().set_api_key(api_key)
intrinio.ApiClient().allow_retries(True)

after = (datetime.today()- timedelta(days=365*5)).strftime('%Y-%m-%d')
before = '2100-12-31'
include_related_symbols = False
page_size = 5000
MAX_CONCURRENT_REQUESTS = 100  # Adjust based on API rate limits
BATCH_SIZE = 1500

def get_all_expirations(symbol):
    response = intrinio.OptionsApi().get_options_expirations_eod(
        symbol, 
        after=after, 
        before=before, 
        include_related_symbols=include_related_symbols
    )
    all_expirations = (response.__dict__).get('_expirations', [])
    
    # Filter out expirations that are outside our valid range
    valid_expirations = []
    for expiration in all_expirations:
        try:
            # Convert expiration string to date for comparison
            exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            if current_date <= exp_date:
                valid_expirations.append(expiration)
                
        except Exception as e:
            print(f"Error parsing expiration date {expiration}: {e}")
    
    return valid_expirations

def get_contracts_from_directory(symbol):
    directory = f"json/all-options-contracts/{symbol}/"
    try:
        all_contracts = [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
        # Filter out contracts with invalid expiration dates
        valid_contracts = [contract for contract in all_contracts if is_valid_expiration(contract)]
        return valid_contracts
    except:
        return []

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
                    contract_id = item.option.code
                    # Filter out contracts with invalid expiration dates before adding to set
                    if is_valid_expiration(contract_id):
                        contracts.add(contract_id)
                    else:
                        print(f"Skipping contract {contract_id}: invalid expiration date")
                except Exception as e:
                    print(f"Error processing contract in {expiration}: {e}")
            return contracts
            
        except:
            return set()


async def get_single_contract_eod_data(symbol, contract_id, semaphore):

    async with semaphore:
        try:
            next_page = ''  # Reset for each source
            all_prices = []  # Accumulate all prices across pages
            key_data = {}  # Store option metadata
            
            while True:
                url = f"https://api-v2.intrinio.com/options/prices/{contract_id}/eod?api_key={api_key}"
                if next_page:
                    url += f"&next_page={next_page}"
                    
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status != 200:
                            print(f"Failed to fetch data for {contract_id}: {response.status}")
                            return None
                            
                        response_data = await response.json()
                        
                        # Store option metadata from first response
                        if not key_data and "option" in response_data:
                            key_data = {k: v for k, v in response_data.get("option", {}).items() 
                                      if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                        
                        # Accumulate prices from this page
                        if "prices" in response_data:
                            for price in response_data["prices"]:
                                all_prices.append({
                                    k: v for k, v in price.items() 
                                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                                })
                        
                        # Check for next page
                        next_page = response_data.get("next_page",None)
                        if not next_page:
                            break

            # Clean the data

            history = [
                {key.lstrip('_'): value for key, value in record.items() if key not in ('close_time', 'open_ask', 'ask_low', 'close_size', 'exercise_style', 'discriminator', 'open_bid', 'bid_low', 'bid_high', 'ask_high')}
                for record in all_prices
            ]

            # Ignore small volume and open interest contracts
            total_volume = sum(item.get('volume', 0) or 0 for item in history)
            total_open_interest = sum(item.get('open_interest', 0) or 0 for item in history)
            count = len(history)
            #avg_volume = int(total_volume / count) if count > 0 else 0
            #avg_open_interest = int(total_open_interest / count) if count > 0 else 0

            #filter out the trash
            res_list = []

            for item in history:
                try:
                    new_item = {
                        key: safe_round(value)
                        for key, value in item.items()
                    }
                    res_list.append(new_item)
                except:
                    pass

            res_list = sorted(res_list, key=lambda x: x['date'])

            for i in range(1, len(res_list)):
                try:
                    current_open_interest = res_list[i]['open_interest']
                    previous_open_interest = res_list[i-1]['open_interest'] or 0
                    changes_percentage_oi = round((current_open_interest / previous_open_interest - 1) * 100, 2)
                    res_list[i]['changeOI'] = current_open_interest - previous_open_interest
                    res_list[i]['changesPercentageOI'] = changes_percentage_oi
                except:
                    res_list[i]['changeOI'] = None
                    res_list[i]['changesPercentageOI'] = None

            for i in range(1, len(res_list)):
                try:
                    volume = res_list[i]['volume']
                    avg_fill = res_list[i]['mark']
                    res_list[i]['gex'] = res_list[i]['gamma'] * res_list[i]['open_interest'] * 100
                    res_list[i]['dex'] = res_list[i]['delta'] * res_list[i]['open_interest'] * 100
                    res_list[i]['total_premium'] = int(avg_fill * volume * 100)
                except:
                    res_list[i]['total_premium'] = 0

            data = {'expiration': key_data.get('expiration'), 'strike': key_data.get('strike'), 'optionType': key_data.get('type'), 'history': res_list}

            if data:
                await save_json(data, symbol, contract_id)

        except Exception as e:
            print(f"Error fetching data for {contract_id}: {e}")
            return None


    

async def get_data(symbol, expiration_list):
    # Use a semaphore to limit concurrent requests
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
    return contract_list


async def process_batch(symbol, batch, semaphore, pbar):
    tasks = [get_single_contract_eod_data(symbol, contract, semaphore) for contract in batch]
    results = []
    
    for task in asyncio.as_completed(tasks):
        result = await task
        if result:
            results.append(result)
        pbar.update(1)
    
    return results

async def process_contracts(symbol, contract_list):
    results = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Calculate total batches for better progress tracking
    total_contracts = len(contract_list)
    total_batches = (total_contracts + BATCH_SIZE - 1) // BATCH_SIZE
    with tqdm(total=total_contracts, desc="Processing contracts") as pbar:
        for batch_num in range(total_batches):
            try:
                start_idx = batch_num * BATCH_SIZE
                batch = contract_list[start_idx:start_idx + BATCH_SIZE]
                            
                # Process the batch concurrently
                batch_results = await process_batch(symbol, batch, semaphore, pbar)
                results.extend(batch_results)
            except:
                pass
        
    
    return results

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

    with sqlite3.connect('index.db') as index_con:
        index_cursor = index_con.cursor()
        index_cursor.execute("PRAGMA journal_mode = wal")
        index_cursor.execute("SELECT DISTINCT symbol FROM indices")
        #important: don't add ^ since intrino doesn't add it to the symbol
        index_symbols = [row[0].replace("^","") for row in index_cursor.fetchall()]

    return stocks_symbols + etf_symbols +index_symbols


def is_valid_expiration(contract_id):
    """Check if a contract has a valid expiration date (not expired and not too far in future)"""
    try:
        expiration_date = get_expiration_date(contract_id)
        return current_date <= expiration_date
    except Exception:
        return False

def get_expiration_date(option_symbol):
    # Define regex pattern to match the symbol structure
    match = re.match(r"([A-Z]+)(\d{6})([CP])(\d+)", option_symbol)
    if not match:
        raise ValueError(f"Invalid option_symbol format: {option_symbol}")
    
    ticker, expiration, option_type, strike_price = match.groups()
    
    # Convert expiration to datetime
    date_expiration = datetime.strptime(expiration, "%y%m%d").date()
    return date_expiration

def check_contract_expiry(symbol):
    directory = f"{directory_path}/{symbol}/"
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        
        # Iterate through all JSON files in the directory
        for file in os.listdir(directory):
            try:
                if file.endswith(".json"):
                    contract_id = file.replace(".json", "")
                    
                    # Check if the contract has invalid expiration date
                    if not is_valid_expiration(contract_id):
                        # Delete the invalid contract JSON file
                        os.remove(os.path.join(directory, file))
                        try:
                            expiration_date = get_expiration_date(contract_id)
                            if expiration_date < current_date:
                                print(f"Deleted expired contract: {contract_id} (expired on {expiration_date})")
                        except:
                            print(f"Deleted invalid contract: {contract_id} (could not parse expiration)")
            except Exception as e:
                print(f"Error processing contract {file}: {e}")
        
        # Return the list of valid contracts
        valid_contracts = []
        for file in os.listdir(directory):
            if file.endswith(".json"):
                contract_id = file.replace(".json", "")
                if is_valid_expiration(contract_id):
                    valid_contracts.append(contract_id)
        
        return valid_contracts
    
    except Exception as e:
        print(f"Error checking contract expiry for {symbol}: {e}")
        return []

async def process_symbol(symbol):
    try:
        print(f"==========Start Process for {symbol}==========")
        expiration_list = get_all_expirations(symbol)
        if len(expiration_list) < 0:
            expiration_list = get_contracts_from_directory(symbol)

        #check existing contracts and delete expired ones
        check_contract_expiry(symbol)

        print(f"Found {len(expiration_list)} expiration dates")
        contract_list = await get_data(symbol, expiration_list)
        print(f"Unique contracts: {len(contract_list)}")

        if len(contract_list) > 0:
            results = await process_contracts(symbol, contract_list)
    except Exception as e:
        print(e)


def get_tickers_from_directory(directory: str):
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return []
    try:
        return [
            folder 
            for folder in os.listdir(directory) 
            if os.path.isdir(os.path.join(directory, folder))
        ]
    except Exception as e:
        print(f"An error occurred while accessing '{directory}': {e}")
        return []

async def main():
    total_symbols = get_total_symbols() #get_tickers_from_directory(directory_path)
    
    for symbol in tqdm(total_symbols):
        await process_symbol(symbol)


if __name__ == "__main__":
    asyncio.run(main())