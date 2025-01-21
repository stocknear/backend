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

directory_path = "json/all-options-contracts"

async def save_json(data, symbol, contract_id):
    directory_path = f"json/all-options-contracts/{symbol}"
    os.makedirs(directory_path, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory_path}/{contract_id}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))

def safe_round(value):
    try:
        return round(float(value), 2)
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
#intrinio.ApiClient().allow_retries(True)

after = datetime.today().strftime('%Y-%m-%d')
before = '2100-12-31'
N_year_ago = datetime.now() - timedelta(days=365)
include_related_symbols = False
page_size = 5000
MAX_CONCURRENT_REQUESTS = 50  # Adjust based on API rate limits
BATCH_SIZE = 1500

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
            
        except Exception as e:
            print(f"Error fetching chain for {expiration}: {e}")
            return set()


async def get_single_contract_eod_data(symbol, contract_id, semaphore):
    async with semaphore:
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                # Fetch data using ThreadPoolExecutor
                response = await loop.run_in_executor(
                    pool,
                    lambda: intrinio.OptionsApi().get_options_prices_eod(identifier=contract_id)
                )
            
            # Extract and process the response data
            key_data = {k: v for k, v in response._option.__dict__.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}

            history = []
            if response and hasattr(response, '_prices'):
                for price in response._prices:
                    history.append({
                        k: v for k, v in price.__dict__.items() 
                        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                    })


            #clean the data
            history = [
                {key.lstrip('_'): value for key, value in record.items() if key not in ('_close_time','_open_ask', '_ask_low','_close_bid_size','_close_ask_size','_close_ask','_close_bid','_close_size','_exercise_style','discriminator','_open_bid','_bid_low','_bid_high','_ask_high')}
                for record in history
            ]


            #ignore small volume and oi contracts to filter trash contracts... oh hi mark
            total_volume = sum(item['volume'] or 0 for item in history)
            total_open_interest = sum(item['open_interest'] or 0 for item in history)
            count = len(history)
            avg_volume = int(total_volume / count) if count > 0 else 0
            avg_open_interest = int(total_open_interest / count) if count > 0 else 0

            if avg_volume > 10 and avg_open_interest > 10:
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
                        changes_percentage_oi = round((current_open_interest/previous_open_interest -1)*100,2)
                        res_list[i]['changeOI'] = current_open_interest - previous_open_interest
                        res_list[i]['changesPercentageOI'] = changes_percentage_oi
                    except:
                        res_list[i]['changeOI'] = None
                        res_list[i]['changesPercentageOI'] = None

                for i in range(1,len(res_list)):
                    try:
                        volume = res_list[i]['volume']
                        avg_fill = res_list[i]['mark']
                        res_list[i]['total_premium'] = int(avg_fill*volume*100)
                    except:
                        res_list[i]['total_premium'] = 0

                
                data = {'expiration': key_data['_expiration'], 'strike': key_data['_strike'], 'optionType': key_data['_type'], 'history': res_list}
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
            start_idx = batch_num * BATCH_SIZE
            batch = contract_list[start_idx:start_idx + BATCH_SIZE]
            
            print(f"\nProcessing batch {batch_num + 1}/{total_batches} ({len(batch)} contracts)")
            batch_start_time = time.time()
            
            # Process the batch concurrently
            batch_results = await process_batch(symbol, batch, semaphore, pbar)
            results.extend(batch_results)
            
            batch_time = time.time() - batch_start_time
            print(f"Batch completed in {batch_time:.2f} seconds")
            
            # Sleep between batches if not the last batch
            if batch_num < total_batches - 1:
                print(f"Sleeping for 60 seconds before next batch...")
                await asyncio.sleep(60)
    
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

    return stocks_symbols + etf_symbols

async def main():
    
    total_symbols = ['AA'] #get_total_symbols()

    for symbol in tqdm(total_symbols):
        try:
            print(f"==========Start Process for {symbol}==========")
            expiration_list = get_all_expirations(symbol)
            print(f"Found {len(expiration_list)} expiration dates")
            contract_list = await get_data(symbol,expiration_list)
            print(f"Unique contracts: {len(contract_list)}")

            results = await process_contracts(symbol, contract_list)
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())