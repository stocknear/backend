from __future__ import print_function
import asyncio
import time
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
from datetime import datetime, timedelta
import orjson
from tqdm import tqdm
import os
from collections import defaultdict
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor


load_dotenv()
api_key = os.getenv('INTRINIO_API_KEY')

# Configure Intrinio SDK
intrinio.ApiClient().set_api_key(api_key)
intrinio.ApiClient().allow_retries(True)

# Configuration
MAX_CONCURRENT_REQUESTS = 50
BATCH_SIZE = 1500
include_related_symbols = False

def save_json(data, symbol, category="strike"):
    directory_path = f"json/oi/{category}/"
    os.makedirs(directory_path, exist_ok=True)
    with open(f"{directory_path}/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))

def get_tickers_from_directory():
    directory = "json/options-historical-data/companies"
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
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    except:
        return []

async def get_single_contract_data(symbol, expiration, semaphore):
    async with semaphore:
        try:
            # Use ThreadPoolExecutor to run synchronous API calls
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                response = await loop.run_in_executor(
                    pool,
                    lambda: intrinio.OptionsApi().get_options_chain_eod(symbol, expiration, include_related_symbols=include_related_symbols)
                )
            
            # Process the options chain data
            contract_data = []
            for item in response.chain:
                try:
                    option_price_data = item.prices
                    dict_data = option_price_data.__dict__

                    contract_data.append({
                        'strike': item.option.strike,
                        'expiration': item.option.expiration,
                        'type': item.option.type,
                        'open_interest': dict_data.get('_open_interest', 0),
                        'contract_code': item.option.code
                    })
                except Exception as e:
                    print(f"Error processing contract item: {e}")
            
            return {
                'expiration': expiration,
                'contracts': contract_data
            }
        except Exception as e:
            print(f"Error processing expiration {expiration}: {e}")
            return None

async def process_batch(symbol, batch, semaphore, pbar):
    tasks = [get_single_contract_data(symbol, contract, semaphore) for contract in batch]
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
    
    total_contracts = len(contract_list)
    total_batches = (total_contracts + BATCH_SIZE - 1) // BATCH_SIZE
    
    with tqdm(total=total_contracts, desc=f"Processing {symbol} contracts") as pbar:
        for batch_num in range(total_batches):
            start_idx = batch_num * BATCH_SIZE
            batch = contract_list[start_idx:start_idx + BATCH_SIZE]
            
            print(f"\nProcessing batch {batch_num + 1}/{total_batches} ({len(batch)} contracts)")
            batch_start_time = time.time()
            
            batch_results = await process_batch(symbol, batch, semaphore, pbar)
            results.extend(batch_results)
            
            batch_time = time.time() - batch_start_time
            
            if batch_num < total_batches - 1:
                print(f"Sleeping for 30 seconds before next batch...")
                await asyncio.sleep(30)
    return results

def aggregate_open_interest(symbol, results):
    strike_data = defaultdict(lambda: {'call_open_interest': 0, 'put_open_interest': 0})
    expiration_data = defaultdict(lambda: {'call_open_interest': 0, 'put_open_interest': 0})
    
    for result in results:
        if not result or 'contracts' not in result:
            continue
        
        for contract in result['contracts']:
            try:
                strike = contract['strike']
                option_type = contract['type']
                open_interest = contract['open_interest']
                expiration = contract['expiration']

                if option_type == 'call':
                    strike_data[strike]['call_open_interest'] += open_interest
                    expiration_data[expiration]['call_open_interest'] += open_interest
                elif option_type == 'put':
                    strike_data[strike]['put_open_interest'] += open_interest
                    expiration_data[expiration]['put_open_interest'] += open_interest
            except Exception as e:
                print(f"Error processing contract: {e}")

    # Convert to sortable list format
    strike_data = sorted(strike_data.items(), key=lambda x: x[0], reverse=True)
    strike_data = [
        {
            "call_oi": data[1]['call_open_interest'],
            "put_oi": data[1]['put_open_interest'],
            "strike": data[0],
        }
        for data in strike_data
    ]

    expiration_data = sorted(expiration_data.items(), key=lambda x: x[0])
    expiration_data = [
        {
            "call_oi": data[1]['call_open_interest'],
            "put_oi": data[1]['put_open_interest'],
            "expiry": data[0],
        }
        for data in expiration_data
    ]

    # Save aggregated data
    if strike_data:
        save_json(strike_data, symbol, 'strike')
    if expiration_data:
        save_json(expiration_data, symbol, 'expiry')


async def main():
    # Get list of symbols
    total_symbols = get_tickers_from_directory()
    print(f"Number of tickers: {len(total_symbols)}")

    total_symbols = ['AA']

    for symbol in total_symbols:
        try:
            # Get list of contracts for the symbol
            contract_list = get_contracts_from_directory(symbol)
            
            if not contract_list:
                print(f"No contracts found for {symbol}")
                continue

            # Process contracts
            results = await process_contracts(symbol, contract_list)
            # Aggregate and save open interest data
            aggregate_open_interest(symbol, results)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    asyncio.run(main())