from __future__ import print_function
import asyncio
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
import orjson
from tqdm import tqdm
import os
from collections import defaultdict
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

MAX_CONCURRENT_REQUESTS = 100
BATCH_SIZE = 4000

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

async def get_single_contract_data(symbol, contract_id):
    try:
        # Fixed: Use .json extension instead of .symbol
        with open(f"json/all-options-contracts/{symbol}/{contract_id}.json", "rb") as file:
            contract_data = orjson.loads(file.read())
        
        # Get the latest data point (most recent date)
        if not contract_data.get('history'):
            print(f"No history data found for contract {contract_id}")
            return None
            
        # Get the most recent entry
        latest_data = contract_data['history'][-1]
        
        # Extract the open interest from the latest data point
        open_interest = latest_data.get('open_interest', 0.0)
        
        # Create the contract info
        contract_info = {
            'strike': contract_data.get('strike'),
            'expiration': contract_data.get('expiration'),
            'type': contract_data.get('optionType'),  # Fixed: use 'optionType' from your data
            'open_interest': open_interest,
            'contract_code': contract_id
        }
        
        return {
            'expiration': contract_data.get('expiration'),
            'contracts': [contract_info]  # Wrap in list for consistency
        }
    except Exception as e:
        print(f"Error processing contract {contract_id}: {e}")
        return None

async def process_batch(symbol, batch, pbar):
    symbol = symbol.replace("^","") #for index symbols

    tasks = [get_single_contract_data(symbol, contract) for contract in batch]
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
            
            batch_results = await process_batch(symbol, batch, pbar)
            results.extend(batch_results)
                    
    return results

def aggregate_open_interest(symbol, results):
    # Group by expiration date, similar to get_strike_data pattern
    data_by_expiry = defaultdict(lambda: defaultdict(lambda: {
        "strike": 0,
        "call_oi": 0,
        "put_oi": 0,
    }))
    
    expiration_data = defaultdict(lambda: {'call_oi': 0, 'put_oi': 0})
    
    for result in results:
        if not result or 'contracts' not in result:
            continue
        
        expiration = result['expiration']
        
        for contract in result['contracts']:
            try:
                strike = float(contract['strike'])
                option_type = contract['type']
                open_interest = contract['open_interest'] or 0

                # Aggregate by strike for each expiration
                slot = data_by_expiry[expiration][strike]
                slot["strike"] = strike
                
                if option_type == 'call':
                    slot["call_oi"] += open_interest
                    expiration_data[expiration]['call_oi'] += open_interest
                elif option_type == 'put':
                    slot["put_oi"] += open_interest
                    expiration_data[expiration]['put_oi'] += open_interest
                    
            except Exception as e:
                print(f"Error processing contract: {e}")
                continue
    
    # Convert strike data to the format similar to get_strike_data
    strike_result = {}
    for expiry, strikes in data_by_expiry.items():
        strike_list = []
        for strike, stats in sorted(strikes.items(), key=lambda x: x[0]):
            # Only include strikes with non-zero open interest
            if stats["call_oi"] + stats["put_oi"] != 0:
                strike_list.append({
                    "strike": stats["strike"],
                    "call_oi": stats["call_oi"],
                    "put_oi": stats["put_oi"]
                })
        
        # Only include expiration dates that have data
        if strike_list:
            strike_result[expiry] = strike_list
    
    # Convert expiration data to list format
    expiration_list = [
        {
            "call_oi": data['call_oi'],
            "put_oi": data['put_oi'],
            "expiry": expiry,
        }
        for expiry, data in sorted(expiration_data.items())
        if data['call_oi'] + data['put_oi'] != 0  # Filter out empty entries
    ]

    # Save aggregated data
    if strike_result:
        save_json(strike_result, symbol, 'strike')
        print(f"Saved strike data for {symbol}: {len(strike_result)} expirations")
    if expiration_list:
        save_json(expiration_list, symbol, 'expiry')
        print(f"Saved expiry data for {symbol}: {len(expiration_list)} expirations")


async def main():
    # Get list of symbols
    total_symbols = get_tickers_from_directory()
    print(f"Number of tickers: {len(total_symbols)}")

    for symbol in total_symbols:
        try:
            # Get list of contracts for the symbol
            contract_list = get_contracts_from_directory(symbol)
            
            if not contract_list:
                print(f"No contracts found for {symbol}")
                continue

            print(f"Found {len(contract_list)} contracts for {symbol}")

            # Process contracts
            results = await process_contracts(symbol, contract_list)
            
            # Aggregate and save open interest data
            aggregate_open_interest(symbol, results)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    asyncio.run(main())