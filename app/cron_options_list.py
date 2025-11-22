import orjson
import sqlite3
import asyncio
import aiofiles
from heapq import nlargest
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import List, Dict, Any

async def get_contracts_from_directory(directory: str) -> List[str]:
    """Get contract filenames from directory using pathlib for better performance."""
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        return [file.stem for file in dir_path.glob("*.json")]
    except Exception as e:
        print(f"Error reading directory {directory}: {e}")
        return []

async def process_contract_file(symbol: str, contract: str) -> Dict[str, Any]:
    """Process a single contract file asynchronously."""
    try:
        file_path = f"json/all-options-contracts/{symbol}/{contract}.json"
        async with aiofiles.open(file_path, "rb") as file:
            data = orjson.loads(await file.read())
        
        history = data.get("history", [])
        if not history:
            return None

        latest = history[-1]
        volume = latest.get("volume", 0) or 0
        open_interest = latest.get("open_interest", 0) or 0

        return {
            "symbol": symbol,
            "optionSymbol": contract,
            "optionType": data.get('optionType'),
            "strike": data.get('strike'),
            "expirationDate": data.get('expiration'),
            "optionVolume": int(volume),
            "totalOI": int(open_interest),
            "volumeOIRatio": round(volume/open_interest,2) if open_interest > 0 else None
        }
    except Exception:
        return None

async def process_symbol(symbol: str) -> List[Dict[str, Any]]:
    """Process all contracts for a single symbol and return valid results."""
    contract_dir = f"json/all-options-contracts/{symbol}"
    contracts = await get_contracts_from_directory(contract_dir)
    
    if not contracts:
        return []

    # Process contracts concurrently
    tasks = [process_contract_file(symbol, contract) for contract in contracts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None values and exceptions
    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    
    return valid_results

def print_top_contracts(top_volume: list, top_oi: list, batch_num: int):
    """Print the highest volume and OI contracts for the current batch."""
    if top_volume:
        top_vol_contract = nlargest(1, top_volume, key=lambda x: x["optionVolume"])
        if top_vol_contract:
            print(f"Epoch {batch_num} - Highest Volume: {top_vol_contract[0]['optionSymbol']} "
                  f"(Volume: {top_vol_contract[0]['optionVolume']:,})")
    
    if top_oi:
        top_oi_contract = nlargest(1, top_oi, key=lambda x: x["totalOI"])
        if top_oi_contract:
            print(f"Epoch {batch_num} - Highest OI: {top_oi_contract[0]['optionSymbol']} "
                  f"(OI: {top_oi_contract[0]['totalOI']:,})")

async def get_option_contract_list():
    """Main function to get option contract lists."""
    # Get symbols from database
    with sqlite3.connect('stocks.db') as con:
        con.execute("PRAGMA journal_mode = wal")
        cursor = con.cursor()
        cursor.execute("""
            SELECT DISTINCT symbol FROM stocks 
            WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'
        """)
        symbols = [row[0] for row in cursor.fetchall()]

    # Use lists to maintain top 100 efficiently
    top_volume = []
    top_oi = []

    # Process symbols in batches to control memory usage
    batch_size = 50
    batch_num = 0
    
    for i in range(0, len(symbols), batch_size):
        batch_num += 1
        batch = symbols[i:i + batch_size]
        
        # Process batch concurrently
        tasks = [process_symbol(symbol) for symbol in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and filter out exceptions
        all_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                continue
            all_results.extend(result)
        
        # Update top lists
        if all_results:
            top_volume.extend(all_results)
            top_oi.extend(all_results)
            
            # Keep only top 100 in memory using nlargest for efficiency
            top_volume = nlargest(100, top_volume, key=lambda x: x["optionVolume"])
            top_oi = nlargest(100, top_oi, key=lambda x: x["totalOI"])
        
        print(f"Epoch {batch_num}: Processed {min(i + batch_size, len(symbols))}/{len(symbols)} symbols")
        print_top_contracts(top_volume, top_oi, batch_num)
        print("-" * 80)

    # Final sort and add ranks
    top_volume_final = nlargest(100, top_volume, key=lambda x: x["optionVolume"])
    top_oi_final = nlargest(100, top_oi, key=lambda x: x["totalOI"])

    for rank, item in enumerate(top_volume_final, 1):
        item["rank"] = rank
    for rank, item in enumerate(top_oi_final, 1):
        item["rank"] = rank

    # Ensure output directory exists
    output_dir = Path("json/stocks-list/list")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(output_dir / "highest-volume-by-contract.json", "wb") as f:
        f.write(orjson.dumps(top_volume_final, option=orjson.OPT_INDENT_2))

    with open(output_dir / "highest-open-interest-by-contract.json", "wb") as f:
        f.write(orjson.dumps(top_oi_final, option=orjson.OPT_INDENT_2))

    print(f"Completed: {len(top_volume_final)} volume records, {len(top_oi_final)} OI records")
    
    # Print final top contracts
    if top_volume_final:
        print(f"\nFINAL - Highest Volume: {top_volume_final[0]['optionSymbol']} "
              f"(Volume: {top_volume_final[0]['optionVolume']:,})")
    if top_oi_final:
        print(f"FINAL - Highest OI: {top_oi_final[0]['optionSymbol']} "
              f"(OI: {top_oi_final[0]['totalOI']:,})")

if __name__ == "__main__":
    asyncio.run(get_option_contract_list())