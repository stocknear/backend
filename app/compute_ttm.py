import orjson
import sqlite3
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from itertools import islice
from collections import deque

def convert_types(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

def save_json(symbol, data, file_type):
    """Save JSON data to the appropriate directory based on file type."""
    if file_type == "ratios":
        path = f"json/financial-statements/ratios/ttm-updated"
    elif file_type == "income-statement-growth":
        path = f"json/financial-statements/income-statement-growth/ttm-updated"
    else:
        raise ValueError(f"Unknown file type: {file_type}")
        
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data, default=convert_types))

def get_symbols():
    with sqlite3.connect('stocks.db') as con:
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA cache_size = -50000")  # 50MB cache
        con.execute("PRAGMA temp_store = MEMORY")
        con.execute("PRAGMA synchronous = NORMAL")
        symbols = con.execute(
            "SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'"
        ).fetchall()
    return [row[0] for row in symbols]

def calculate_ttm_ratios(statements):
    if not statements or len(statements) < 4:
        return []
    
    # Sort by date
    statements.sort(key=lambda x: x['date'])
    
    # Identify columns to exclude from TTM calculation
    exclude_columns = {'fiscalYear', 'period', 'reportedCurrency', 'date', 'symbol'}
    
    # Get all keys from the first entry
    all_columns = set(statements[0].keys())
    
    # Identify numeric columns for TTM calculation
    numeric_columns = [col for col in all_columns if col not in exclude_columns]
    
    ttm_results = []
    window = deque(maxlen=4)
    
    # Use a sliding window of 4 quarters for TTM calculation
    for entry in statements:
        window.append(entry)
        
        if len(window) == 4:  # We have 4 quarters
            ttm_entry = {
                'symbol': entry['symbol'],
                'date': entry['date'],
                'period': 'TTM',
                'reportedCurrency': entry['reportedCurrency'],
                'fiscalYear': entry['fiscalYear']
            }
            
            # Calculate TTM values for all numeric columns
            for col in numeric_columns:
                try:
                    ttm_entry[col] = sum(q.get(col, 0) for q in window)
                except (TypeError, ValueError):
                    # Skip columns that can't be summed
                    continue
            
            ttm_results.append(ttm_entry)
    
    return ttm_results

def process_file(symbol, file_type):
    """Process a single file for TTM calculation."""
    try:
        if file_type == "ratios":
            file_path = f"json/financial-statements/ratios/quarter/{symbol}.json"
        elif file_type == "income-statement-growth":
            file_path = f"json/financial-statements/income-statement-growth/quarter/{symbol}.json"
        else:
            return (symbol, file_type, False, f"Unknown file type: {file_type}")
        
        if not os.path.exists(file_path):
            return (symbol, file_type, False, "File not found")
        
        with open(file_path, "rb") as file:
            statements = orjson.loads(file.read())
        
        data = calculate_ttm_ratios(statements)
        
        if data:
            save_json(symbol, data, file_type)
            return (symbol, file_type, True, "Success")
        return (symbol, file_type, False, "No TTM data generated")
    except Exception as e:
        return (symbol, file_type, False, str(e))

def process_symbol(symbol):
    """Process both file types for a given symbol."""
    results = []
    
    # Process ratios
    results.append(process_file(symbol, "ratios"))
    
    # Process income statement growth
    results.append(process_file(symbol, "income-statement-growth"))
    
    return results

def batch(iterable, size):
    """Yield successive batches from iterable."""
    iterator = iter(iterable)
    while True:
        batch_iter = islice(iterator, size)
        batch_list = list(batch_iter)
        if not batch_list:
            break
        yield batch_list

def main():
    symbols = get_symbols()
    batch_size = 100  # Process in batches to avoid memory issues
    
    # Track success and error counts for each file type
    stats = {
        "ratios": {"success": 0, "error": 0},
        "income-statement-growth": {"success": 0, "error": 0}
    }
    
    # Use maximum number of available cores minus 1
    max_workers = max(os.cpu_count() - 1, 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for symbol_batch in batch(symbols, batch_size):
            # Process each symbol (which now handles both file types)
            batch_results = list(executor.map(process_symbol, symbol_batch))
            
            # Flatten results
            results = [item for sublist in batch_results for item in sublist]
            
            for symbol, file_type, success, message in results:
                if success:
                    stats[file_type]["success"] += 1
                else:
                    stats[file_type]["error"] += 1
                    print(f"Error processing {file_type} for {symbol}: {message}")
    
    # Print final statistics
    for file_type, counts in stats.items():
        print(f"{file_type}: {counts['success']} successful, {counts['error']} failed")

if __name__ == "__main__":
    main()