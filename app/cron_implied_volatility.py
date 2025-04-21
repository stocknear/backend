import orjson
import os
import sqlite3
import time
from tqdm import tqdm
import numpy as np


con = sqlite3.connect('stocks.db')
etf_con = sqlite3.connect('etf.db')
index_con = sqlite3.connect('index.db')

cursor = con.cursor()
cursor.execute("PRAGMA journal_mode = wal")
#cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND marketCap > 1E9")
cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
stocks_symbols = [row[0] for row in cursor.fetchall()]

etf_cursor = etf_con.cursor()
etf_cursor.execute("PRAGMA journal_mode = wal")
#etf_cursor.execute("SELECT DISTINCT symbol FROM etfs WHERE marketCap > 1E9")
etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
etf_symbols = [row[0] for row in etf_cursor.fetchall()]

index_cursor = index_con.cursor()
index_cursor.execute("PRAGMA journal_mode = wal")
#index_cursor.execute("SELECT DISTINCT symbol FROM etfs WHERE marketCap > 1E9")
index_cursor.execute("SELECT DISTINCT symbol FROM indices")
index_symbols = [row[0] for row in index_cursor.fetchall()]

con.close()
etf_con.close()
index_cursor.close()

def get_tickers_from_directory(directory: str):
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        
        # Get all tickers from filenames
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def convert_to_serializable(obj):
    if isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (list, np.ndarray)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

def save_json(data, symbol):
    directory_path = "json/implied-volatility"
    os.makedirs(directory_path, exist_ok=True)  # Ensure the directory exists
    
    # Convert numpy types to JSON-serializable types
    serializable_data = convert_to_serializable(data)
    
    with open(f"{directory_path}/{symbol}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(serializable_data))


def is_outlier(value, values, n_sigmas=3):
    """
    Detect if a value is an outlier using the z-score method
    
    Args:
        value: The value to check
        values: List of values to compare against
        n_sigmas: Number of standard deviations to use as threshold (default: 3)
    
    Returns:
        bool: True if the value is an outlier, False otherwise
    """
    if value is None or not values:
        return False
    
    values = [v for v in values if v is not None]
    if not values:
        return False

    mean = np.mean(values)
    std = np.std(values)
    
    if std == 0:
        return False
        
    z_score = abs((value - mean) / std)
    return z_score > n_sigmas

def clean_iv_data(data):
    """
    Clean IV data by handling outliers
    
    Args:
        data: List of dictionaries containing IV values
        
    Returns:
        List of dictionaries with cleaned IV values
    """
    # Extract IV values
    iv_values = [item.get('iv') for item in data]
    
    # Create a copy of the data to modify
    cleaned_data = []
    
    window_size = 20  # Rolling window size for outlier detection
    
    for i, item in enumerate(data):
        cleaned_item = item.copy()
        iv = item.get('iv')
        
        if iv is not None:
            # Get a window of IV values centered around the current point
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2)
            window_values = [data[j].get('iv') for j in range(start_idx, end_idx)]
            
            # Check if the current IV is an outlier
            if is_outlier(iv, window_values):
                # Replace outlier with the median of nearby non-outlier values
                non_outlier_values = [
                    v for v in window_values 
                    if v is not None and not is_outlier(v, window_values)
                ]
                
                if non_outlier_values:
                    cleaned_item['iv'] = round(np.median(non_outlier_values), 2)
                else:
                    cleaned_item['iv'] = None
            else:
                cleaned_item['iv'] = round(iv, 2)
                
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

def compute_realized_volatility(data, window_size=20):
    """
    Compute the realized volatility of stock prices over a rolling window.
    Realized volatility is the annualized standard deviation of log returns of stock prices.
    """
    # First clean the IV data
    data = clean_iv_data(data)
    
    # Sort data by date (oldest first)
    data = sorted(data, key=lambda x: x['date'])
    
    # Extract stock prices and dates
    prices = [item.get('price') for item in data]  # Use .get() to handle missing keys
    dates = [item['date'] for item in data]
    
    # Compute log returns of stock prices, skipping None values
    log_returns = []
    for i in range(1, len(prices)):
        if prices[i] is not None and prices[i - 1] is not None and prices[i - 1] != 0:
            log_returns.append(np.log(prices[i] / prices[i - 1]))
        else:
            log_returns.append(None)  # Append None if price is missing or invalid
    
    # Compute realized volatility using a rolling window
    realized_volatility = []
    for i in range(len(log_returns)):
        if i < window_size - 1:
            # Not enough data for the window, append None
            realized_volatility.append(None)
        else:
            # Collect valid log returns in the window
            window_returns = []
            for j in range(i - window_size + 1, i + 1):
                if log_returns[j] is not None:
                    window_returns.append(log_returns[j])
            
            if len(window_returns) >= window_size:
                # Compute standard deviation of log returns over the window
                rv_daily = np.sqrt(np.sum(np.square(window_returns)) / window_size)
                # Annualize the realized volatility
                rv_annualized = rv_daily * np.sqrt(252)
                realized_volatility.append(rv_annualized)
            else:
                # Not enough valid data in the window, append None
                realized_volatility.append(None)
    
    # Shift realized volatility FORWARD by window_size days to align with IV from window_size days ago
    realized_volatility = realized_volatility[window_size - 1:] + [None] * (window_size - 1)
    
    # Create the resulting list
    rv_list = []
    for i in range(len(data)):
        try:
            rv_list.append({
                "date": data[i]["date"],
                "price": data[i].get("price"),  # Use .get() to handle missing keys
                "changesPercentage": data[i].get("changesPercentage", None),  # Default to None if missing
                "putCallRatio": data[i].get("putCallRatio", None),  # Default to None if missing
                "total_open_interest": data[i].get("total_open_interest", None),  # Default to None if missing
                "changesPercentageOI": data[i].get("changesPercentageOI", None),  # Default to None if missing
                "iv": data[i].get("iv", None),  # Default to None if missing
                "rv": round(realized_volatility[i], 2) if realized_volatility[i] is not None else None
            })
        except Exception as e:
            # If any error occurs, append a dictionary with default values
            rv_list.append({
                "date": data[i]["date"],
                "price": data[i].get("price", None),
                "changesPercentage": data[i].get("changesPercentage", None),
                "putCallRatio": data[i].get("putCallRatio", None),
                "total_open_interest": data[i].get("total_open_interest", None),
                "changesPercentageOI": data[i].get("changesPercentageOI", None),
                "iv": data[i].get("iv", None),
                "rv": None
            })
    
    # Sort the final list by date in descending order
    rv_list = sorted(rv_list, key=lambda x: x['date'], reverse=True)
    return rv_list

if __name__ == '__main__':
    directory_path = "json/implied-volatility"
    total_symbols = stocks_symbols + etf_symbols + index_symbols


    for symbol in tqdm(total_symbols):
        try:
            with open(f"json/options-historical-data/companies/{symbol}.json", "r") as file:
                data = orjson.loads(file.read())
            rv_list = compute_realized_volatility(data)

            if rv_list:
                save_json(rv_list, symbol)
        except:
            pass
