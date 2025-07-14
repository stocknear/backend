from datetime import datetime, date, timedelta
from collections import defaultdict
from tqdm import tqdm
import sqlite3
import orjson
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import math

today = date.today()



def get_contracts_from_directory(directory: str):
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith(".json")]

def safe_div(a, b):
    return round(a / b, 2) if b else 0


def compute_option_chain_statistics(symbol):
    base_dir = os.path.join("json/all-options-contracts", symbol)
    contract_files = get_contracts_from_directory(base_dir)
    
    by_exp = defaultdict(lambda: {
        "volume_calls": 0,
        "volume_puts": 0,
        "oi_calls": 0,
        "oi_puts": 0,
        "iv_all": [],  # Store all IV values for this expiration
    })
    
    # Track overall statistics
    total_volume = 0
    total_call_volume = 0
    total_put_volume = 0
    total_oi = 0
    total_call_oi = 0
    total_put_oi = 0
    
    for filepath in contract_files:
        try:
            with open(filepath, "rb") as f:
                data = orjson.loads(f.read())
            
            exp_str = data.get("expiration")
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            
            if exp_date < today:
                continue
            

            history = data.get("history", [])
            if not history:
                continue
            
            latest = history[-1]
            opt_type = data.get("optionType", "").lower()
            volume = latest.get("volume", 0) or 0
            oi = latest.get("open_interest", 0) or 0
            iv = latest.get("implied_volatility", 0) or 0
            
            # Track overall volume and OI
            total_volume += volume
            total_oi += oi
            
            if iv > 0 and oi > 0 and volume > 0:
                by_exp[exp_str]["iv_all"].append(iv)
            print(latest)
            
            if opt_type == "call":
                by_exp[exp_str]["volume_calls"] += volume
                by_exp[exp_str]["oi_calls"] += oi
                total_call_volume += volume
                total_call_oi += oi
            elif opt_type == "put":
                by_exp[exp_str]["volume_puts"] += volume
                by_exp[exp_str]["oi_puts"] += oi
                total_put_volume += volume
                total_put_oi += oi
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    # Calculate overall statistics


    
    # Load max pain data
    max_pain_by_exp = {}
    try:
        with open(f"json/max-pain/{symbol}.json", "rb") as file:
            max_pain_data = orjson.loads(file.read())
            for entry in max_pain_data:
                exp_date = entry.get("expiration")
                max_pain = entry.get("maxPain", 0)
                max_pain_by_exp[exp_date] = max_pain
    except Exception as e:
        print(f"Error loading max pain data for {symbol}: {e}")
    
    # Build expiration-specific results
    expiration_data = []
    for exp, stats in by_exp.items():
        try:
            calls_vol = stats["volume_calls"]
            puts_vol = stats["volume_puts"]
            calls_oi = stats["oi_calls"]
            puts_oi = stats["oi_puts"]
            
            vol_ratio = safe_div(puts_vol, calls_vol)
            oi_ratio = safe_div(puts_oi, calls_oi)
            
            # Calculate average IV for all contracts with this expiration
            avg_iv = round(statistics.median(stats["iv_all"]) * 100, 2) if stats["iv_all"] else 0
            
            # Get max pain for this expiration
            max_pain = max_pain_by_exp.get(exp, 0)
            
            expiration_data.append({
                "expiration": exp,
                "callVol": calls_vol,
                "putVol": puts_vol,
                "pcVol": vol_ratio,
                "callOI": calls_oi,
                "putOI": puts_oi,
                "pcOI": oi_ratio,
                "avgIV": avg_iv,
                "maxPain": max_pain,
            })
        except:
            pass
    
    # Sort by expiration
    expiration_data.sort(key=lambda x: x["expiration"])
    
    print(expiration_data[1])

if __name__ == "__main__":
    symbol = 'GME'  # override for testing
    compute_option_chain_statistics(symbol)