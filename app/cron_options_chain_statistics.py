from datetime import datetime, date
from collections import defaultdict
from tqdm import tqdm
import sqlite3
import orjson
import os

today = date.today()


def save_json(data, symbol):
    directory_path="json/options-chain-statistics/"
    os.makedirs(directory_path, exist_ok=True)
    filepath = os.path.join(directory_path, f"{symbol}.json")
    with open(filepath, "wb") as f:
        f.write(orjson.dumps(data))

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
            
            # Add IV to the combined list regardless of option type
            if iv > 0:  # Only include non-zero IV values
                by_exp[exp_str]["iv_all"].append(iv)
            
            if opt_type == "call":
                by_exp[exp_str]["volume_calls"] += volume
                by_exp[exp_str]["oi_calls"] += oi
            elif opt_type == "put":
                by_exp[exp_str]["volume_puts"] += volume
                by_exp[exp_str]["oi_puts"] += oi
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    
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
    

    # Sort by expiration (fixed the key name)
    result = []
    for exp, stats in by_exp.items():
        try:
            calls_vol = stats["volume_calls"]
            puts_vol = stats["volume_puts"]
            calls_oi = stats["oi_calls"]
            puts_oi = stats["oi_puts"]
            
            vol_ratio = safe_div(puts_vol, calls_vol)
            oi_ratio = safe_div(puts_oi, calls_oi)
            
            # Calculate average IV for all contracts with this expiration
            avg_iv = round(sum(stats["iv_all"]) / len(stats["iv_all"]) * 100, 2) if stats["iv_all"] else 0
            
            # Get max pain for this expiration
            max_pain = max_pain_by_exp.get(exp, 0)
            
            result.append({
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
    result.sort(key=lambda x: x["expiration"])
    return result


def load_symbol_list():
    symbols = []
    db_configs = [
        ("stocks.db", "SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'") ,
        ("etf.db",    "SELECT DISTINCT symbol FROM etfs"),
        ("index.db",  "SELECT DISTINCT symbol FROM indices")
    ]

    for db_file, query in db_configs:
        try:
            con = sqlite3.connect(db_file)
            cur = con.cursor()
            cur.execute(query)
            symbols.extend([r[0] for r in cur.fetchall()])
            con.close()
        except Exception:
            continue

    return symbols

if __name__ == "__main__":
    symbols = load_symbol_list()
    #symbols = ['TSLA']  # override for testing
    for symbol in tqdm(symbols, desc="Option Chain Statistics Computation"):
        try:
            data = compute_option_chain_statistics(symbol)
            save_json(data, symbol)
        except:
            pass