import os
import sqlite3
import orjson
import numpy as np

from datetime import datetime, date
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
today = date.today()


def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value


def save_json(data, symbol):
    directory_path="json/max-pain/"
    os.makedirs(directory_path, exist_ok=True)
    filepath = os.path.join(directory_path, f"{symbol}.json")
    with open(filepath, "wb") as f:
        f.write(orjson.dumps(data))


def get_contracts_from_directory(directory: str):
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith(".json")]


def compute_max_pain_for_symbol(symbol: str):
    base_dir = os.path.join("json/all-options-contracts", symbol)
    contract_files = get_contracts_from_directory(base_dir)
    by_exp = defaultdict(lambda: defaultdict(lambda: {"call_oi": 0, "put_oi": 0}))

    # Load and bucket by expiration and strike
    for filepath in contract_files:
        try:
            with open(filepath, "rb") as f:
                data = orjson.loads(f.read())
            exp_str = data.get("expiration")
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            if exp_date < today:
                continue

            strike = float(data.get("strike", 0))
            oi = data.get("history", [])[-1].get("open_interest", 0) or 0
            opt_type = data.get("optionType", "").lower()

            if opt_type == "call":
                by_exp[exp_str][strike]["call_oi"] += oi
            elif opt_type == "put":
                by_exp[exp_str][strike]["put_oi"] += oi
        except Exception:
            continue

    results = []
    # Compute detailed payouts per expiration
    for exp_str, strikes_dict in by_exp.items():
        strikes = sorted(strikes_dict.keys())
        Ks = np.array(strikes)
        call_oi = np.array([strikes_dict[K]["call_oi"] for K in strikes])
        put_oi = np.array([strikes_dict[K]["put_oi"] for K in strikes])

        call_payouts = []
        put_payouts = []
        combined = []
        for S in Ks:
            pay_calls = call_oi * np.maximum(S - Ks, 0)
            pay_puts = put_oi * np.maximum(Ks - S, 0)
            call_payouts.append(pay_calls.sum())
            put_payouts.append(pay_puts.sum())
            combined.append(pay_calls.sum() + pay_puts.sum())

        # Determine max pain strike
        idx_min = int(np.argmin(combined))
        max_pain_strike = strikes[idx_min]

        # Round arrays for JSON
        call_vals = [safe_round(v) for v in call_payouts]
        put_vals = [safe_round(v) for v in put_payouts]

        result = {
            "expiration": exp_str,
            "strikes": strikes,
            "callPayouts": call_vals,
            "putPayouts": put_vals,
            "maxPain": safe_round(max_pain_strike),
        }
        results.append(result)

        

    # Sort chronologically
    results.sort(key=lambda x: datetime.strptime(x["expiration"], "%Y-%m-%d"))
    # Save summary
    save_json(results, symbol)
    #print(results)
    return results


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


def main():
    symbols = load_symbol_list()
    #symbols = ['TSLA']  # override for testing
    for sym in tqdm(symbols, desc="Max-Pain Computation"):
        try:
            compute_max_pain_for_symbol(sym)
        except Exception as e:
            print(f"Error processing {sym}: {e}")


if __name__ == "__main__":
    main()
