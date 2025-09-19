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


def safe_round(value, decimals=4):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return None


def save_json(data, symbol):
    directory_path = "json/greeks/"
    os.makedirs(directory_path, exist_ok=True)
    filepath = os.path.join(directory_path, f"{symbol}.json")
    with open(filepath, "wb") as f:
        f.write(orjson.dumps(data))


def get_contracts_from_directory(directory: str):
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith(".json")]


def get_data(symbol: str):
    base_dir = os.path.join("json/all-options-contracts", symbol)
    contract_files = get_contracts_from_directory(base_dir)
    by_exp = defaultdict(lambda: defaultdict(lambda: {"call": None, "put": None}))

    # Load and bucket by expiration & strike
    for filepath in contract_files:
        try:
            with open(filepath, "rb") as f:
                data = orjson.loads(f.read())

            exp_str = data.get("expiration")
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            if exp_date < today:
                continue

            strike = float(data.get("strike", 0))
            opt_type = data.get("optionType", "").lower()

            history = data.get("history", [])
            if not history:
                continue

            last = history[-1]  # take latest snapshot
            oi = last.get("open_interest", 0) or 0
            if oi == 0:
                continue

            greeks = {
                "delta": last.get("delta"),
                "gamma": last.get("gamma"),
                "theta": last.get("theta"),
                "vega": last.get("vega"),
            }

            by_exp[exp_str][strike][opt_type] = {"oi": oi, **greeks}

        except Exception:
            continue

    results = []
    # Build per-expiration strike-based arrays
    for exp_str, strikes_dict in by_exp.items():
        strikes = sorted(strikes_dict.keys())

        call_delta, call_gamma, call_theta, call_vega = [], [], [], []
        put_delta, put_gamma, put_theta, put_vega = [], [], [], []

        for K in strikes:
            call_data = strikes_dict[K]["call"]
            put_data = strikes_dict[K]["put"]

            if call_data:
                call_delta.append(safe_round(call_data["delta"]))
                call_gamma.append(safe_round(call_data["gamma"]))
                call_theta.append(safe_round(call_data["theta"]))
                call_vega.append(safe_round(call_data["vega"]))
            else:
                call_delta.append(None)
                call_gamma.append(None)
                call_theta.append(None)
                call_vega.append(None)

            if put_data:
                put_delta.append(safe_round(put_data["delta"]))
                put_gamma.append(safe_round(put_data["gamma"]))
                put_theta.append(safe_round(put_data["theta"]))
                put_vega.append(safe_round(put_data["vega"]))
            else:
                put_delta.append(None)
                put_gamma.append(None)
                put_theta.append(None)
                put_vega.append(None)

        result = {
            "expiration": exp_str,
            "strikes": strikes,
            "callDelta": call_delta,
            "callGamma": call_gamma,
            "callTheta": call_theta,
            "callVega": call_vega,
            "putDelta": put_delta,
            "putGamma": put_gamma,
            "putTheta": put_theta,
            "putVega": put_vega,
        }
        results.append(result)

    # Sort chronologically
    results.sort(key=lambda x: datetime.strptime(x["expiration"], "%Y-%m-%d"))

    save_json(results, symbol)
    return results


def load_symbol_list():
    symbols = []
    db_configs = [
        ("stocks.db", "SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'"),
        ("etf.db", "SELECT DISTINCT symbol FROM etfs"),
        ("index.db", "SELECT DISTINCT symbol FROM indices"),
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
    symbols = ["TSLA"]  # override for testing
    for sym in tqdm(symbols, desc="Greeks Computation"):
        try:
            get_data(sym)
        except Exception as e:
            print(f"Error processing {sym}: {e}")


if __name__ == "__main__":
    main()
