import os
import orjson
import time
import sqlite3
import requests
import random
import math
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('MAIN_STREET_API_KEY')
headers = {
    "accept": "application/json",
    "x-api-key": api_key,
    "Content-Type": "application/json"
}

# Throttle configuration
CALLS_BEFORE_SLEEP = 100
SLEEP_SECONDS = 60
# small jitter between calls to avoid bursting the API
MIN_JITTER = 0.05
MAX_JITTER = 0.25

# Max batch size per API request
MAX_BATCH_SIZE = 20

# global counter for API calls
api_call_count = 0


def clean_metrics(metrics):
    """Remove left/right symbols and rename x→date, y→val in values."""
    cleaned = []

    for m in metrics:
        m.pop("leftSymbol", None)
        m.pop("rightSymbol", None)

        # Rename x/y inside values
        new_values = []
        for v in m.get("values", []):
            new_v = {
                "date": v.get("x"),
                "val": v.get("y")
            }
            # Keep other fields (like percentRevenue, valueType, etc.)
            for k, v2 in v.items():
                if k not in ["x", "y"]:
                    new_v[k] = v2
            new_values.append(new_v)

        m["values"] = new_values
        cleaned.append(m)

    return cleaned


def save_json(data, period, symbol):
    path = f"json/business-metrics/{period}/{symbol}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        file.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def sleep_after_calls():
    """Sleep when api_call_count reaches multiples of CALLS_BEFORE_SLEEP."""
    global api_call_count
    if api_call_count and api_call_count % CALLS_BEFORE_SLEEP == 0:
        print(f"[throttle] reached {api_call_count} calls — sleeping {SLEEP_SECONDS} seconds...")
        time.sleep(SLEEP_SECONDS)


def get_data(batch_symbols):
    """Call the API for the given batch of symbols."""
    global api_call_count

    data = {"tickers": batch_symbols}

    for period in ['annual', 'quarterly', 'ttm']:
        url = f"https://api.mainstreetdata.com/api/v1/companies?freq={period}&YoY=true&percentRevenue=true"

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            companies = response.json().get("companies", [])
        except Exception as e:
            # Log the error and continue; still count this as an API call
            print(f"API request failed for period={period}, batch size={len(batch_symbols)}: {e}")
            companies = []
        finally:
            # increment call counter and maybe sleep
            api_call_count += 1
            sleep_after_calls()

        if companies:
            for item in companies:
                try:
                    symbol = item["ticker"]
                    metrics = clean_metrics(item.get("metrics", []))
                    save_json(metrics, period, symbol)
                except Exception as e:
                    print(f"Error processing {item.get('ticker', 'Unknown')}: {e}")

            # small jitter to avoid making back-to-back requests too quickly
        time.sleep(random.uniform(MIN_JITTER, MAX_JITTER))


def run():
    con = sqlite3.connect("stocks.db")
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    total_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    # Testing (remove when ready)
    # total_symbols = ["AAPL"]

    if not total_symbols:
        print("No symbols found.")
        return

    # Ensure each chunk (batch) contains at most MAX_BATCH_SIZE symbols.
    chunk_size = min(MAX_BATCH_SIZE, len(total_symbols))
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]

    print(f"Total symbols: {len(total_symbols)} → {len(chunks)} chunk(s) of up to {chunk_size} each.")

    for chunk in tqdm(chunks):
        get_data(chunk)


if __name__ == "__main__":
    run()
