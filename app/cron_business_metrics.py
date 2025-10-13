import os
import orjson
import time
import sqlite3
import requests
import random
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('MAIN_STREET_API_KEY')
headers = {
    "accept": "application/json",
    "x-api-key": api_key,
    "Content-Type": "application/json"
}

MAX_BATCH_SIZE = 15

api_call_count = 0


def clean_metrics(metrics):
    """Remove left/right symbols and rename x→date, y→val in values."""
    cleaned = []

    for m in metrics:
        m.pop("leftSymbol", None)
        m.pop("rightSymbol", None)

        new_values = []
        for v in m.get("values", []):
            new_v = {
                "date": v.get("x"),
                "val": v.get("y")
            }
            for k, v2 in v.items():
                if k not in ["x", "y"]:
                    new_v[k] = v2
            new_values.append(new_v)

        m["values"] = new_values
        cleaned.append(m)

    return cleaned


def save_json(data, period, symbol):
    """Save metrics as JSON file."""
    path = f"json/business-metrics/{period}/{symbol}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        file.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))



def get_data(batch_symbols):
    """Fetch company data for a batch of symbols."""

    data = {"tickers": batch_symbols}

    for period in ['quarterly']:
        url = f"https://api.mainstreetdata.com/api/v1/companies?freq={period}&percentRevenue=true"

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            companies = response.json().get("companies", [])
        except Exception as e:
            print(f"API request failed for {period}, batch={batch_symbols}: {e}")
            companies = []

        if companies:
            for item in companies:
                try:
                    symbol = item["ticker"]
                    metrics = clean_metrics(item.get("metrics", []))
                    if len(metrics) > 0:
                        save_json(metrics, period, symbol)
                except Exception as e:
                    print(f"Error processing {item.get('ticker', 'Unknown')}: {e}")



def run():
    """Main execution function."""
    con = sqlite3.connect("stocks.db")
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    total_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    if not total_symbols:
        print("No symbols found.")
        return

    # Split into chunks of MAX_BATCH_SIZE
    chunks = [total_symbols[i:i + MAX_BATCH_SIZE] for i in range(0, len(total_symbols), MAX_BATCH_SIZE)]
    print(f"Total symbols: {len(total_symbols)} → {len(chunks)} chunks of up to {MAX_BATCH_SIZE} symbols each.")

    for chunk in tqdm(chunks, desc="Fetching data", unit="chunk"):    
        try:
            print(chunk)
            get_data(chunk)
            time.sleep(10)
        except Exception as e:
            print(f"Error in chunk {chunk}: {e}")

if __name__ == "__main__":
    run()
