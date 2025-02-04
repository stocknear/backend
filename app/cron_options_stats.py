from __future__ import print_function
import asyncio
import time
from datetime import datetime, timedelta
import orjson
from tqdm import tqdm
import sqlite3
from dotenv import load_dotenv
import os
import re
from statistics import mean


# Database connection and symbol retrieval
def get_total_symbols():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
        stocks_symbols = [row[0] for row in cursor.fetchall()]

    with sqlite3.connect('etf.db') as etf_con:
        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    return stocks_symbols + etf_symbols


def get_tickers_from_directory():
    directory = "json/options-historical-data/companies"
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        
        # Get all tickers from filenames
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def save_json(data, symbol):
    directory = "json/options-stats/companies"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))


def safe_round(value):
    try:
        return round(float(value), 2)
    except (ValueError, TypeError):
        return value


async def main():
    total_symbols = get_total_symbols()
    
    for symbol in tqdm(total_symbols):
        try:
            # Load previous data and calculate changes
            with open(f"json/options-historical-data/companies/{symbol}.json", "r") as file:
                data = orjson.loads(file.read())

           # Keys to compute the average for
            keys_to_average = [key for key in data[0] if key != "date"]

            # Compute averages and round to 2 decimal places
            averages = {
                key: round(mean(d[key] for d in data if d.get(key) is not None), 2)
                for key in keys_to_average
            }

            save_json(averages, symbol)

        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
