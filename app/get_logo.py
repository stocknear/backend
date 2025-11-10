import os
import requests
from PIL import Image
from io import BytesIO
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create output directory if it doesn't exist
output_dir = "../../frontend/static/logo/"
os.makedirs(output_dir, exist_ok=True)

# Retrieve symbols from SQLite databases
def get_symbols(db_path, query):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode = wal")
    cur.execute(query)
    symbols = [row[0] for row in cur.fetchall()]
    con.close()
    return symbols

stocks_symbols = get_symbols('stocks.db', "SELECT DISTINCT symbol FROM stocks")
etf_symbols = get_symbols('etf.db', "SELECT DISTINCT symbol FROM etfs")
index_symbols = ['^SPX', '^VIX']

total_symbols = stocks_symbols + etf_symbols + index_symbols

# Function to download and convert image for one symbol
def process_symbol(symbol, session):
    url = f"https://financialmodelingprep.com/image-stock/{symbol}.png"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        output_path = os.path.join(output_dir, f"{symbol}.webp")
        image.save(output_path, "WEBP")
        return f"Successfully converted {symbol} to WebP."
    except requests.exceptions.RequestException as e:
        return f"Failed to download {symbol}: {e}"
    except Exception as e:
        return f"Error processing {symbol}: {e}"

# Using ThreadPoolExecutor for concurrent downloads and conversions
def main():
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_symbol, symbol, session): symbol for symbol in total_symbols}
            for future in as_completed(futures):
                result = future.result()
                print(result)

if __name__ == "__main__":
    main()
