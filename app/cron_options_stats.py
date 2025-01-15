import aiohttp
import asyncio
import orjson
from dotenv import load_dotenv
import os
import sqlite3

load_dotenv()

api_key = os.getenv('UNUSUAL_WHALES_API_KEY')

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


def calculate_neutral_premium(data_item):
    call_premium = float(data_item['call_premium'])
    put_premium = float(data_item['put_premium'])
    bearish_premium = float(data_item['bearish_premium'])
    bullish_premium = float(data_item['bullish_premium'])

    total_premiums = bearish_premium + bullish_premium
    observed_premiums = call_premium + put_premium
    neutral_premium = observed_premiums - total_premiums

    return safe_round(neutral_premium)


def prepare_data(data):
    for item in data:
        try:
            symbol = item['ticker']
            bearish_premium = float(item['bearish_premium'])
            bullish_premium = float(item['bullish_premium'])
            neutral_premium = calculate_neutral_premium(item)

            new_item = {
                key: safe_round(value)
                for key, value in item.items()
                if key != 'in_out_flow'
            }

            new_item['premium_ratio'] = [
                safe_round(bearish_premium),
                neutral_premium,
                safe_round(bullish_premium)
            ]
            new_item['open_interest_change'] = (
                new_item['total_open_interest'] - 
                (new_item.get('prev_call_oi', 0) + new_item.get('prev_put_oi', 0))
                if 'total_open_interest' in new_item else None
            )

            if new_item:
                save_json(new_item, symbol)
        except:
            pass


async def fetch_data(session, chunk):
    chunk_str = ",".join(chunk)
    url = "https://api.unusualwhales.com/api/screener/stocks"
    params = {"ticker": chunk_str}
    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": api_key
    }

    try:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                json_data = await response.json()
                data = json_data.get('data', [])
                prepare_data(data)
                print(f"Processed chunk with {len(data)} results.")
            else:
                print(f"Error fetching chunk {chunk_str}: {response.status}")
    except Exception as e:
        print(f"Exception fetching chunk {chunk_str}: {e}")


async def main():
    total_symbols = get_tickers_from_directory()
    if len(total_symbols) < 3000:
        total_symbols = get_total_symbols()
    print(f"Number of tickers: {len(total_symbols)}")
    chunk_size = 50
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(chunks), 200):  # Process 200 chunks at a time
            try:
                tasks = [fetch_data(session, chunk) for chunk in chunks[i:i + 200]]
                await asyncio.gather(*tasks)
                print("Processed 200 chunks. Sleeping for 60 seconds...")
                await asyncio.sleep(60)  # Avoid API rate limits
            except:
                pass


if __name__ == "__main__":
    asyncio.run(main())
