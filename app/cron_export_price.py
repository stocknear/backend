from datetime import datetime, timedelta
import ujson
import sqlite3
import asyncio
import aiohttp
from tqdm import tqdm
import os
from dotenv import load_dotenv
from aiohttp import TCPConnector

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

def date_range_days(steps=20):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)  # 6 months ago
    while start_date < end_date:
        next_date = start_date + timedelta(days=steps)
        yield start_date.strftime("%Y-%m-%d"), min(next_date, end_date).strftime("%Y-%m-%d")
        start_date = next_date

async def get_data_batch(session, symbol, url_list):
    tasks = []
    for url in url_list:
        tasks.append(fetch_data(session, url))
    
    results = await asyncio.gather(*tasks)
    data = []
    for result in results:
        if result:
            data.extend(result)
    return data
    
def get_existing_data(symbol, interval):
    file_path = f"json/export/price/{interval}/{symbol}.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return ujson.load(file)
    return []

def get_missing_date_ranges(existing_data, start_date, end_date):
    existing_dates = {item['date'].split()[0] for item in existing_data}  # Use a set for O(1) lookup time
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    missing_ranges = []
    current_date = start_date
    range_start = None

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str not in existing_dates:
            if range_start is None:
                range_start = current_date
        else:
            # If we found an existing date, and we have a start for a missing range, add it
            if range_start:
                missing_ranges.append((range_start.strftime("%Y-%m-%d"), date_str))
                range_start = None
        current_date += timedelta(days=1)

    # If the loop ends and we still have an open range, add it
    if range_start:
        missing_ranges.append((range_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))

    return missing_ranges

async def fetch_data(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return []
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return []

async def get_data(session, symbol, time_period):
    steps = 20 if time_period == '30min' else 40
    existing_data = get_existing_data(symbol, time_period)
    res_list = existing_data
    urls_to_fetch = []

    for start_date, end_date in date_range_days(steps=steps):
        missing_ranges = get_missing_date_ranges(existing_data, start_date, end_date)
        for missing_start, missing_end in missing_ranges:
            url = f"https://financialmodelingprep.com/api/v3/historical-chart/{time_period}/{symbol}?serietype=bar&extend=false&from={missing_start}&to={missing_end}&apikey={api_key}"
            urls_to_fetch.append(url)

    if urls_to_fetch:
        fetched_data = await get_data_batch(session, symbol, urls_to_fetch)
        res_list.extend(fetched_data)

    if res_list:
        current_datetime = datetime.utcnow()
        filtered_data = {item['date']: item for item in res_list if datetime.strptime(item['date'], "%Y-%m-%d %H:%M:%S") <= current_datetime}
        sorted_data = sorted(filtered_data.values(), key=lambda x: x['date'], reverse=False)
        await save_json(symbol, sorted_data, time_period)

async def save_json(symbol, data, interval):
    os.makedirs(f"json/export/price/{interval}", exist_ok=True)
    with open(f"json/export/price/{interval}/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

async def process_symbol(session, symbol):
    # Process both 30min and 60min intervals
    await get_data(session, symbol, '30min')
    await get_data(session, symbol, '1hour')

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    con.close()
    etf_con.close()

    total_symbols = stock_symbols + etf_symbols

    # Use aiohttp connector with a higher limit for performance
    connector = TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i, symbol in enumerate(tqdm(total_symbols), 1):
            await process_symbol(session, symbol)
            if i % 100 == 0:
                print(f'Sleeping after processing {i} symbols')
                await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(run())
