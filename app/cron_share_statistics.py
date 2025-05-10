import os
import orjson
import sqlite3
import asyncio
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import requests
from dotenv import load_dotenv

load_dotenv()

# Constants
next_year = datetime.now().year + 1
start_date = "2015-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
api_key = os.getenv("BENZINGA_API_KEY")

def get_total_symbols():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
        stocks_symbols = [row[0] for row in cursor.fetchall()]
    return stocks_symbols

def _save_files(symbol, forward_pe_dict, short_dict):
    Path("json/share-statistics").mkdir(parents=True, exist_ok=True)
    Path("json/forward-pe").mkdir(parents=True, exist_ok=True)
    with open(f"json/share-statistics/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(short_dict))
    with open(f"json/forward-pe/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(forward_pe_dict))


def calculate_forward_pe(symbol):
    estimates_path = Path("json/analyst-estimate") / f"{symbol}.json"
    quote_path = Path("json/quote") / f"{symbol}.json"
    try:
        with estimates_path.open('rb') as file:
            estimates = orjson.loads(file.read())
        with quote_path.open('rb') as file:
            price_data = orjson.loads(file.read())
        price = price_data.get('price')
        item = next((i for i in estimates if i.get('date') == next_year), None)
        if item:
            eps = item.get('estimatedEpsAvg')
            if eps:
                return round(price / eps, 2)
    except Exception as e:
        print(e)
        return None
    return None


def get_short_data(ticker):
    url = (
        f"https://api.benzinga.com/api/v1/shortinterest"
        f"?token={api_key}&pageSize=5000&symbols={ticker}"
        f"&from={start_date}&to={end_date}"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json().get('shortInterestData', {}).get(ticker, {}).get('data', [])
    keep_keys = [
        'recordDate','totalShortInterest','shortPercentOfFloat',
        'daysToCover','shortPriorMo','percentChangeMoMo',
        'sharesOutstanding','sharesFloat'
    ]
    history = [{k: r.get(k) for k in keep_keys} for r in data]

    if not history:
        return {'sharesShort': None,'shortRatio': None,'sharesShortPriorMonth': None,
                'shortOutstandingPercent': None,'shortFloatPercent': None,'history': []}

    # Load price history
    path = Path("json/historical-price/max") / f"{ticker}.json"
    with path.open('rb') as f:
        price_history = orjson.loads(f.read())
    price_map = {item['time']: item.get('close') for item in price_history}


    for entry in history:
        try:
            entry['price'] = price_map.get(entry.get('recordDate'))
            if int(entry.get('sharesOutstanding', 0)) > 0:
                entry['shortPercentOfOut'] = round(
                    int(entry.get('totalShortInterest', 0)) / int(entry.get('sharesOutstanding', 0)) * 100,
                    2
                )
            else:
                entry['shortPercentOfOut'] = 0
        except:
            pass

    latest = history[-1]
    out = int(latest.get('sharesOutstanding') or 0)
    flt = int(latest.get('sharesFloat') or 0)
    si = int(latest.get('totalShortInterest') or 0)
    spm = int(latest.get('shortPriorMo') or 0)
    ratio = float(latest.get('daysToCover') or 0)
    out_pct = round(si / out * 100, 2) if out else 0
    float_pct = round(si / flt * 100, 2) if flt else 0

    return {
        'sharesShort': si,
        'shortRatio': ratio,
        'sharesShortPriorMonth': spm,
        'shortOutstandingPercent': out_pct,
        'shortFloatPercent': float_pct,
        'history': history
    }

def _wrap_to_thread(func, *args):
    return asyncio.to_thread(func, *args)

async def get_data(ticker):
    # Parallel compute forward PE and short data
    fpe_task = _wrap_to_thread(calculate_forward_pe, ticker)
    short_task = _wrap_to_thread(get_short_data, ticker)
    forward_pe, short_data = await asyncio.gather(fpe_task, short_task)
    return {'forwardPE': forward_pe}, short_data

async def save_as_json(symbol, forward_pe_dict, short_dict):
    await asyncio.to_thread(_save_files, symbol, forward_pe_dict, short_dict)

async def process_symbol(ticker, semaphore):
    async with semaphore:
        try:
            fpe_dict, short_dict = await get_data(ticker)
            if short_dict and fpe_dict:
                await save_as_json(ticker, fpe_dict, short_dict)
        except Exception as e:
            print(e)

async def run():
    total_symbols = get_total_symbols()

    #Testing mode
    #total_symbols = ['JD']
    # Limit concurrent tasks
    concurrency = 10
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [asyncio.create_task(process_symbol(sym, semaphore)) for sym in total_symbols]
    pbar = tqdm(total=len(tasks))
    for coro in asyncio.as_completed(tasks):
        await coro
        pbar.update(1)
    pbar.close()

if __name__ == '__main__':
    try:
        asyncio.run(run())
    except Exception as e:
        print(e)
