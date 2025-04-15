import orjson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import yfinance as yf
import csv
from io import StringIO
from pathlib import Path
import requests

next_year = datetime.now().year + 1

async def save_as_json(symbol, forward_pe_dict, short_dict):
    with open(f"json/share-statistics/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(short_dict))
    with open(f"json/forward-pe/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(forward_pe_dict))

with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

def calculate_forward_pe(symbol):
    estimates_path = Path("json/analyst-estimate") / f"{symbol}.json"
    quote_path = Path("json/quote") / f"{symbol}.json"
    
    try:
        with estimates_path.open('rb') as file:
            estimates = orjson.loads(file.read())
        
        with quote_path.open('rb') as file:
            price_data = orjson.loads(file.read())
        price = price_data.get('price')
        
        estimate_item = next((item for item in estimates if item.get('date') == next_year), None)
        if estimate_item:
            eps = estimate_item.get('estimatedEpsAvg')
            if eps and eps != 0:
                return round(price / eps, 2)
    except (FileNotFoundError, ValueError, KeyError):
        return None
    return None

def download_csv_data(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def parse_csv_data(csv_text):
    csv_file = StringIO(csv_text)
    reader = csv.DictReader(csv_file, delimiter='|')
    return list(reader)

def get_short_data(ticker, outstanding_shares, float_shares, record_dict):
    row = record_dict.get(ticker.upper())
    if not row:
        return {'sharesShort': None, 'shortRatio': None, 'sharesShortPriorMonth': None, 
                'shortOutStandingPercent': None, 'shortFloatPercent': None}
    
    try:
        shares_short = int(row.get('currentShortPositionQuantity', 0))
    except ValueError:
        shares_short = 0

    try:
        shares_short_prior = int(row.get('previousShortPositionQuantity', 0))
    except ValueError:
        shares_short_prior = 0

    try:
        short_ratio = float(row.get('daysToCoverQuantity', 0))
    except ValueError:
        short_ratio = 0.0

    short_outstanding_percent = round((shares_short / outstanding_shares) * 100, 2) if outstanding_shares else 0
    short_float_percent = round((shares_short / float_shares) * 100, 2) if float_shares else 0

    return {
        'sharesShort': shares_short,
        'shortRatio': short_ratio,
        'sharesShortPriorMonth': shares_short_prior,
        'shortOutStandingPercent': short_outstanding_percent,
        'shortFloatPercent': short_float_percent
    }

async def get_data(ticker, record_dict):
    try:
        latest_outstanding_shares = stock_screener_data_dict[ticker]['sharesOutStanding']
        latest_float_shares = stock_screener_data_dict[ticker]['floatShares']

        forward_pe = calculate_forward_pe(ticker)
        forward_pe_dict = {'forwardPE': forward_pe}
        short_data = get_short_data(ticker, latest_outstanding_shares, latest_float_shares, record_dict)
        return forward_pe_dict, short_data
    except Exception as e:
        print(e)
        return {}, {}

async def run():
    url = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt20250228.csv"
    record_dict = {}
    
    try:
        csv_text = download_csv_data(url)
        records = parse_csv_data(csv_text)
        record_dict = {}
        for row in records:
            symbol_code = row.get('symbolCode', '').strip().upper()
            if symbol_code:
                record_dict[symbol_code] = row
    except Exception as e:
        print(f"Error processing CSV data: {e}")

    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    
    #Testing mode
    #stock_symbols = ['NVDA','AAPL']
    
    for ticker in tqdm(stock_symbols):
        try:
            forward_pe_dict, short_dict = await get_data(ticker, record_dict)
            if forward_pe_dict and short_dict:
                await save_as_json(ticker, forward_pe_dict, short_dict)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

try:
    asyncio.run(run())
except Exception as e:
    print(e)