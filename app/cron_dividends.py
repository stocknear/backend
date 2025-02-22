import aiohttp
import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import pytz
import orjson
import os
from dotenv import load_dotenv

headers = {"accept": "application/json"}
url = "https://api.benzinga.com/api/v2.1/calendar/dividends"
load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

ny_tz = pytz.timezone('America/New_York')
today = datetime.now(ny_tz).replace(hour=0, minute=0, second=0, microsecond=0)
N_days_ago = today - timedelta(days=10)


async def save_as_json(symbol, data, file_name):
    with open(f"{file_name}/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

async def get_data(ticker, con, etf_con, stock_symbols, etf_symbols):
    try:
        if ticker in etf_symbols:
            table_name = 'etfs'
            column_name = 'etf_dividend'
        else:
            table_name = 'stocks'
            column_name = 'stock_dividend'

        query_template = f"""
        SELECT 
            {column_name}, quote
        FROM 
            {table_name}
        WHERE
            symbol = ?
        """

        df = pd.read_sql_query(query_template, etf_con if table_name == 'etfs' else con, params=(ticker,))
    
        dividend_data = orjson.loads(df[column_name].iloc[0])
        res = dividend_data.get('historical', [])
        filtered_res = [item for item in res if item['recordDate'] and item['paymentDate']]
    
        # Get the current and previous year
        today = datetime.today()
        current_year = str(today.year)
        previous_year = str(today.year - 1)
    
        # Compute the previous year's total dividend (strictly based on last year)
        previous_year_records = [item for item in filtered_res if previous_year in item['recordDate']]
        previous_annual_dividend = round(sum(float(item['adjDividend']) for item in previous_year_records), 2) if previous_year_records else 0

        # Estimate the payout frequency dynamically from the current year's dividends
        current_year_records = [item for item in filtered_res if current_year in item['recordDate']]
        record_dates = sorted(
            [datetime.strptime(item['recordDate'], '%Y-%m-%d') for item in current_year_records]
        )
    
        if len(record_dates) > 1:
            total_days = (record_dates[-1] - record_dates[0]).days
            intervals = len(record_dates) - 1
            average_interval = total_days / intervals if intervals > 0 else None
            estimated_frequency = round(365 / average_interval) if average_interval and average_interval > 0 else len(record_dates)
        else:
            estimated_frequency = 52 if record_dates else 0  # Default to weekly if only one record exists
    
        quote_data = orjson.loads(df['quote'].iloc[0])[0]
        eps = quote_data.get('eps')
        current_price = quote_data.get('price')
    
        dividend_yield = round((previous_annual_dividend / current_price) * 100, 2) if current_price else None
        payout_ratio = round((1 - (eps - previous_annual_dividend) / eps) * 100, 2) if eps else None
        dividend_growth = None  # No calculation since we are strictly using the past year's data
    
        return {
            'payoutFrequency': estimated_frequency,
            'annualDividend': previous_annual_dividend,  # Strictly using past year’s data
            'dividendYield': dividend_yield,
            'payoutRatio': payout_ratio,
            'dividendGrowth': dividend_growth,
            'history': filtered_res,
        }
    
    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
        return {}


async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    total_symbols = stock_symbols + etf_symbols
    
    for ticker in tqdm(total_symbols):
        res = await get_data(ticker, con, etf_con, stock_symbols, etf_symbols)
        try:
            if len(res.get('history', [])) > 0:
                await save_as_json(ticker, res, 'json/dividends/companies')
        except Exception as e:
            print(f"Error saving data for {ticker}: {e}")
    
    con.close()
    etf_con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)
