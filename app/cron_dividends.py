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

def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

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
    
        # Dynamically compute the current and previous year based on New York timezone
        current_year = str(datetime.now(ny_tz).year)
        previous_year = str(datetime.now(ny_tz).year - 1)
    
        # Filter records for the current year
        current_year_records = [item for item in filtered_res if current_year in item['recordDate']]
        dividends_current_year = [float(item['adjDividend']) for item in current_year_records]
    
        # Compute the estimated payout frequency using the intervals between record dates
        record_dates = []
        for item in current_year_records:
            try:
                record_date = datetime.strptime(item['recordDate'], '%Y-%m-%d')
                record_dates.append(record_date)
            except Exception as e:
                continue
        record_dates.sort()
    
        if len(record_dates) > 1:
            total_days = (record_dates[-1] - record_dates[0]).days
            intervals = len(record_dates) - 1
            average_interval = total_days / intervals if intervals > 0 else None
            estimated_frequency = round(365 / average_interval) if average_interval and average_interval > 0 else len(record_dates)
        else:
            # If there's only one record, assume weekly (52 payments) as a fallback;
            # if no record exists, frequency remains 0.
            estimated_frequency = 52 if record_dates else 0
    
        # Project the annual dividend using the average dividend amount
        if dividends_current_year:
            avg_dividend = sum(dividends_current_year) / len(dividends_current_year)
            annual_dividend = round(avg_dividend * estimated_frequency, 2)
        else:
            annual_dividend = 0
    
        # For the previous year, assume the data is complete and sum the dividends
        dividends_previous_year = [
            float(item['adjDividend'])
            for item in filtered_res
            if previous_year in item['recordDate']
        ]
        previous_annual_dividend = round(sum(dividends_previous_year), 2) if dividends_previous_year else 0
    
        quote_data = orjson.loads(df['quote'].iloc[0])[0]
        eps = quote_data.get('eps')
        current_price = quote_data.get('price')
    
        dividend_yield = round((annual_dividend / current_price) * 100, 2) if current_price else None
        payout_ratio = round((1 - (eps - annual_dividend) / eps) * 100, 2) if eps else None
        dividend_growth = (
            round(((annual_dividend - previous_annual_dividend) / previous_annual_dividend) * 100, 2)
            if previous_annual_dividend else None
        )
    
        return {
            'payoutFrequency': estimated_frequency,
            'annualDividend': annual_dividend,
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
                print(res)
                await save_as_json(ticker, res, 'json/dividends/companies')
        except Exception as e:
            print(f"Error saving data for {ticker}: {e}")
    
    con.close()
    etf_con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)
