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

load_dotenv()

ny_tz = pytz.timezone('America/New_York')
today = datetime.now(ny_tz).replace(hour=0, minute=0, second=0, microsecond=0)
N_days_ago = today - timedelta(days=10)


async def save_as_json(symbol, data, file_name):
    # Ensure the directory exists
    os.makedirs(file_name, exist_ok=True)
    file_path = os.path.join(file_name, f"{symbol}.json")
    with open(file_path, 'w') as file:
        ujson.dump(data, file)

async def get_data(ticker, con, etf_con, stock_symbols, etf_symbols):
    try:
        # Choose the appropriate table and column names
        if ticker in etf_symbols:
            table_name = 'etfs'
            column_name = 'etf_dividend'
        else:
            table_name = 'stocks'
            column_name = 'stock_dividend'

        # Build and execute the SQL query
        query_template = f"""
            SELECT {column_name}
            FROM {table_name}
            WHERE symbol = ?
        """
        df = pd.read_sql_query(
            query_template, 
            etf_con if table_name == 'etfs' else con,
            params=(ticker,)
        )

        # Load the JSON data
        res = orjson.loads(df[column_name].iloc[0])
        # Filter out records that do not have a recordDate or paymentDate
        filtered_res = [item for item in res if item['recordDate'] and item['paymentDate']]
        
        if not filtered_res:
            raise ValueError("No valid dividend records found.")

        # Extract payout frequency and dividend yield from the first valid record
        payout_frequency = filtered_res[0]['frequency']
        dividend_yield = filtered_res[0]['yield']

        # Determine the period for the last year using the maximum record date
        max_record_date = max(datetime.fromisoformat(item['recordDate']) for item in filtered_res)
        one_year_ago = max_record_date - timedelta(days=365)

        # Calculate dividend growth rate
        # Sort records by record date
        sorted_records = sorted(filtered_res, key=lambda x: datetime.fromisoformat(x['recordDate']))
        
        # Get the year of the latest dividend
        latest_year = datetime.fromisoformat(sorted_records[-1]['recordDate']).year
        
        # Find the first dividend in the current year and the first dividend from previous year
        latest_dividend = None
        previous_year_dividend = None
        
        for record in sorted_records:
            try:
                record_date = datetime.fromisoformat(record['recordDate'])
                if record_date.year == latest_year and latest_dividend is None:
                    latest_dividend = record['adjDividend']
                elif record_date.year == latest_year - 1 and previous_year_dividend is None:
                    previous_year_dividend = record['adjDividend']
                
                # Break if we found both dividends
                if latest_dividend is not None and previous_year_dividend is not None:
                    break
            except:
                pass
        
        # Calculate growth rate if both values exist
        dividend_growth = None
        if latest_dividend is not None and previous_year_dividend is not None and previous_year_dividend != 0:
            dividend_growth = round(((latest_dividend - previous_year_dividend) / previous_year_dividend) * 100, 2)

        # Sum up all adjDividend values for records in the last year
        annual_dividend = sum(
            item['adjDividend'] 
            for item in filtered_res 
            if datetime.fromisoformat(item['recordDate']) >= one_year_ago
        )

        with open(f"json/quote/{ticker}.json","r") as file:
            try:
                quote_data = orjson.loads(file.read())
                eps = quote_data['eps']
                payout_ratio = round((1 - (eps - annual_dividend) / eps) * 100, 2) if eps else None
            except:
                payout_ratio = None
        
        return {
            'payoutFrequency': payout_frequency,
            'annualDividend': round(annual_dividend,2) if annual_dividend != None else annual_dividend,
            'dividendYield': round(dividend_yield,2) if dividend_yield != None else dividend_yield,
            'payoutRatio': round(payout_ratio,2) if payout_ratio != None else payout_ratio,
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
        try:
            res = await get_data(ticker, con, etf_con, stock_symbols, etf_symbols)
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
