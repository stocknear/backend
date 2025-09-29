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
        
        # Helper function to get date from record
        def get_date_from_record(record):
            """Get date from record, checking recordDate first, then date field"""
            # Try recordDate first
            if record.get('recordDate') and record['recordDate'].strip():
                return datetime.fromisoformat(record['recordDate'])
            # Fall back to date field
            elif record.get('date') and record['date'].strip():
                return datetime.fromisoformat(record['date'])
            else:
                return None
        
        # Filter out records that don't have any valid date
        filtered_res = []
        for item in res:
            if get_date_from_record(item) is not None:
                filtered_res.append(item)
        
        if not filtered_res:
            raise ValueError("No valid dividend records found.")

        # Extract payout frequency and dividend yield from the first valid record
        payout_frequency = filtered_res[0].get('frequency')
        dividend_yield = filtered_res[0].get('yield')

        # Determine the period for the last year using the maximum record date
        dates = [get_date_from_record(item) for item in filtered_res]
        valid_dates = [d for d in dates if d is not None]
        
        if not valid_dates:
            raise ValueError("No valid dates found in dividend records.")
        
        max_record_date = max(valid_dates)
        one_year_ago = max_record_date - timedelta(days=365)

        # Calculate dividend growth rate
        # Sort records by date
        sorted_records = sorted(
            filtered_res, 
            key=lambda x: get_date_from_record(x) or datetime.min
        )
        
        # Get the year of the latest dividend
        latest_date = get_date_from_record(sorted_records[-1])
        if latest_date is None:
            dividend_growth = None
        else:
            latest_year = latest_date.year
            
            # Find the first dividend in the current year and the first dividend from previous year
            latest_dividend = None
            previous_year_dividend = None
            
            for record in sorted_records:
                try:
                    record_date = get_date_from_record(record)
                    if record_date is None:
                        continue
                        
                    if record_date.year == latest_year and latest_dividend is None:
                        latest_dividend = record.get('adjDividend')
                    elif record_date.year == latest_year - 1 and previous_year_dividend is None:
                        previous_year_dividend = record.get('adjDividend')
                    
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
        annual_dividend = 0
        for item in filtered_res:
            item_date = get_date_from_record(item)
            if item_date and item_date >= one_year_ago:
                annual_dividend += item.get('adjDividend', 0)

        # Try to get payout ratio from quote data
        payout_ratio = None
        try:
            with open(f"json/quote/{ticker}.json", "r") as file:
                quote_data = orjson.loads(file.read())
                eps = quote_data.get('eps')
                if eps and eps != 0 and annual_dividend is not None:
                    payout_ratio = round((annual_dividend / eps) * 100, 2)
        except:
            pass
        
        return {
            'payoutFrequency': payout_frequency,
            'annualDividend': round(annual_dividend, 2) if annual_dividend else None,
            'dividendYield': round(dividend_yield, 2) if dividend_yield else None,
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
