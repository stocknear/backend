import aiohttp
import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import pytz
import orjson
import time
import os
from dotenv import load_dotenv


load_dotenv()

ny_tz = pytz.timezone('America/New_York')
today = datetime.now(ny_tz).replace(hour=0, minute=0, second=0, microsecond=0)
N_days_ago = today - timedelta(days=10)

api_key = os.getenv("FMP_API_KEY")
FREQUENCY_PAYOUTS = {
    'annual': 1,
    'annually': 1,
    'quarterly': 4,
    'semi-annual': 2,
    'semiannual': 2,
    'monthly': 12,
    'bi-monthly': 6,
    'bimonthly': 6,
    'biweekly': 26,
    'weekly': 52,
}
SPECIAL_FREQUENCIES = {'special', 'irregular'}


async def save_as_json(symbol, data, file_name):
    # Ensure the directory exists
    os.makedirs(file_name, exist_ok=True)
    file_path = os.path.join(file_name, f"{symbol}.json")
    with open(file_path, 'w') as file:
        ujson.dump(data, file)


async def get_data(ticker, session, semaphore, stock_symbols, etf_symbols):
    async with semaphore:
        try:
            url = f'https://financialmodelingprep.com/stable/dividends?symbol={ticker}&apikey={api_key}'

            async with session.get(url) as response:
                res = await response.json()

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
            records_with_dates = []
            for item in res:
                record_date = get_date_from_record(item)
                if record_date is not None:
                    filtered_res.append(item)
                    records_with_dates.append((item, record_date))

            if not filtered_res:
                raise ValueError("No valid dividend records found.")

            today_naive = today.replace(tzinfo=None)
            historical_records = [
                (item, record_date)
                for item, record_date in records_with_dates
                if record_date <= today_naive
            ]
            records_for_metrics = historical_records or records_with_dates

            # Extract payout frequency and dividend yield from the most recent record with a valid date
            most_recent_record = max(records_for_metrics, key=lambda x: x[1])[0]
            payout_frequency = most_recent_record.get('frequency')
            dividend_yield = res[0].get('yield')

            # Calculate dividend growth rate
            # Sort records by date
            dividend_growth = None
            if historical_records:
                sorted_historical_records = sorted(
                    historical_records,
                    key=lambda x: x[1]
                )
                latest_date = sorted_historical_records[-1][1]
                latest_year = latest_date.year

                # Find the first dividend in the current year and the first dividend from previous year
                latest_dividend = None
                previous_year_dividend = None

                for record, record_date in sorted_historical_records:
                    try:
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
            annual_dividend = None
            if historical_records:
                one_year_ago = today_naive - timedelta(days=365)
                sorted_desc = sorted(historical_records, key=lambda x: x[1], reverse=True)
                primary_freq_label = None
                for record, _ in sorted_desc:
                    freq_label = (record.get('frequency') or '').strip().lower()
                    if freq_label and freq_label not in SPECIAL_FREQUENCIES:
                        primary_freq_label = freq_label
                        break
                expected_count = FREQUENCY_PAYOUTS.get(primary_freq_label)
                annual_total = 0.0
                added_any = False
                regular_taken = 0
                for record, record_date in sorted_desc:
                    adj_div = record.get('adjDividend')
                    if not isinstance(adj_div, (int, float)):
                        continue
                    freq_label = (record.get('frequency') or '').strip().lower()
                    if freq_label in SPECIAL_FREQUENCIES:
                        if record_date >= one_year_ago:
                            annual_total += adj_div
                            added_any = True
                        continue
                    if expected_count is None:
                        if record_date >= one_year_ago:
                            annual_total += adj_div
                            added_any = True
                    else:
                        if (
                            freq_label == primary_freq_label
                            and regular_taken < expected_count
                            and record_date >= one_year_ago
                        ):
                            annual_total += adj_div
                            added_any = True
                            regular_taken += 1
                if added_any:
                    annual_dividend = annual_total

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
                'dividendYield': round(dividend_yield, 3) if dividend_yield else None,
                'payoutRatio': None if payout_ratio < 0 else payout_ratio,
                'dividendGrowth': dividend_growth,
                'history': filtered_res,
            }

        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")
            return {}


async def process_ticker(ticker, session, semaphore, con, etf_con, stock_symbols, etf_symbols):
    """Process a single ticker and save results"""
    try:
        res = await get_data(ticker, session, semaphore, stock_symbols, etf_symbols)
        if len(res.get('history', [])) > 0:
            await save_as_json(ticker, res, 'json/dividends/companies')
    except Exception as e:
        print(f"Error saving data for {ticker}: {e}")


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
    con.close()
    etf_con.close()

    total_symbols = stock_symbols + etf_symbols
    #testing
    #total_symbols = ['KHC']

    # Configuration for concurrent processing
    BATCH_SIZE = 300  # Process 300 tickers per batch
    MAX_CONCURRENT = 50  # Maximum concurrent requests

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Create a single session for all requests
    async with aiohttp.ClientSession() as session:
        # Process tickers in batches
        for i in tqdm(range(0, len(total_symbols), BATCH_SIZE)):
            batch = total_symbols[i:i + BATCH_SIZE]

            # Create tasks for all tickers in the batch
            tasks = [
                process_ticker(ticker, session, semaphore, con, etf_con, stock_symbols, etf_symbols)
                for ticker in batch
            ]

            # Run all tasks concurrently
            await asyncio.gather(*tasks)

            # Rate limiting: sleep after each batch (except the last one)
            if i + BATCH_SIZE < len(total_symbols):
                await asyncio.sleep(30)

try:
    asyncio.run(run())
except Exception as e:
    print(e)
