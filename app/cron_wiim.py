import aiohttp
import aiofiles
import ujson
import sqlite3
import pandas as pd
import asyncio
import pytz
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tqdm import tqdm
import pytz


date_format = "%a, %d %b %Y %H:%M:%S %z"
timezone = pytz.timezone("Europe/Berlin")

load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

headers = {"accept": "application/json"}

N_weeks_ago = datetime.now(pytz.UTC) - timedelta(weeks=50)

query_template = """
    SELECT
        close
    FROM
        "{symbol}"
    WHERE
        date BETWEEN ? AND ?
"""

# List of holidays when the stock market is closed
holidays = ['2025-01-01', '2025-01-09','2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25']


def is_holiday(date):
    """Check if the given date is a holiday"""
    str_date = date.strftime("%Y-%m-%d")
    return str_date in holidays

def correct_weekday(selected_date):
    # Monday is 0 and Sunday is 6
    if selected_date.weekday() == 0:
        selected_date -= timedelta(3)
    elif selected_date.weekday() <= 4:
        selected_date -= timedelta(1)
    elif selected_date.weekday() == 5:
        selected_date -= timedelta(1)
    elif selected_date.weekday() == 6:
        selected_date -= timedelta(2)
    
    # Check if the selected date is a holiday and adjust if necessary
    while is_holiday(selected_date):
        selected_date -= timedelta(1)
    
    # Adjust again if the resulting date is a Saturday or Sunday
    if selected_date.weekday() >= 5:
        selected_date -= timedelta(selected_date.weekday() - 4)
    
    return selected_date

# Create a semaphore to limit concurrent requests
REQUEST_LIMIT = 30
PAUSE_TIME = 1

def check_existing_file(symbol):
    file_path = f"json/wiim/company/{symbol}.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                existing_data = ujson.load(file)


            # Filter out elements older than two weeks
            updated_data = []
            for item in existing_data:
                try:
                    # Parse the date
                    date_obj = datetime.strptime(item['date'], "%Y-%m-%d %H:%M:%S")
                    if date_obj.tzinfo is None:
                        date_obj = date_obj.replace(tzinfo=pytz.UTC)

                    if date_obj >= N_weeks_ago:
                        updated_data.append(item)
                except Exception as e:
                    print(f"Error processing existing item: {e}")

            # Write back the filtered data
            if updated_data:
                with open(file_path, 'w') as file:
                    ujson.dump(updated_data, file)
                print(f"Updated existing file for {symbol}, removed old entries.")
            else:
                os.remove(file_path)
                print(f"Deleted file for {symbol} as all entries were older than two weeks.")
        except:
            pass

async def get_endpoint(session, symbol, con, semaphore):
    api_symbol = symbol.replace('-', '/')
    
    async with semaphore:
        url = "https://api.benzinga.com/api/v2/news"
        querystring = {
            "token": api_key,
            "tickers": api_symbol,
            "channels": "WIIM",
            "pageSize": "20",
            "sort":"created:desc",
        }
        
        try:
            async with session.get(url, params=querystring, headers=headers) as response:
                res_list = []
                res = ujson.loads(await response.text())
                
            
                
                for item in res:
                    try:
                        # Parse the date and ensure timezone-awareness
                        date_obj = datetime.strptime(item['created'], date_format)
                        if date_obj.tzinfo is None:
                            date_obj = date_obj.replace(tzinfo=pytz.UTC)
                        
                        # Skip items older than two weeks
                        if date_obj < N_weeks_ago:
                            continue

                        # Convert the date to New York timezone
                        date_obj_ny = date_obj.astimezone(timezone)
                        
                        start_date_obj_utc = correct_weekday(date_obj)
                        start_date = start_date_obj_utc.strftime("%Y-%m-%d")
                        end_date = date_obj.strftime("%Y-%m-%d")
                        new_date_str = date_obj_ny.strftime("%Y-%m-%d %H:%M:%S")
                        
                        query = query_template.format(symbol=symbol)
                        
                        try:
                            df = pd.read_sql_query(query, con, params=(start_date, end_date))
                            if not df.empty:
                                change_percent = round((df['close'].iloc[1] / df['close'].iloc[0] - 1) * 100, 2)
                            else:
                                change_percent = '-'
                        except:
                            change_percent = '-'
                        
                        res_list.append({
                            'date': new_date_str,
                            'text': item['title'],
                            'changesPercentage': change_percent
                        })
                    except:
                        pass
                
                if res_list:
                    print(f"Done processing {symbol}")
                    with open(f"json/wiim/company/{symbol}.json", 'w') as file:
                        ujson.dump(res_list, file)
                else:
                    check_existing_file(symbol)
                    
        except:
            pass

async def run():
    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    etf_con = sqlite3.connect('etf.db')

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]
    
    #stock_symbols = ['DIS']
    #etf_symbols = []

    # Create a semaphore to limit concurrent requests and implement rate limiting
    semaphore = asyncio.Semaphore(REQUEST_LIMIT)
    
    async with aiohttp.ClientSession() as session:
        # Combine stock and ETF symbols
        all_symbols = stock_symbols + etf_symbols
        
        # Split symbols into batches
        for i in range(0, len(all_symbols), REQUEST_LIMIT):
            batch = all_symbols[i:i+REQUEST_LIMIT]
            
            # Determine which symbols are stocks or ETFs
            batch_stocks = [s for s in batch if s in stock_symbols]
            batch_etfs = [s for s in batch if s in etf_symbols]
            
            # Process this batch
            tasks = []
            if batch_stocks:
                tasks.extend(get_endpoint(session, symbol, con, semaphore) for symbol in batch_stocks)
            if batch_etfs:
                tasks.extend(get_endpoint(session, symbol, etf_con, semaphore) for symbol in batch_etfs)
            
            # Wait for this batch to complete
            await asyncio.gather(*tasks)
            
            # If not the last batch, pause
            if i + REQUEST_LIMIT < len(all_symbols):
                print(f"Processed {i+REQUEST_LIMIT} symbols. Pausing for {PAUSE_TIME} seconds...")
                await asyncio.sleep(PAUSE_TIME)

    con.close()
    etf_con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)