from datetime import datetime, timedelta
import ujson
import asyncio
import aiohttp
import os
from dotenv import load_dotenv
import sqlite3
import pandas as pd
from tqdm import tqdm

# Load environment variables
load_dotenv()
api_key = os.getenv('FMP_API_KEY')


query_template = """
    SELECT date, close
    FROM "{ticker}"
    WHERE date BETWEEN ? AND ?
"""

# Function to save JSON data
async def save_json(symbol, data):
    with open(f'json/fomc-impact/companies/{symbol}.json', 'w') as file:
        ujson.dump(data, file)

# Function to fetch data from the API
async def get_data(session, url):
    async with session.get(url) as response:
        data = await response.json()
        return data

async def get_fomc_data():
    fomc_data = []
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    async with aiohttp.ClientSession() as session:
        current_date = start_date
        while current_date < end_date:
            next_date = min(current_date + timedelta(days=10), end_date)
            start_str = current_date.strftime('%Y-%m-%d')
            end_str = next_date.strftime('%Y-%m-%d')
            
            url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={start_str}&to={end_str}&apikey={api_key}"
            data = await get_data(session, url)
            if data:
                # Filter for "FOMC Economic Projections" events
                fomc_events = [item for item in data if item.get('event') == "Fed Interest Rate Decision"]
                fomc_data.extend(fomc_events)
            
            # Move to the next 10-day period
            current_date = next_date

    filtered_data = [
    {
        'date': item['date'][0:10],
        'changePercentage': item['changePercentage'],
        'previous': item['previous'],
        'actual': item['actual'],
        'estimate': item['estimate']
    }
        for item in fomc_data
    ]

    filtered_data = sorted(filtered_data, key=lambda x: x['date'])

    return filtered_data


async def run():
    fomc_dates = await get_fomc_data()  # Assumed to return the list of dictionaries as provided
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    # Extracting the dates for filtering
    fomc_dates_list = [datetime.strptime(fomc['date'], '%Y-%m-%d').date() for fomc in fomc_dates]
    # Connect to SQLite databases
    stock_con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')
    stock_cursor = stock_con.cursor()
    stock_cursor.execute("PRAGMA journal_mode = wal")
    stock_cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND marketCap >= 500E6")
    stock_symbols = [row[0] for row in stock_cursor.fetchall()]
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    total_symbols = stock_symbols + etf_symbols
    
    for ticker in tqdm(total_symbols):
        try:
            query = query_template.format(ticker=ticker)
            connection = stock_con if ticker in stock_symbols else etf_con
            df_price = pd.read_sql_query(query, connection, params=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

            if len(df_price) > 150 and len(fomc_dates) > 0:
                # Convert 'date' column in df_price to datetime.date for comparison
                df_price['date'] = pd.to_datetime(df_price['date']).dt.date
                # Filter out every fifth row, unless the date is in fomc_dates
                filtered_df = df_price[
                    (df_price.index % 5 != 0) | (df_price['date'].isin(fomc_dates_list))
                ]
                filtered_df['date'] = filtered_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
                # Prepare the result with filtered data and original fomc_dates
                fomc_data_unique = {}
                for fomc in fomc_dates:
                    date = fomc['date']
                    if date not in fomc_data_unique:  # Check for duplicates
                        fomc_data_unique[date] = {
                            'date': date,
                            'changePercentage': fomc['changePercentage'],
                            'previous': fomc['previous'],
                            'actual': fomc['actual'],
                            'estimate': fomc['estimate']
                        }
                # Convert the unique FOMC data back to a list
                res = {
                    'fomcData': list(fomc_data_unique.values()),  # Ensure unique dates
                    'history': filtered_df.to_dict('records')
                }
                # Compute percentage changes for FOMC dates
                for i in range(len(res['fomcData'])):
                    current_fomc_date = res['fomcData'][i]['date']
                    current_price_row = filtered_df[filtered_df['date'] == current_fomc_date]
                    if i == len(res['fomcData']) - 1:
                        # This is the last FOMC date, so compare it to the last price in the dataframe
                        last_price_row = filtered_df.iloc[-1]
                        current_price = current_price_row['close'].values[0]
                        next_price = last_price_row['close']
                    else:
                        next_fomc_date = res['fomcData'][i + 1]['date']
                        next_price_row = filtered_df[filtered_df['date'] == next_fomc_date]
                        if not current_price_row.empty and not next_price_row.empty:
                            current_price = current_price_row['close'].values[0]
                            next_price = next_price_row['close'].values[0]
                    # Calculate the percentage change
                    percentage_change = ((next_price - current_price) / current_price) * 100
                    res['fomcData'][i]['changePercentage'] = round(percentage_change, 2)  # Update with the new change percentage
                await save_json(ticker, res)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
# Run the asyncio event loop
loop = asyncio.get_event_loop()
loop.run_until_complete(run())
