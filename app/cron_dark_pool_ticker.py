import ujson
import asyncio
import aiohttp
import sqlite3
from tqdm import tqdm
from datetime import datetime,timedelta
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from finra_api_queries import finra_api_queries

# Load environment variables
load_dotenv()
api_key = os.getenv('FINRA_API_KEY')
api_secret = os.getenv('FINRA_API_SECRET')
api_token = finra_api_queries.retrieve_api_token(finra_api_key_input=api_key, finra_api_secret_input=api_secret)

start_date = datetime.today() - timedelta(365)
end_date = datetime.today() 
start_date = start_date.strftime("%Y-%m-%d")
end_date = end_date.strftime("%Y-%m-%d")

dataset_name = "regsho_daily_shorts_volume"
filtered_columns_input = ['tradeReportDate', 'securitiesInformationProcessorSymbolIdentifier', 'shortParQuantity', 'shortExemptParQuantity', 'totalParQuantity']
date_filter_inputs = [{'startDate': start_date, 'endDate': end_date, 'fieldName': 'tradeReportDate'}]




async def get_data(ticker):
    try:
        filters_input = {'securitiesInformationProcessorSymbolIdentifier': [ticker]}

        df = finra_api_queries.retrieve_dataset(
            dataset_name,
            api_token,
            filtered_columns=filtered_columns_input,
            filters = filters_input,
            date_filter=date_filter_inputs)

        df = df.rename(columns={"tradeReportDate": "date","totalParQuantity": "totalVolume", "shortParQuantity": "shortVolume", "securitiesInformationProcessorSymbolIdentifier": "symbol", "shortExemptParQuantity": "shortExemptVolume"})
        summed_df = df.drop('symbol', axis=1).groupby('date').sum().reset_index()
        data = summed_df.to_dict('records')

        # Iterate through the list and calculate the percentages
        for record in data:
            total_volume = record["totalVolume"]
            short_volume = record["shortVolume"]
            short_exempt_volume = record["shortExemptVolume"]
            
            # Calculate percentages
            short_percent = round((short_volume / total_volume) * 100,2)
            short_exempt_percent = round((short_exempt_volume / total_volume) * 100,2)
            
            # Add new elements to the dictionary
            record["shortPercent"] = short_percent
            record["shortExemptPercent"] = short_exempt_percent

        return data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return []

async def save_json(symbol, data):
    # Use async file writing to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    path = f"json/dark-pool/companies/{symbol}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    await loop.run_in_executor(None, ujson.dump, data, open(path, 'w'))

async def process_ticker(ticker):
    data = await get_data(ticker)
    if len(data)>0:
        await save_json(ticker, data)

async def run():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    con.close()
    etf_con.close()

    total_symbols = stocks_symbols+ etf_symbols

    async with aiohttp.ClientSession() as session:
        tasks = []
        for ticker in total_symbols:
            tasks.append(process_ticker(ticker))
        
        # Run tasks concurrently in batches to avoid too many open connections
        batch_size = 10  # Adjust based on your system's capacity
        for i in tqdm(range(0, len(tasks), batch_size)):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(f"An error occurred: {e}")