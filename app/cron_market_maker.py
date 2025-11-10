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

start_date = datetime.today() - timedelta(180)
end_date = datetime.today() 
start_date = start_date.strftime("%Y-%m-%d")
end_date = end_date.strftime("%Y-%m-%d")

dataset_name = "weekly_summary"
filtered_columns_input = ['issueSymbolIdentifier', 'marketParticipantName', 'totalWeeklyTradeCount', 'totalWeeklyShareQuantity', 'totalNotionalSum', 'initialPublishedDate']
date_filter_inputs = [{'startDate': start_date, 'endDate': end_date, 'fieldName': 'initialPublishedDate'}]


def preserve_title_case(input_string):
    # Convert the input string to title case
    exceptions = ['LLC', 'LP', 'HRT', 'XTX', 'UBS']
    title_case_string = input_string.title()

    # Split the title case string into words
    words = title_case_string.split()

    # Check each word against the exceptions list and replace if necessary
    for i, word in enumerate(words):
        if word.upper() in exceptions:
            words[i] = word.upper()

    # Join the words back into a single string
    result_string = ' '.join(words)
    
    return result_string.replace('And', '&')


async def get_data(ticker):
    try:
        filters_input = {'issueSymbolIdentifier': [ticker]}

        df = finra_api_queries.retrieve_dataset(
            dataset_name,
            api_token,
            filtered_columns=filtered_columns_input,
            filters = filters_input,
            date_filter=date_filter_inputs)

        df = df.rename(columns={"initialPublishedDate": "date","marketParticipantName": "name", "issueSymbolIdentifier": "symbol"})
        df_copy = df.copy()
        #Create new dataset for top 10 market makers with the highest activity
        top_market_makers_df = df_copy.drop(['symbol','date'], axis=1)
        top_market_makers_df = top_market_makers_df.groupby(['name']).mean().reset_index()
        top_market_makers_df = top_market_makers_df.rename(columns={"totalWeeklyTradeCount": "avgWeeklyTradeCount","totalWeeklyShareQuantity": "avgWeeklyShareQuantity", "totalNotionalSum": "avgNotionalSum"})

        top_market_makers_list = top_market_makers_df.to_dict('records')
        top_market_makers_list = sorted(top_market_makers_list, key=lambda x: x['avgNotionalSum'], reverse=True)[0:10]
        for item in top_market_makers_list:
            item['name'] = preserve_title_case(item['name'])

        #Create new dataset for historical movements

        history_df = df_copy.drop(['symbol','name'], axis=1)
        history_df = history_df.groupby(['date']).sum().reset_index()
        history_data = history_df.to_dict('records')

        return {'topMarketMakers': top_market_makers_list, 'history': history_data}

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {}

async def save_json(symbol, data):
    # Use async file writing to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    path = f"json/market-maker/companies/{symbol}.json"
    await loop.run_in_executor(None, ujson.dump, data, open(path, 'w'))

async def process_ticker(ticker):
    data = await get_data(ticker)
    if len(data) > 0:
        await save_json(ticker, data)

async def run():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 1E9 AND symbol NOT LIKE '%.%'")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    con.close()
    etf_con.close()

    total_symbols = stocks_symbols #+ etf_symbols

    async with aiohttp.ClientSession() as session:
        tasks = [process_ticker(ticker) for ticker in total_symbols]
        
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
