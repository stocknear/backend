import os
import ujson
import orjson
import asyncio
import aiohttp
import sqlite3
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

# Configurations
include_current_quarter = False
max_concurrent_requests = 100  # Limit concurrent requests

async def fetch_data(session, url, symbol, attempt=0):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                print(f"Error fetching data for {symbol}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Exception during fetching data for {symbol}: {e}")
        return None

async def save_json(symbol, period, data_type, data):
    os.makedirs(f"json/financial-statements/{data_type}/{period}/", exist_ok=True)
    with open(f"json/financial-statements/{data_type}/{period}/{symbol}.json", 'w') as file:
        ujson.dumps(data,file)

async def calculate_margins(symbol):
    for period in ['annual', 'quarter']:
        # Load income statement data
        income_path = f"json/financial-statements/income-statement/{period}/{symbol}.json"
        with open(income_path, "r") as file:
            income_data = orjson.loads(file.read())

        # Load cash flow statement data
        cash_flow_path = f"json/financial-statements/cash-flow-statement/{period}/{symbol}.json"
        with open(cash_flow_path, "r") as file:
            cash_flow_data = orjson.loads(file.read())

        # Load ratios data
        ratios_path = f"json/financial-statements/ratios/{period}/{symbol}.json"
        with open(ratios_path, "r") as file:
            ratio_data = orjson.loads(file.read())

        # Ensure all datasets are available and iterate through the items
        if income_data and cash_flow_data and ratio_data:
            for ratio_item, income_item, cash_flow_item in zip(ratio_data, income_data, cash_flow_data):
                # Extract required data
                revenue = income_item.get('revenue', 0)
                ebitda = income_item.get('ebitda',0)
                free_cash_flow = cash_flow_item.get('freeCashFlow', 0)

                # Calculate freeCashFlowMargin if data is valid
                if revenue != 0:  # Avoid division by zero
                    ratio_item['freeCashFlowMargin'] = round((free_cash_flow / revenue) * 100, 2)
                    ratio_item['ebitdaMargin'] = round((ebitda / revenue) * 100,2)
                    ratio_item['grossProfitMargin'] = round(ratio_item['grossProfitMargin']*100,2)
                    ratio_item['operatingProfitMargin'] = round(ratio_item['operatingProfitMargin']*100,2)
                    ratio_item['pretaxProfitMargin'] = round(ratio_item['pretaxProfitMargin']*100,2)
                    ratio_item['netProfitMargin'] = round(ratio_item['netProfitMargin']*100,2)
                else:
                    ratio_item['freeCashFlowMargin'] = None  # Handle missing or zero revenue
                    ratio_item['ebitdaMargin'] = None
                    ratio_item['grossProfitMargin'] = None
                    ratio_item['operatingProfitMargin'] = None
                    ratio_item['pretaxProfitMargin'] = None
                    ratio_item['netProfitMargin'] = None

            # Save the updated ratios data back to the JSON file
            with open(ratios_path, "wb") as file:
                file.write(orjson.dumps(data,option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8'))


async def get_financial_statements(session, symbol, semaphore, request_counter):
    base_url = "https://financialmodelingprep.com/api/v3"
    periods = ['quarter', 'annual']
    financial_data_types = ['key-metrics', 'income-statement', 'balance-sheet-statement', 'cash-flow-statement', 'ratios']
    growth_data_types = ['income-statement-growth', 'balance-sheet-statement-growth', 'cash-flow-statement-growth']
    
    async with semaphore:
        for period in periods:
            # Fetch regular financial statements
            for data_type in financial_data_types:
                url = f"{base_url}/{data_type}/{symbol}?period={period}&apikey={api_key}"
                data = await fetch_data(session, url, symbol)
                if data:
                    await save_json(symbol, period, data_type, data)
                
                request_counter[0] += 1  # Increment the request counter
                if request_counter[0] >= 500:
                    await asyncio.sleep(60)  # Pause for 60 seconds
                    request_counter[0] = 0  # Reset the request counter after the pause
            
            # Fetch financial statement growth data
            for growth_type in growth_data_types:
                growth_url = f"{base_url}/{growth_type}/{symbol}?period={period}&apikey={api_key}"
                growth_data = await fetch_data(session, growth_url, symbol)
                if growth_data:
                    await save_json(symbol, period, growth_type, growth_data)

                request_counter[0] += 1  # Increment the request counter
                if request_counter[0] >= 500:
                    await asyncio.sleep(60)  # Pause for 60 seconds
                    request_counter[0] = 0  # Reset the request counter after the pause


        url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}?apikey={api_key}"
        data = await fetch_data(session, url, symbol)
        if data:
            await save_json(symbol, 'ttm', 'key-metrics', data)

        # Fetch owner earnings data
        owner_earnings_url = f"https://financialmodelingprep.com/api/v4/owner_earnings?symbol={symbol}&apikey={api_key}"
        owner_earnings_data = await fetch_data(session, owner_earnings_url, symbol)
        if owner_earnings_data:
            await save_json(symbol, 'quarter', 'owner-earnings', owner_earnings_data)

        request_counter[0] += 1  # Increment the request counter
        if request_counter[0] >= 500:
            await asyncio.sleep(60)  # Pause for 60 seconds
            request_counter[0] = 0  # Reset the request counter after the pause
    
    await calculate_margins(symbol)

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    semaphore = asyncio.Semaphore(max_concurrent_requests)
    request_counter = [0]  # Using a list to keep a mutable counter across async tasks

    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in tqdm(symbols):
            task = asyncio.create_task(get_financial_statements(session, symbol, semaphore, request_counter))
            tasks.append(task)
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run())
