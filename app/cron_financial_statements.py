import os
import ujson
import asyncio
import aiohttp
import sqlite3
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

# Configurations
include_current_quarter = False
max_concurrent_requests = 100

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = 0
        self.last_reset = asyncio.get_event_loop().time()

    async def acquire(self):
        current_time = asyncio.get_event_loop().time()
        if current_time - self.last_reset >= self.time_window:
            self.requests = 0
            self.last_reset = current_time

        if self.requests >= self.max_requests:
            wait_time = self.time_window - (current_time - self.last_reset)
            if wait_time > 0:
                #print(f"\nRate limit reached. Waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                self.requests = 0
                self.last_reset = asyncio.get_event_loop().time()

        self.requests += 1

async def fetch_data(session, url, symbol, rate_limiter):
    await rate_limiter.acquire()
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                #print(f"Error fetching data for {symbol}: HTTP {response.status}")
                return None
    except Exception as e:
        #print(f"Exception during fetching data for {symbol}: {e}")
        return None

async def save_json(symbol, period, data_type, data):
    os.makedirs(f"json/financial-statements/{data_type}/{period}/", exist_ok=True)
    with open(f"json/financial-statements/{data_type}/{period}/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

async def add_ratio_elements(symbol):
    for period in ['annual', 'quarter']:
        try:

            # Load key-metrics data
            key_metrics_path = f"json/financial-statements/key-metrics/{period}/{symbol}.json"
            with open(key_metrics_path, "r") as file:
                key_metrics_data = ujson.load(file)

            # Load ratios data
            ratios_path = f"json/financial-statements/ratios/{period}/{symbol}.json"
            with open(ratios_path, "r") as file:
                ratio_data = ujson.load(file)

            if ratio_data and key_metrics_data:
                for ratio_item, key_metrics_item in zip(ratio_data,key_metrics_data):
                    try:
                        ratio_item['returnOnEquity'] = round(key_metrics_item.get('returnOnEquity',0),2)
                        ratio_item['returnOnAssets'] = round(key_metrics_item.get('returnOnAssets',0),2)
                        ratio_item['returnOnInvestedCapital'] = round(key_metrics_item.get('returnOnInvestedCapital',0),2)
                        ratio_item['evToSales'] = round(key_metrics_item.get('evToSales',0),2)
                        ratio_item['evToEBITDA'] = round(key_metrics_item.get('evToEBITDA',0),2)
                        ratio_item['evToFreeCashFlow'] = round(key_metrics_item.get('evToFreeCashFlow',0),2)
                        ratio_item['earningsYield'] = round(key_metrics_item.get('earningsYield',0),2)
                        ratio_item['freeCashFlowYield'] = round(key_metrics_item.get('freeCashFlowYield',0),2)

                    except:
                        pass

                with open(ratios_path, "w") as file:
                    ujson.dump(ratio_data, file)

        except Exception as e:
            print(f"Error calculating margins for {symbol}: {e}")

async def get_financial_statements(session, symbol, semaphore, rate_limiter):

    base_url = "https://financialmodelingprep.com/stable"
    periods = ['quarter', 'annual']
    financial_data_types = ['key-metrics', 'income-statement', 'balance-sheet-statement', 'cash-flow-statement', 'ratios']
    growth_data_types = ['income-statement-growth', 'balance-sheet-statement-growth', 'cash-flow-statement-growth']
    ttm_data_types = ['income-statement-ttm', 'balance-sheet-statement-ttm', 'cash-flow-statement-ttm', 'ratios-ttm']
    
    async with semaphore:
        for period in periods:
            # Fetch regular financial statements
            for data_type in financial_data_types:
                url = f"{base_url}/{data_type}/?symbol={symbol}&limit=2000&period={period}&apikey={api_key}"
                data = await fetch_data(session, url, symbol, rate_limiter)
                if data:
                    await save_json(symbol, period, data_type, data)
            
            # Fetch financial statement growth data
            for growth_type in growth_data_types:
                growth_url = f"{base_url}/{growth_type}/?symbol={symbol}&limit=2000&period={period}&apikey={api_key}"
                growth_data = await fetch_data(session, growth_url, symbol, rate_limiter)
                if growth_data:
                    await save_json(symbol, period, growth_type, growth_data)

        for ttm_type in ttm_data_types:
            url = f"{base_url}/{ttm_type}/?symbol={symbol}&limit=2000&apikey={api_key}"
            data = await fetch_data(session, url, symbol, rate_limiter)
            if data:
                await save_json(symbol, 'ttm', ttm_type.replace("-ttm",''), data)


        # Fetch TTM metrics
        url = f"https://financialmodelingprep.com/stable/key-metrics-ttm/?symbol={symbol}&limit=2000&apikey={api_key}"
        data = await fetch_data(session, url, symbol, rate_limiter)
        if data:
            await save_json(symbol, 'ttm', 'key-metrics', data)

        # Fetch owner earnings data
        owner_earnings_url = f"https://financialmodelingprep.com/stable/owner-earnings?symbol={symbol}&apikey={api_key}"
        owner_earnings_data = await fetch_data(session, owner_earnings_url, symbol, rate_limiter)
        if owner_earnings_data:
            await save_json(symbol, 'quarter', 'owner-earnings', owner_earnings_data)

    
    await add_ratio_elements(symbol)

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    total_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    rate_limiter = RateLimiter(max_requests=1000, time_window=60)
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in tqdm(total_symbols):
            task = asyncio.create_task(get_financial_statements(session, symbol, semaphore, rate_limiter))
            tasks.append(task)
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run())