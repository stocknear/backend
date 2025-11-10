import os
import ujson
import orjson
import asyncio
import aiohttp
import sqlite3
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from utils.helper import get_exchange_to_usd_factor 


load_dotenv()
api_key = os.getenv('FMP_API_KEY')

# Configurations
include_current_quarter = False
max_concurrent_requests = 100

cache_factor: dict[str, float] = {}


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

def get_historical_price_for_date(price_data, target_date):
    """
    Find the stock price on or before the target date (fiscal period end).
    This ensures we use the last available trading price of that period.
    price_data is sorted with most recent first.

    Args:
        price_data: List of price records sorted newest to oldest
        target_date: Target date in "YYYY-MM-DD" format (e.g., fiscal year end)

    Returns:
        The adjusted close price on the target date, or the closest trading day before it
    """
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")

    # Iterate through prices (newest to oldest) to find the last price on or before target date
    for price_item in price_data:
        price_date = datetime.strptime(price_item['date'], "%Y-%m-%d")
        if price_date <= target_dt:
            # This is the most recent price on or before the fiscal period end
            return price_item.get('adjClose')

    # If no price found before target date, return None
    return None



def calculate_historical_forward_pe(symbol, fiscal_year, period_end_date, analyst_estimates, price_data):
    """
    Calculate forward PE for a specific historical period.

    Args:
        symbol: Stock symbol
        fiscal_year: Fiscal year of the statement (e.g., 2024)
        period_end_date: End date of the fiscal period (e.g., "2024-09-30")
        analyst_estimates: List of analyst estimate data
        price_data: List of historical price data

    Returns:
        Forward PE ratio or None if cannot be calculated
    """
    try:
        # Get the stock price at the end of this fiscal period
        price = get_historical_price_for_date(price_data, period_end_date)
        if not price or price <= 0:
            return None

        currency_path = Path("json/financial-statements/income-statement/annual") / f"{symbol}.json"
        with currency_path.open('rb') as f:
            currency = orjson.loads(f.read())[0]['reportedCurrency']

        # Look for analyst estimate for the NEXT fiscal year
        next_year = int(fiscal_year) + 1
        estimate_item = next((item for item in analyst_estimates if item.get('date') == next_year), None)

        if not estimate_item:
            return None

        estimated_eps = estimate_item.get('estimatedEpsAvg')
        if not estimated_eps or estimated_eps <= 0:
            return None

        factor = cache_factor.get(currency)
        if factor is None:
            factor = get_exchange_to_usd_factor(currency, cache_factor)

        if estimated_eps and factor:
            eps_usd = float(estimated_eps) * float(factor)
            if eps_usd:
                forward_pe = float(price) / eps_usd
                forward_pe = round(forward_pe, 2) if forward_pe != None else None
                print(forward_pe)
                return forward_pe

    except Exception as e:
        print(e)
        return None


async def add_ratio_elements(symbol):
    # Load analyst estimates data once for all periods
    analyst_estimates = []
    price_data = []

    try:
        analyst_estimates_path = f"json/analyst-estimate/{symbol}.json"
        with open(analyst_estimates_path, "r") as file:
            analyst_estimates = ujson.load(file)
    except Exception as e:
        # No analyst estimates available for this symbol
        pass

    try:
        price_data_path = f"json/historical-price/adj/{symbol}.json"
        with open(price_data_path, "r") as file:
            price_data = ujson.load(file)
    except Exception as e:
        # No price data available for this symbol
        pass

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

            # Load income statement data for revenue
            income_statement_path = f"json/financial-statements/income-statement/{period}/{symbol}.json"
            with open(income_statement_path, "r") as file:
                income_statement_data = ujson.load(file)

            # Load cash flow statement data for free cash flow
            cash_flow_path = f"json/financial-statements/cash-flow-statement/{period}/{symbol}.json"
            with open(cash_flow_path, "r") as file:
                cash_flow_data = ujson.load(file)

            if ratio_data and key_metrics_data:
                for i, (ratio_item, key_metrics_item) in enumerate(zip(ratio_data,key_metrics_data)):
                    try:
                        ratio_item['returnOnEquity'] = round(key_metrics_item.get('returnOnEquity',0),4)
                        ratio_item['returnOnAssets'] = round(key_metrics_item.get('returnOnAssets',0),4)
                        ratio_item['returnOnInvestedCapital'] = round(key_metrics_item.get('returnOnInvestedCapital',0),4)
                        ratio_item['returnOnTangibleAssets'] = round(key_metrics_item.get('returnOnTangibleAssets',0),4)
                        ratio_item['returnOnCapitalEmployed'] = round(key_metrics_item.get('returnOnCapitalEmployed',0),4)

                        ratio_item['evToSales'] = round(key_metrics_item.get('evToSales',0),2)
                        ratio_item['evToEBITDA'] = round(key_metrics_item.get('evToEBITDA',0),2)
                        ratio_item['evToFreeCashFlow'] = round(key_metrics_item.get('evToFreeCashFlow',0),2)
                        ratio_item['earningsYield'] = round(key_metrics_item.get('earningsYield',0),4)
                        ratio_item['freeCashFlowYield'] = round(key_metrics_item.get('freeCashFlowYield',0),4)

                        # Calculate freeCashFlowMargin
                        if i < len(income_statement_data) and i < len(cash_flow_data):
                            revenue = income_statement_data[i].get('revenue', 0)
                            free_cash_flow = cash_flow_data[i].get('freeCashFlow', 0)

                            if revenue and revenue != 0:
                                ratio_item['freeCashFlowMargin'] = round(free_cash_flow / revenue, 2)
                            else:
                                ratio_item['freeCashFlowMargin'] = 0

                        # Calculate historical forwardPE if data is available
                        if analyst_estimates and price_data:
                            fiscal_year = ratio_item.get('fiscalYear')
                            period_end_date = ratio_item.get('date')

                            if fiscal_year and period_end_date:
                                forward_pe = calculate_historical_forward_pe(
                                    symbol,
                                    fiscal_year,
                                    period_end_date,
                                    analyst_estimates,
                                    price_data
                                )
                                if forward_pe is not None:
                                    ratio_item['forwardPE'] = forward_pe

                    except:
                        pass

                with open(ratios_path, "w") as file:
                    ujson.dump(ratio_data, file)

        except Exception as e:
            print(f"Error calculating margins for {symbol}: {e}")


async def add_balance_sheet_elements(symbol):
    for period in ["annual", "quarter", "ttm"]:
        try:
            # Load balance sheet data
            path = f"json/financial-statements/balance-sheet-statement/{period}/{symbol}.json"
            with open(path, "r") as file:
                balance_sheet_data = ujson.load(file)

            # Load income statement data
            path = f"json/financial-statements/income-statement/{period}/{symbol}.json"
            with open(path, "r") as file:
                income_data = ujson.load(file)

            if balance_sheet_data and income_data:
                for balance_item, income_item in zip(balance_sheet_data, income_data):
                    try:
                        total_assets = balance_item.get("totalAssets", 0) or 0
                        total_liabilities = balance_item.get("totalLiabilities", 0) or 0
                        shares_out = income_item.get("weightedAverageShsOut", 0) or 0

                        balance_item["bookValue"] = total_assets - total_liabilities
                        balance_item["bookValuePerShare"] = (
                            balance_item["bookValue"] / shares_out if shares_out > 0 else None
                        )
                    except Exception as inner_e:
                        print(f"Error processing item for {symbol}: {inner_e}")

                # Save updated data
                ratios_path = f"json/financial-statements/balance-sheet-statement/{period}/{symbol}.json"
                with open(ratios_path, "w") as file:
                    ujson.dump(balance_sheet_data, file)

        except Exception as e:
            print(f"Error calculating book values for {symbol} ({period}): {e}")


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
    await add_balance_sheet_elements(symbol)

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    total_symbols = [row[0] for row in cursor.fetchall()]
    #total_symbols = ['TSLA','NVO']
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