import orjson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict

START_YEAR = 2015

async def save_as_json(symbol, data):
    with open(f"json/valuation/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))


def load_json(path: str):
    try:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except FileNotFoundError:
        return []
    except Exception:
        # If you want logging, add it here.
        return []


def compute_cagr(symbol, key_element):
    years = 5
    if key_element == 'freeCashFlow':
        path = f"json/financial-statements/cash-flow-statement/annual/{symbol}.json"
    elif key_element == 'weightedAverageShsOutDil':
        path = f"json/financial-statements/income-statement/annual/{symbol}.json"
    elif key_element == 'dividends':
        path = f"json/dividends/companies/{symbol}.json"
    else:
        return 0

    try:
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
    except Exception:
        return 0

    # Special handling for dividends
    if key_element == 'dividends':
        history = data.get("history", [])
        yearly = defaultdict(float)
        latest_dt = None

        # sum up dividends by year
        for row in history:
            try:
                dt = datetime.strptime(row["date"], "%Y-%m-%d")
                val = float(row.get("adjDividend") or row.get("dividend") or 0)
                yearly[dt.year] += val
                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt
            except Exception:
                continue

        # if no dividends found
        if not yearly or latest_dt is None:
            return 0

        # check if last dividend is older than 1 year
        if latest_dt < datetime.today() - timedelta(days=365):
            return 0

        # build cleaned list from yearly totals
        cleaned = [{"date": f"{y}-12-31", "dt": datetime(y, 12, 31), "val": v} 
                   for y, v in sorted(yearly.items()) if v > 0]
    else:
        # Generic handling
        cleaned = []
        for row in sorted(data, key=lambda x: x.get("date", "")):
            date_s = row.get("date")
            if not date_s:
                continue
            try:
                dt = datetime.strptime(date_s, "%Y-%m-%d")
            except Exception:
                continue
            try:
                val = row.get(key_element)
                if val is None:
                    continue
                val = float(val)
            except Exception:
                continue
            cleaned.append({"date": date_s, "dt": dt, "val": val})

    if len(cleaned) < years + 1:
        return 0

    start = cleaned[-(years+1)]
    end = cleaned[-1]

    start_val = start["val"]
    end_val = end["val"]
    years_span = end["dt"].year - start["dt"].year

    cagr = None
    if start_val > 0 and years_span > 0:
        try:
            cagr = (end_val / start_val) ** (1.0 / years_span) - 1.0
        except Exception:
            cagr = 0

    def pct(x):
        return round(x * 100, 2) if x is not None else 0

    return pct(cagr)


def compute_price_ratio(symbol, metric_type = 'freeCashFlow'):
   
    try:
        
        cashflow_path = f"json/financial-statements/cash-flow-statement/ttm/{symbol}.json"
        cashflow_raw = load_json(cashflow_path)
        cashflow_raw = sorted(cashflow_raw, key=lambda x: x.get("date", ""))
        
        mkt_cap_path = f"json/market-cap/companies/{symbol}.json"
        mkt_cap_raw = load_json(mkt_cap_path)
        mkt_cap_raw = sorted(mkt_cap_raw, key=lambda x: x.get("date", ""))
        
        price_to_fcf_list = []
        
        # Keep track of the current cash flow index
        cf_index = 0
        current_fcf = None
        
        for mkt_item in mkt_cap_raw:
            mkt_date = mkt_item.get('date')
            mkt_cap = mkt_item.get('marketCap')
            
            if not mkt_date or not mkt_cap:
                continue
            
            # Update current_fcf if we've reached a new cash flow date
            while cf_index < len(cashflow_raw):
                cf_date = cashflow_raw[cf_index].get('date')
                if cf_date and cf_date <= mkt_date:
                    # Update to this cash flow value
                    fcf = cashflow_raw[cf_index].get('freeCashFlow')
                    if fcf is not None and fcf != 0:  # Avoid division by zero
                        current_fcf = fcf
                    cf_index += 1
                else:
                    # The next cash flow date is in the future, stop updating
                    break
            
            # Calculate ratio if we have a valid FCF
            if current_fcf is not None and current_fcf != 0:
                try:
                    price_to_fcf = round(mkt_cap / current_fcf, 2)
                    price_to_fcf_list.append({
                        'date': mkt_date, 
                        'priceToFCFRatio': price_to_fcf
                    })
                except Exception as e:
                    print(f"Error calculating ratio for {mkt_date}: {e}")
        
        print(price_to_fcf_list)
    

        '''
        # Load income statement for shares outstanding
        income_path = f"json/financial-statements/income-statement/ttm/{symbol}.json"
        with open(income_path, "rb") as f:
            income_data = orjson.loads(f.read())
        
        if isinstance(income_data, list) and income_data:
            income_data = income_data[0]
        
        shares = income_data.get("weightedAverageShsOutDil")
        if not shares:
            return None
        
        # Calculate per-share metrics and ratios
        if metric_type == 'fcf':
            cf_path = f"json/financial-statements/cash-flow-statement/ttm/{symbol}.json"
            with open(cf_path, "rb") as f:
                cf_data = orjson.loads(f.read())
            if isinstance(cf_data, list) and cf_data:
                cf_data = cf_data[0]
            fcf = cf_data.get("freeCashFlow")
            if fcf:
                fcf_per_share = fcf / shares
                return current_price / fcf_per_share if fcf_per_share > 0 else None
        
        elif metric_type == 'earnings':
            net_income = income_data.get("netIncome")
            if net_income:
                eps = net_income / shares
                return current_price / eps if eps > 0 else None
        
        elif metric_type == 'sales':
            revenue = income_data.get("revenue")
            if revenue:
                sales_per_share = revenue / shares
                return current_price / sales_per_share if sales_per_share > 0 else None
        
        elif metric_type == 'book':
            balance_path = f"json/financial-statements/balance-sheet-statement/quarter/{symbol}.json"
            with open(balance_path, "rb") as f:
                balance_data = orjson.loads(f.read())
            if balance_data:
                latest_balance = sorted(balance_data, key=lambda x: x.get("date", ""))[-1]
                book_value = latest_balance.get("totalStockholdersEquity")
                if book_value:
                    book_per_share = book_value / shares
                    return current_price / book_per_share if book_per_share > 0 else None
        '''
    
    except Exception as e:
        print(f"Error computing per-share ratio for {symbol}: {e}")
    
    return None

async def get_data(symbol: str):

    # --- Cash flow (quarter) ---
    cf_path = f"json/financial-statements/cash-flow-statement/quarter/{symbol}.json"
    cf_raw = load_json(cf_path)

    cf_list = []
    # sort by date string; invalid/missing dates are skipped
    for item in sorted(cf_raw, key=lambda x: x.get("date", "")):
        date_s = item.get("date")
        if not date_s:
            continue
        try:
            dt = datetime.strptime(date_s, "%Y-%m-%d")
        except ValueError:
            continue
        if dt.year >= START_YEAR:
            cf_list.append(
                {
                    "date": date_s,
                    "freeCashFlow": item.get("freeCashFlow"),
                }
            )

    # --- Historical adjusted prices (weekly downsample) ---
    price_path = f"json/historical-price/adj/{symbol}.json"
    price_raw = load_json(price_path)

    # Build list of (datetime, date_string, price) for valid rows
    price_tuples: List[tuple] = []
    for item in price_raw:
        date_s = item.get("date")
        if not date_s:
            continue
        try:
            dt = datetime.strptime(date_s, "%Y-%m-%d")
        except ValueError:
            continue
        if dt.year < START_YEAR:
            continue
        price = item.get("adjClose")
        if price is None:
            continue
        price_tuples.append((dt, date_s, price))

    # Ensure chronological order
    price_tuples.sort(key=lambda x: x[0])

    # Group by ISO year-week and keep the last (chronologically) entry in each week.
    # Using dict; later entries overwrite earlier ones so the last trading day remains.
    weekly_map = {}
    for dt, date_s, price in price_tuples:
        iso = dt.isocalendar()  # (iso_year, iso_week, iso_weekday) in Python >=3.8
        key = (iso[0], iso[1])  # iso_year, iso_week
        weekly_map[key] = (dt, date_s, price)

    # Extract and sort by the stored datetime
    weekly_items = [v for k, v in weekly_map.items()]
    weekly_items.sort(key=lambda v: v[0])

    weekly_list = [{"date": v[1], "price": v[2]} for v in weekly_items]

    return {"freeCashFlows": cf_list, "historicalPrice": weekly_list}



async def run():

    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    #stock_symbols = [row[0] for row in cursor.fetchall()]
    #Testing mode
    stock_symbols = ['GOOG']
    con.close()

    for symbol in tqdm(stock_symbols):
        data = await get_data(symbol)

        compute_price_ratio(symbol)
        #if len(shareholders_list) > 0:
        #    await save_as_json(symbol, shareholders_list)

    con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)