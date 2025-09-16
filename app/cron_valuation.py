import orjson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict
import os

START_YEAR = 2015

async def save_as_json(symbol, data):
    directory = "json/valuation"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{symbol}.json", 'wb') as file:
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
        
        price_to_fcf_list = [item for item in price_to_fcf_list if int(item['date'][:4]) >= START_YEAR]

        five_year_avg = None
        if price_to_fcf_list:
            # Get the latest date and calculate 5 years back
            latest_date = max(item['date'] for item in price_to_fcf_list)
            five_years_ago = str(int(latest_date[:4]) - 5) + latest_date[4:]
            
            # Filter for last 5 years
            last_5_years = [
                item['priceToFCFRatio'] 
                for item in price_to_fcf_list 
                if item['date'] >= five_years_ago
            ]
            
            # Calculate average
            if last_5_years:
                five_year_avg = round(sum(last_5_years) / len(last_5_years), 2)
    
        return {
            'history': price_to_fcf_list,
            'five_year_average': five_year_avg
        }

    except Exception as e:
        print(f"Error computing per-share ratio for {symbol}: {e}")

    return None

async def get_data(symbol: str):
    # --- Cash flow (quarter) ---
    cf_path = f"json/financial-statements/cash-flow-statement/quarter/{symbol}.json"
    cf_raw = load_json(cf_path)

    cf_list = []
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

    # --- Historical adjusted prices (bi-weekly downsample) ---
    price_path = f"json/historical-price/adj/{symbol}.json"
    price_raw = load_json(price_path)

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

    # Group by (iso_year, biweekly_number)
    # biweekly_number = ceil(week / 2)
    biweekly_map = {}
    for dt, date_s, price in price_tuples:
        iso_year, iso_week, _ = dt.isocalendar()
        biweek = (iso_week + 1) // 2  # group weeks [1,2]=1, [3,4]=2, etc.
        key = (iso_year, biweek)
        # keep last (chronologically) entry within that 2-week block
        biweekly_map[key] = (dt, date_s, price)

    # Extract and sort
    biweekly_items = list(biweekly_map.values())
    biweekly_items.sort(key=lambda v: v[0])

    biweekly_list = [{"date": v[1], "price": v[2]} for v in biweekly_items]

    return {"freeCashFlowHistory": cf_list, "historicalPrice": biweekly_list}




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
        try:
            data = await get_data(symbol)

            shares_growth = compute_cagr(symbol, 'weightedAverageShsOutDil')
            free_cash_flow_growth = compute_cagr(symbol, 'freeCashFlow')
            dividend_growth = compute_cagr(symbol, 'dividends')
            price_ratio = compute_price_ratio(symbol)
            price_ratio_avg = price_ratio.get('five_year_average')
            price_ratio_history = price_ratio.get('history')


            ratio_dict = {item['date']: item['priceToFCFRatio'] for item in price_ratio_history}
            for price_entry in data['historicalPrice']:
                date = price_entry['date']
                if date in ratio_dict:
                    price_entry['priceToFCFRatio'] = ratio_dict[date]

        
            res = {'sharesGrowth': shares_growth, 'dividendGrowth': dividend_growth, "freeCashFlowGrowth": free_cash_flow_growth,
                    'priceRatioAvg': price_ratio_avg, **data}
            if res:
                await save_as_json(symbol, res)
        except:
            pass

try:
    asyncio.run(run())
except Exception as e:
    print(e)