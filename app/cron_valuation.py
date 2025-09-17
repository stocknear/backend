import orjson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict
import os
from typing import List

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
        return []


def compute_cagr(symbol, key_element):
    years = 5
    if key_element == 'freeCashFlow' or key_element == 'operatingCashFlow':
        path = f"json/financial-statements/cash-flow-statement/annual/{symbol}.json"
    elif key_element == 'operatingIncome':
        path = f"json/financial-statements/income-statement/annual/{symbol}.json"
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

    # --- Special handling for dividends ---
    if key_element == 'dividends':
        history = data.get("history", [])
        yearly = defaultdict(float)
        latest_dt = None

        for row in history:
            try:
                dt = datetime.strptime(row["date"], "%Y-%m-%d")
                val = float(row.get("adjDividend") or row.get("dividend") or 0)
                yearly[dt.year] += val
                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt
            except Exception:
                continue

        if not yearly or latest_dt is None:
            return 0
        if latest_dt < datetime.today() - timedelta(days=365):
            return 0

        cleaned = [{"date": f"{y}-12-31", "dt": datetime(y, 12, 31), "val": v} 
                   for y, v in sorted(yearly.items()) if v > 0]
    else:
        # --- Generic handling ---
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


def compute_price_ratio(symbol, metric_type='freeCashFlow'):
    try:
        if metric_type == 'freeCashFlow':
            data_path = f"json/financial-statements/cash-flow-statement/ttm/{symbol}.json"
            metric_key = 'freeCashFlow'
            ratio_name = 'priceToFCFRatio'
        elif metric_type == 'operatingIncome':
            data_path = f"json/financial-statements/income-statement/ttm/{symbol}.json"
            metric_key = 'operatingIncome'
            ratio_name = 'priceToOperatingIncomeRatio'
        elif metric_type == 'operatingCashFlow':
            data_path = f"json/financial-statements/cash-flow-statement/ttm/{symbol}.json"
            metric_key = 'operatingCashFlow'
            ratio_name = 'priceToOCFRatio'
        else:
            return None

        metric_raw = load_json(data_path)
        metric_raw = sorted(metric_raw, key=lambda x: x.get("date", ""))

        mkt_cap_path = f"json/market-cap/companies/{symbol}.json"
        mkt_cap_raw = load_json(mkt_cap_path)
        mkt_cap_raw = sorted(mkt_cap_raw, key=lambda x: x.get("date", ""))

        ratio_list = []
        cf_index = 0
        current_val = None

        for mkt_item in mkt_cap_raw:
            mkt_date = mkt_item.get('date')
            mkt_cap = mkt_item.get('marketCap')

            if not mkt_date or not mkt_cap:
                continue

            while cf_index < len(metric_raw):
                cf_date = metric_raw[cf_index].get('date')
                if cf_date and cf_date <= mkt_date:
                    val = metric_raw[cf_index].get(metric_key)
                    if val is not None and val != 0:
                        current_val = val
                    cf_index += 1
                else:
                    break

            if current_val is not None and current_val != 0:
                try:
                    ratio = round(mkt_cap / current_val, 2)
                    ratio_list.append({
                        'date': mkt_date,
                        ratio_name: ratio
                    })
                except Exception as e:
                    print(f"Error calculating {ratio_name} for {mkt_date}: {e}")

        ratio_list = [item for item in ratio_list if int(item['date'][:4]) >= START_YEAR]

        five_year_avg = None
        if ratio_list:
            latest_date = max(item['date'] for item in ratio_list)
            five_years_ago = str(int(latest_date[:4]) - 5) + latest_date[4:]

            last_5_years = [
                list(item.values())[1]
                for item in ratio_list
                if item['date'] >= five_years_ago
            ]

            if last_5_years:
                five_year_avg = round(sum(last_5_years) / len(last_5_years), 2)

        return {
            'history': ratio_list,
            'five_year_average': five_year_avg,
            'ratio_name': ratio_name
        }

    except Exception as e:
        print(f"Error computing {metric_type} ratio for {symbol}: {e}")

    return None


async def get_data(symbol: str):
    # --- Free Cash Flow (quarter) ---
    cf_path = f"json/financial-statements/cash-flow-statement/quarter/{symbol}.json"
    cf_raw = load_json(cf_path)

    cf_list = []
    ocf_list = []
    for item in sorted(cf_raw, key=lambda x: x.get("date", "")):
        date_s = item.get("date")
        if not date_s:
            continue
        try:
            dt = datetime.strptime(date_s, "%Y-%m-%d")
        except ValueError:
            continue
        if dt.year >= START_YEAR:
            cf_list.append({"date": date_s, "freeCashFlow": item.get("freeCashFlow")})
            ocf_list.append({"date": date_s, "operatingCashFlow": item.get("operatingCashFlow")})

    # --- Operating Income (quarter) ---
    oi_path = f"json/financial-statements/income-statement/quarter/{symbol}.json"
    oi_raw = load_json(oi_path)

    oi_list = []
    for item in sorted(oi_raw, key=lambda x: x.get("date", "")):
        date_s = item.get("date")
        if not date_s:
            continue
        try:
            dt = datetime.strptime(date_s, "%Y-%m-%d")
        except ValueError:
            continue
        if dt.year >= START_YEAR:
            oi_list.append({"date": date_s, "operatingIncome": item.get("operatingIncome")})

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

    price_tuples.sort(key=lambda x: x[0])

    biweekly_map = {}
    for dt, date_s, price in price_tuples:
        iso_year, iso_week, _ = dt.isocalendar()
        biweek = (iso_week + 1) // 2
        key = (iso_year, biweek)
        biweekly_map[key] = (dt, date_s, price)

    biweekly_items = list(biweekly_map.values())
    biweekly_items.sort(key=lambda v: v[0])

    biweekly_list = [{"date": v[1], "price": v[2]} for v in biweekly_items]

    return {
        "freeCashFlowHistory": cf_list,
        "operatingCashFlowHistory": ocf_list,
        "operatingIncomeHistory": oi_list,
        "historicalPrice": biweekly_list,
    }


async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    # Testing mode
    stock_symbols = ['GOOG']
    con.close()

    for symbol in tqdm(stock_symbols):
        try:
            data = await get_data(symbol)

            shares_growth = compute_cagr(symbol, 'weightedAverageShsOutDil')
            free_cash_flow_growth = compute_cagr(symbol, 'freeCashFlow')
            operating_income_growth = compute_cagr(symbol, 'operatingIncome')
            operating_cash_flow_growth = compute_cagr(symbol, 'operatingCashFlow')
            dividend_growth = compute_cagr(symbol, 'dividends')
            print(operating_cash_flow_growth)
            free_cf_ratio = compute_price_ratio(symbol, 'freeCashFlow')
            oper_income_ratio = compute_price_ratio(symbol, 'operatingIncome')
            oper_cf_ratio = compute_price_ratio(symbol, 'operatingCashFlow')

            # Map history into prices
            ratio_dict_fcf = {item['date']: item['priceToFCFRatio'] for item in free_cf_ratio['history']}
            ratio_dict_oi  = {item['date']: item['priceToOperatingIncomeRatio'] for item in oper_income_ratio['history']}
            ratio_dict_ocf = {item['date']: item['priceToOCFRatio'] for item in oper_cf_ratio['history']}

            for price_entry in data['historicalPrice']:
                date = price_entry['date']
                if date in ratio_dict_fcf:
                    price_entry['priceToFCFRatio'] = ratio_dict_fcf[date]
                if date in ratio_dict_oi:
                    price_entry['priceToOperatingIncomeRatio'] = ratio_dict_oi[date]
                if date in ratio_dict_ocf:
                    price_entry['priceToOCFRatio'] = ratio_dict_ocf[date]

            res = {
                'sharesGrowth': shares_growth,
                'dividendGrowth': dividend_growth,
                "freeCashFlowGrowth": free_cash_flow_growth,
                "operatingIncomeGrowth": operating_income_growth,
                "operatingCashFlowGrowth": operating_cash_flow_growth,
                'priceRatioAvgFCF': free_cf_ratio['five_year_average'],
                'priceRatioAvgOI': oper_income_ratio['five_year_average'],
                'priceRatioAvgOCF': oper_cf_ratio['five_year_average'],
                **data
            }

            if res:
                await save_as_json(symbol, res)
        except Exception as e:
            print(f"Error with {symbol}: {e}")
            pass


try:
    asyncio.run(run())
except Exception as e:
    print(e)
