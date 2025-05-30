import orjson
from datetime import datetime,timedelta
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
from tqdm import tqdm
from collections import defaultdict



today = datetime.today()
#N_days_ago = today - timedelta(days=90)

query_template = """
    SELECT date, close, change_percent
    FROM "{ticker}"
    WHERE date BETWEEN ? AND ?
"""


def save_json(data, symbol):
    directory_path = f"json/options-historical-data/companies"
    os.makedirs(directory_path, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory_path}/{symbol}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))

def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value


def aggregate_data_by_date(symbol):
    # Pre-load price data and create lookup dictionary for better performance
    with open(f"json/historical-price/max/{symbol}.json", "r") as file:
        price_list = {p['time']: p['close'] for p in orjson.loads(file.read())}
    
    # Use dict instead of defaultdict for better performance
    data_by_date = {}
    
    today = datetime.today().date()
    one_year_ago = today - timedelta(days=365*5)
    one_year_ago_str = one_year_ago.strftime('%Y-%m-%d')
    
    contract_dir = f"json/all-options-contracts/{symbol}"
    contract_list = get_contracts_from_directory(contract_dir)
    if not contract_list:
        return []

    for item in tqdm(contract_list):
        try:
            file_path = os.path.join(contract_dir, f"{item}.json")
            with open(file_path, "r") as file:
                data = orjson.loads(file.read())
            
            expiration_date = datetime.strptime(data['expiration'], "%Y-%m-%d").date()
            if today > expiration_date:
                continue

            option_type = data.get('optionType')
            if option_type not in ['call', 'put']:
                continue
            
            is_call = option_type == 'call'
            
            for entry in data.get('history', []):
                try:
                    date = entry.get('date')
                    #if date < one_year_ago_str:
                    #    continue
                    
                    spot_price = price_list.get(date)
                    if not spot_price:
                        continue

                    volume = entry.get('volume', 0) or 0
                    open_interest = entry.get('open_interest', 0) or 0
                    total_premium = entry.get('total_premium', 0) or 0
                    implied_volatility = entry.get('implied_volatility', 0) or 0
                    gamma = entry.get('gamma', 0) or 0
                    delta = entry.get('delta', 0) or 0

                    gex = open_interest * gamma * spot_price
                    dex = open_interest * delta * spot_price

                    if date not in data_by_date:
                        data_by_date[date] = {
                            "date": date,
                            "call_volume": 0,
                            "put_volume": 0,
                            "call_open_interest": 0,
                            "put_open_interest": 0,
                            "call_premium": 0,
                            "put_premium": 0,
                            "call_gex": 0,
                            "put_gex": 0,
                            "call_dex": 0,
                            "put_dex": 0,
                            "iv": [],
                            "iv_count": 0,
                        }

                    daily_data = data_by_date[date]
                    
                    # Use conditional indexing instead of if-else
                    type_prefix = 'call_' if is_call else 'put_'
                    daily_data[f"{type_prefix}volume"] += int(volume)
                    daily_data[f"{type_prefix}open_interest"] += int(open_interest)
                    daily_data[f"{type_prefix}premium"] += int(total_premium)

                    if type_prefix == 'call_':
                        daily_data[f"{type_prefix}gex"] += round(gex, 2)
                        daily_data[f"{type_prefix}dex"] += round(dex, 2)
                    else:
                        daily_data[f"{type_prefix}gex"] -= round(gex, 2)
                        daily_data[f"{type_prefix}dex"] += round(dex, 2)
                    
                    daily_data["iv"].append(round(implied_volatility, 2))
                    daily_data["iv_count"] += 1
                    
                    try:
                        daily_data["putCallRatio"] = round(daily_data["put_volume"] / daily_data["call_volume"], 2)
                    except ZeroDivisionError:
                        daily_data["putCallRatio"] = None
                except:
                    pass
        
        except:
            continue

    # Convert to list and calculate median IV
    data = list(data_by_date.values())
    
    # Use vectorized operations with pandas for IV calculations
    df = pd.DataFrame(data)
    df['iv'] = df.apply(lambda x: round(float(pd.Series(x['iv']).median()), 2) if x['iv_count'] > 0 else None, axis=1)
    
    # Sort and calculate IV Rank
    data = df.to_dict('records')
    data = sorted(data, key=lambda x: x['date'])
    data = calculate_iv_rank_for_all(data)
    return sorted(data, key=lambda x: x['date'], reverse=True)

def calculate_iv_rank_for_all(data):
    if not data:
        return []
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Check if 'iv' exists and filter out entries without IV
    if 'iv' not in df.columns or df['iv'].isnull().all():
        for entry in data:
            entry['iv_rank'] = None
        return data
    
    # Convert date to datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    # Calculate rolling 365-day min and max for IV
    df.set_index('date', inplace=True)
    rolling_min = df['iv'].rolling('365D', min_periods=1).min()
    rolling_max = df['iv'].rolling('365D', min_periods=1).max()
    
    # Merge back into DataFrame
    df['rolling_min'] = rolling_min
    df['rolling_max'] = rolling_max
    
    # Calculate IV Rank
    df['iv_rank'] = ((df['iv'] - df['rolling_min']) / (df['rolling_max'] - df['rolling_min'])) * 100
    df['iv_rank'] = df['iv_rank'].round(2)
    
    # Handle cases where max == min
    df.loc[df['rolling_max'] == df['rolling_min'], 'iv_rank'] = 100.0
    
    # Replace NaN with None
    df['iv_rank'] = df['iv_rank'].where(pd.notnull(df['iv_rank']), None)
    
    # Drop temporary columns
    df.drop(['rolling_min', 'rolling_max'], axis=1, inplace=True)
    
    # Convert back to list of dicts
    df.reset_index(inplace=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    result = df.to_dict('records')
    
    # Sort in reverse chronological order
    result = sorted(result, key=lambda x: x['date'], reverse=True)
    
    return result


def prepare_data(data, symbol):
    # Filter data first to reduce processing
    data = [entry for entry in data if entry['call_volume'] != 0 or entry['put_volume'] != 0]
    if not data:
        return
    
    start_date_str = data[-1]['date']
    end_date_str = data[0]['date']

    # Determine query connection
    query_con = (con if symbol in stocks_symbols else 
                etf_con if symbol in etf_symbols else 
                index_con)

    # Use pandas efficient reading and processing
    df_price = pd.read_sql_query(
        query_template.format(ticker=symbol),
        query_con,
        params=(start_date_str, end_date_str)
    ).round(2)

    df_price = df_price.rename(columns={"change_percent": "changesPercentage"})
    price_lookup = df_price.set_index('date').to_dict('index')

    res_list = []
    for item in data:
        try:
            new_item = {
                key: safe_round(value) if isinstance(value, (int, float, str)) else value
                for key, value in item.items()
            }

            # Calculate derived fields
            new_item.update({
                'volume': new_item['call_volume'] + new_item['put_volume'],
                'putCallRatio': round(new_item['put_volume'] / new_item['call_volume'], 2),
                'total_premium': new_item['call_premium'] + new_item['put_premium'],
                'total_open_interest': new_item['call_open_interest'] + new_item['put_open_interest']
            })

            # Get price data from lookup
            if price_data := price_lookup.get(item['date']):
                new_item['changesPercentage'] = float(price_data['changesPercentage'])
                new_item['price'] = float(price_data['close'])
            else:
                new_item['changesPercentage'] = None
                new_item['price'] = None

            res_list.append(new_item)
        except:
            continue

    # Calculate OI changes using vectorized operations
    df = pd.DataFrame(res_list)
    df = df.sort_values('date')
    df['changeOI'] = df['total_open_interest'].diff()
    df['changesPercentageOI'] = (df['total_open_interest'].pct_change() * 100).round(2)
    
    res_list = df.sort_values('date', ascending=False).to_dict('records')
    
    if res_list:
        save_json(res_list, symbol)


def get_contracts_from_directory(directory: str):
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            return []
        
        # Get all tickers from filenames
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    
    except Exception as e:
        print(e)
        return []





# Connect to the databases
con = sqlite3.connect('stocks.db')
etf_con = sqlite3.connect('etf.db')
index_con = sqlite3.connect("index.db")

cursor = con.cursor()
cursor.execute("PRAGMA journal_mode = wal")
#cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND marketCap > 1E9")
cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
stocks_symbols = [row[0] for row in cursor.fetchall()]

etf_cursor = etf_con.cursor()
etf_cursor.execute("PRAGMA journal_mode = wal")
#etf_cursor.execute("SELECT DISTINCT symbol FROM etfs WHERE marketCap > 1E9")
etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
etf_symbols = [row[0] for row in etf_cursor.fetchall()]


index_cursor = index_con.cursor()
index_cursor.execute("PRAGMA journal_mode = wal")
index_cursor.execute("SELECT DISTINCT symbol FROM indices")
index_symbols = [row[0] for row in index_cursor.fetchall()]

total_symbols = stocks_symbols + etf_symbols + index_symbols
#Testing mode
#total_symbols = ['AUR']

for symbol in tqdm(total_symbols):
    try:
        data = aggregate_data_by_date(symbol)
        data = prepare_data(data, symbol)
    except:
        pass

con.close()
etf_con.close()
index_con.close()