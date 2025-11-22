import orjson
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load environment variables if needed
load_dotenv()

today = datetime.today()
current_date = datetime.now().date()

def save_json(data, symbol):
    directory_path = f"json/options-historical-data/companies"
    os.makedirs(directory_path, exist_ok=True)
    with open(f"{directory_path}/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))

def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value

def safe_float(value):
    try:
        if value is None: 
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def load_price_lookup(symbol):
    price_lookup = {}
    try:
        path = f"json/historical-price/adj/{symbol}.json"
        with open(path, "rb") as file:
            price_data = orjson.loads(file.read())
        
        # Sort by date to ensure percentage calculation is correct
        price_data.sort(key=lambda x: x['date'])
        
        prev_close = None
        
        for entry in price_data:
            date_str = entry.get('date')
            close_price = entry.get('adjClose')
            
            if close_price is None:
                continue
                
            change_percent = 0.0
            if prev_close and prev_close > 0:
                change_percent = ((close_price - prev_close) / prev_close) * 100
            
            price_lookup[date_str] = {
                'close': close_price,
                'changesPercentage': round(change_percent, 2)
            }
            
            prev_close = close_price
            
    except (FileNotFoundError, ValueError, orjson.JSONDecodeError):
        return {}
        
    return price_lookup

def aggregate_data_by_date(symbol):
    # Pre-load price data
    price_lookup = load_price_lookup(symbol)
    simple_price_map = {k: v['close'] for k, v in price_lookup.items()}
    
    data_by_date = {}
    today_date = datetime.today().date()
    
    contract_dir = f"json/all-options-contracts/{symbol}"
    contract_list = get_contracts_from_directory(contract_dir)
    if not contract_list:
        return []

    for item in tqdm(contract_list, leave=False):
        try:
            file_path = os.path.join(contract_dir, f"{item}.json")
            with open(file_path, "rb") as file:
                data = orjson.loads(file.read())
            
            expiration_str = data['expiration']
            expiration_date = datetime.strptime(expiration_str, "%Y-%m-%d").date()
            
            if today_date > expiration_date:
                continue

            option_type = data.get('optionType')
            if option_type not in ['call', 'put']:
                continue
            
            is_call = option_type == 'call'
            
            for entry in data.get('history', []):
                try:
                    date_str = entry.get('date')
                    
                    # DTE Calculation
                    entry_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    days_to_expiry = (expiration_date - entry_date).days
                    
                    volume = safe_float(entry.get('volume'))
                    open_interest = safe_float(entry.get('open_interest'))
                    total_premium = safe_float(entry.get('total_premium'))
                    implied_volatility = safe_float(entry.get('implied_volatility'))
                    gamma = safe_float(entry.get('gamma'))
                    delta = safe_float(entry.get('delta'))

                    spot_price = simple_price_map.get(date_str)
                    
                    if spot_price:
                        gex = open_interest * gamma * spot_price
                        dex = open_interest * delta * spot_price
                    else:
                        gex = 0
                        dex = 0

                    if date_str not in data_by_date:
                        data_by_date[date_str] = {
                            "date": date_str,
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
                            # Change structure: Store IVs grouped by expiration date
                            # Key: Expiration Date, Value: List of IVs
                            "iv_expirations": {}, 
                            "iv_count": 0
                        }

                    daily_data = data_by_date[date_str]
                    type_prefix = 'call_' if is_call else 'put_'
                    
                    daily_data[f"{type_prefix}volume"] += int(volume)
                    daily_data[f"{type_prefix}open_interest"] += int(open_interest)
                    daily_data[f"{type_prefix}premium"] += int(total_premium)

                    if is_call:
                        daily_data[f"{type_prefix}gex"] += round(gex, 2)
                        daily_data[f"{type_prefix}dex"] += round(dex, 2)
                    else:
                        daily_data[f"{type_prefix}gex"] -= round(gex, 2)
                        daily_data[f"{type_prefix}dex"] += round(dex, 2)
                    
                    # --- LOGIC UPDATE: IV Aggregation (Median of Medians) ---
                    # Only collect IV if within 30 days window (0-30 inclusive)
                    if 0 <= days_to_expiry <= 30 and implied_volatility > 0:
                        if expiration_str not in daily_data["iv_expirations"]:
                            daily_data["iv_expirations"][expiration_str] = []
                        
                        # Store raw IV (we will multiply by 100 later)
                        daily_data["iv_expirations"][expiration_str].append(implied_volatility)
                        daily_data["iv_count"] += 1
                    
                except Exception:
                    continue
        
        except Exception:
            continue

    if not data_by_date:
        return []

    # --- Post-Processing Loop ---
    final_data = []
    
    for date_str, daily_data in data_by_date.items():
        try:
            # 1. Calculate Median IV (Median of Medians)
            expiration_medians = []
            
            for exp_date, iv_list in daily_data["iv_expirations"].items():
                if iv_list:
                    # Median for this specific expiration
                    median_for_exp = np.median(iv_list)
                    expiration_medians.append(median_for_exp)
            
            final_iv = None
            if expiration_medians:
                # Median of the expiration medians * 100 (To match File 2 scaling)
                final_iv = round(np.median(expiration_medians) * 100, 2)
            
            daily_data['iv'] = final_iv
            
            # Remove the heavy temporary dictionary
            del daily_data["iv_expirations"]

            # 2. Calculate Put/Call Ratio
            if daily_data["call_volume"] > 0:
                daily_data["putCallRatio"] = round(daily_data["put_volume"] / daily_data["call_volume"], 2)
            else:
                daily_data["putCallRatio"] = None
            
            final_data.append(daily_data)
            
        except Exception as e:
            pass

    # Convert to DataFrame for Rolling calculations
    df = pd.DataFrame(final_data)
    
    # Calculate IV Rank
    data = df.to_dict('records')
    data = sorted(data, key=lambda x: x['date'])
    data = calculate_iv_rank_for_all(data)
    
    return sorted(data, key=lambda x: x['date'], reverse=True)

def calculate_iv_rank_for_all(data):
    if not data:
        return []
    
    df = pd.DataFrame(data)
    
    if 'iv' not in df.columns or df['iv'].isnull().all():
        for entry in data:
            entry['iv_rank'] = None
        return data
    
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    
    # Rolling 365D Window
    rolling_min = df['iv'].rolling('365D', min_periods=1).min()
    rolling_max = df['iv'].rolling('365D', min_periods=1).max()
    
    df['rolling_min'] = rolling_min
    df['rolling_max'] = rolling_max
    
    # Calculation
    df['iv_rank'] = ((df['iv'] - df['rolling_min']) / (df['rolling_max'] - df['rolling_min'])) * 100
    
    # Clean up rounding and limits
    df['iv_rank'] = df['iv_rank'].round(2)
    df.loc[df['rolling_max'] == df['rolling_min'], 'iv_rank'] = 0.0  # Avoid division by zero or stuck max
    
    # None handling
    df['iv_rank'] = df['iv_rank'].where(pd.notnull(df['iv_rank']), None)
    
    df.drop(['rolling_min', 'rolling_max'], axis=1, inplace=True)
    
    df.reset_index(inplace=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    result = df.to_dict('records')
    
    return sorted(result, key=lambda x: x['date'], reverse=True)


def prepare_data(data, symbol):
    data = [entry for entry in data if entry['call_volume'] != 0 or entry['put_volume'] != 0 or entry['call_open_interest'] != 0]
    
    if not data:
        return

    price_lookup = load_price_lookup(symbol)

    res_list = []
    for item in data:
        try:
            new_item = {
                key: safe_round(value) if isinstance(value, (int, float, str)) else value
                for key, value in item.items()
            }

            new_item.update({
                'volume': new_item['call_volume'] + new_item['put_volume'],
                'putCallRatio': round(new_item['put_volume'] / new_item['call_volume'], 2) if new_item['call_volume'] > 0 else None,
                'total_premium': new_item['call_premium'] + new_item['put_premium'],
                'total_open_interest': new_item['call_open_interest'] + new_item['put_open_interest']
            })

            if price_data := price_lookup.get(item['date']):
                new_item['changesPercentage'] = float(price_data['changesPercentage'])
                new_item['price'] = float(price_data['close'])
            else:
                new_item['changesPercentage'] = None
                new_item['price'] = None

            res_list.append(new_item)
        except:
            continue

    if not res_list:
        return

    df = pd.DataFrame(res_list)
    df = df.sort_values('date')
    df['changeOI'] = df['total_open_interest'].diff()
    df['changesPercentageOI'] = (df['total_open_interest'].pct_change() * 100).round(2)
    
    res_list = df.sort_values('date', ascending=False).to_dict('records')
    
    if res_list:
        save_json(res_list, symbol)


def get_contracts_from_directory(directory: str):
    try:
        if not os.path.exists(directory):
            return []
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    except Exception as e:
        print(e)
        return []


def check_contract_expiry(symbol):
    directory = f"json/all-options-contracts/{symbol}/"
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        
        # Iterate through all JSON files in the directory
        for file in os.listdir(directory):
            try:
                if file.endswith(".json"):
                    contract_id = file.replace(".json", "")
                    
                    # Check if the contract has invalid expiration date
                    if not is_valid_expiration(contract_id):
                        # Delete the invalid contract JSON file
                        os.remove(os.path.join(directory, file))
                        try:
                            expiration_date = get_expiration_date(contract_id)
                            if expiration_date < current_date:
                                print(f"Deleted expired contract: {contract_id} (expired on {expiration_date})")
                        except:
                            print(f"Deleted invalid contract: {contract_id} (could not parse expiration)")
            except Exception as e:
                print(f"Error processing contract {file}: {e}")
        
        # Return the list of valid contracts
        valid_contracts = []
        for file in os.listdir(directory):
            if file.endswith(".json"):
                contract_id = file.replace(".json", "")
                if is_valid_expiration(contract_id):
                    valid_contracts.append(contract_id)
        
        return valid_contracts
    
    except Exception as e:
        print(f"Error checking contract expiry for {symbol}: {e}")
        return []

def is_valid_expiration(contract_id):
    """Check if a contract has a valid expiration date (not expired and not too far in future)"""
    try:
        expiration_date = get_expiration_date(contract_id)
        return current_date <= expiration_date
    except Exception:
        return False

def get_expiration_date(option_symbol):
    # Define regex pattern to match the symbol structure
    match = re.match(r"([A-Z]+)(\d{6})([CP])(\d+)", option_symbol)
    if not match:
        raise ValueError(f"Invalid option_symbol format: {option_symbol}")
    
    ticker, expiration, option_type, strike_price = match.groups()
    
    # Convert expiration to datetime
    date_expiration = datetime.strptime(expiration, "%y%m%d").date()
    return date_expiration

# Database connections
con = sqlite3.connect('stocks.db')
etf_con = sqlite3.connect('etf.db')
index_con = sqlite3.connect("index.db")

cursor = con.cursor()
cursor.execute("PRAGMA journal_mode = wal")
cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
stocks_symbols = [row[0] for row in cursor.fetchall()]

etf_cursor = etf_con.cursor()
etf_cursor.execute("PRAGMA journal_mode = wal")
etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
etf_symbols = [row[0] for row in etf_cursor.fetchall()]

index_cursor = index_con.cursor()
index_cursor.execute("PRAGMA journal_mode = wal")
index_cursor.execute("SELECT DISTINCT symbol FROM indices")
index_symbols = [row[0] for row in index_cursor.fetchall()]

total_symbols = stocks_symbols + etf_symbols + index_symbols

#testing
#total_symbols = ['NVDA']

# Main execution
for symbol in tqdm(total_symbols):
    try:
        #check_contract_expiry(symbol)
        data = aggregate_data_by_date(symbol)
        data = prepare_data(data, symbol)
    except Exception as e:
        pass

con.close()
etf_con.close()
index_con.close()