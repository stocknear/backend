from datetime import datetime, date, timedelta
from collections import defaultdict
from tqdm import tqdm
import sqlite3
import orjson
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import math

today = date.today()


def save_json(data, symbol):
    directory_path="json/options-chain-statistics/"
    os.makedirs(directory_path, exist_ok=True)
    filepath = os.path.join(directory_path, f"{symbol}.json")
    with open(filepath, "wb") as f:
        f.write(orjson.dumps(data))

def get_contracts_from_directory(directory: str):
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith(".json")]

def safe_div(a, b):
    return round(a / b, 2) if b else 0

# Format dates 
def format_date(date_str):
    if not date_str:
        return None
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%b %d, %Y")
    except Exception:
        return date_str

def calculate_iv_rank(current_iv, historical_ivs):
    """Calculate IV rank - (current IV - min IV) / (max IV - min IV) * 100"""
    if not historical_ivs or current_iv == 0:
        return 0
    min_iv = min(historical_ivs)
    max_iv = max(historical_ivs)
    if max_iv == min_iv:
        return 0
    iv_rank = round(((current_iv - min_iv) / (max_iv - min_iv)) * 100, 2)
    if iv_rank < 0:
        iv_rank = None
    return iv_rank

def calculate_iv_percentile(current_iv, historical_ivs):
    """Calculate IV percentile - percentage of days with IV below current IV"""
    if not historical_ivs or current_iv == 0:
        return 0
    below_count = sum(1 for iv in historical_ivs if iv < current_iv)
    return round((below_count / len(historical_ivs)) * 100, 2)

def find_iv_extremes(historical_ivs):
    """Find IV high and low with dates"""
    if not historical_ivs:
        return 0, None, 0, None
    
    iv_high = max(historical_ivs, key=lambda x: x['iv'])
    iv_low = min(historical_ivs, key=lambda x: x['iv'])
    
    
    return (
        round(iv_high['iv'] * 100, 2), 
        format_date(iv_high['date']),
        round(iv_low['iv'] * 100, 2), 
        format_date(iv_low['date'])
    )

def calculate_historical_volatility(symbol, lookback_days=30):
    path = os.path.join("json", "historical-price", "max", f"{symbol}.json")
    try:
        with open(path, "rb") as f:
            prices = orjson.loads(f.read())
    except FileNotFoundError:
        print(f"No price file for {symbol} at {path}")
        return 0.0

    # Ensure we have enough data points
    if len(prices) < 2:
        return 0.0

    # Take the last lookback_days + 1 records (so we get lookback_days returns)
    slice_start = max(0, len(prices) - (lookback_days + 1))
    window = prices[slice_start:]

    # Compute log returns of 'close'
    log_returns = []
    for prev, curr in zip(window, window[1:]):
        p0 = prev.get("close")
        p1 = curr.get("close")
        if p0 and p1 and p0 > 0:
            log_returns.append(math.log(p1 / p0))
    if len(log_returns) < 2:
        return 0.0

    # Daily volatility = stdev of log returns
    daily_vol = statistics.stdev(log_returns)

    # Annualize (√252 trading days) and convert to percentage
    annual_vol_pct = daily_vol * math.sqrt(252) * 100
    return round(annual_vol_pct, 2)

def get_sentiment_from_pc_ratio(pc_ratio):
    """Determine market sentiment based on put/call ratio"""
    if pc_ratio < 0.7:
        return "bullish"
    elif pc_ratio > 1.3:
        return "bearish"
    else:
        return "neutral"

def calculate_historical_iv_stats(symbol, lookback_days=365):
  
    base_dir = os.path.join("json/all-options-contracts", symbol)
    contract_files = get_contracts_from_directory(base_dir)
    
    cutoff = today - timedelta(days=lookback_days)
    # temp storage: date_str -> [iv1, iv2, ...]
    ivs_by_date = defaultdict(list)

    for path in contract_files:
        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())

            history = data.get('history')
            for entry in history:
                try:
                    date = entry.get("date")
                    d = datetime.strptime(date, "%Y-%m-%d").date()
                    if d <= cutoff:
                        continue
                    iv = entry.get("implied_volatility", None) or None
                    if iv and iv >= 0:
                        ivs_by_date[date].append(iv)
                except Exception as e:
                    print(e)
        except Exception:
            continue
    # Now build sorted lists
    historical_ivs       = []
    historical_with_dates = []
    for date, ivs in sorted(ivs_by_date.items()):
        avg_iv = statistics.mean(ivs)
        historical_ivs.append(avg_iv)
        historical_with_dates.append({
            "date": date,
            "iv":   avg_iv
        })

    return historical_ivs, historical_with_dates


def calculate_average_daily_volume(symbol, lookback_days=30):
    """Calculate average daily volume over the past N days"""
    base_dir = os.path.join("json/all-options-contracts", symbol)
    contract_files = get_contracts_from_directory(base_dir)
    
    cutoff_date = today - timedelta(days=lookback_days)
    daily_volumes = defaultdict(int)
    
    for filepath in contract_files:
        try:
            with open(filepath, "rb") as f:
                data = orjson.loads(f.read())
            
            history = data.get("history", [])
            for entry in history:
                entry_date_str = entry.get("date")
                if entry_date_str:
                    entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
                    if entry_date >= cutoff_date:
                        volume = entry.get("volume", 0) or 0
                        daily_volumes[entry_date_str] += volume
        except Exception:
            continue
    
    if not daily_volumes:
        return 0
    
    return round(statistics.mean(daily_volumes.values()), 0)

def calculate_average_daily_oi(symbol, lookback_days=30):
    """Calculate average daily open interest over the past N days"""
    base_dir = os.path.join("json/all-options-contracts", symbol)
    contract_files = get_contracts_from_directory(base_dir)
    
    cutoff_date = today - timedelta(days=lookback_days)
    daily_oi = defaultdict(int)
    
    for filepath in contract_files:
        try:
            with open(filepath, "rb") as f:
                data = orjson.loads(f.read())
            
            history = data.get("history", [])
            for entry in history:
                entry_date_str = entry.get("date")
                if entry_date_str:
                    entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
                    if entry_date >= cutoff_date:
                        oi = entry.get("open_interest", 0) or 0
                        daily_oi[entry_date_str] += oi
        except Exception:
            continue
    
    if not daily_oi:
        return 0
    
    return round(statistics.mean(daily_oi.values()), 0)

def compute_option_chain_statistics(symbol):
    base_dir = os.path.join("json/all-options-contracts", symbol)
    contract_files = get_contracts_from_directory(base_dir)
    
    if len(contract_files) == 0:
        try:
            os.remove(f"json/options-chain-statistics/{symbol}.json")
            print(f'Deleted file for {symbol}')
        except:
            pass
        return {}

    by_exp = defaultdict(lambda: {
        "volume_calls": 0,
        "volume_puts": 0,
        "oi_calls": 0,
        "oi_puts": 0,
        "iv_all": [],  # Store all IV values for this expiration
    })
    
    # Track overall statistics
    total_volume = 0
    total_call_volume = 0
    total_put_volume = 0
    total_oi = 0
    total_call_oi = 0
    total_put_oi = 0
    
    for filepath in contract_files:
        try:
            with open(filepath, "rb") as f:
                data = orjson.loads(f.read())
            
            exp_str = data.get("expiration")
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            
            if exp_date < today:
                continue
            

            history = data.get("history", [])
            if not history:
                continue
            
            latest = history[-1]
            opt_type = data.get("optionType", "").lower()
            volume = latest.get("volume", 0) or 0
            oi = latest.get("open_interest", 0) or 0
            iv = latest.get("implied_volatility", 0) or 0
            
            if iv >= 0:
                by_exp[exp_str]["iv_all"].append(iv)

            # Track overall volume and OI
            total_volume += volume
            total_oi += oi
            

            
            if opt_type == "call":
                by_exp[exp_str]["volume_calls"] += volume
                by_exp[exp_str]["oi_calls"] += oi
                total_call_volume += volume
                total_call_oi += oi
            elif opt_type == "put":
                by_exp[exp_str]["volume_puts"] += volume
                by_exp[exp_str]["oi_puts"] += oi
                total_put_volume += volume
                total_put_oi += oi
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    # Calculate overall statistics



    overall_volume_pc_ratio = safe_div(total_put_volume, total_call_volume)
    overall_oi_pc_ratio = safe_div(total_put_oi, total_call_oi)
    volume_sentiment = get_sentiment_from_pc_ratio(overall_volume_pc_ratio)
    oi_sentiment = get_sentiment_from_pc_ratio(overall_oi_pc_ratio)
    
    # Calculate historical IV statistics
    historical_ivs, historical_with_dates = calculate_historical_iv_stats(symbol)
    
    
    # Calculate historical volatility
    historical_volatility = calculate_historical_volatility(symbol)
    
    # Calculate average daily volume and OI
    avg_daily_volume = calculate_average_daily_volume(symbol)
    avg_daily_oi = calculate_average_daily_oi(symbol)
    
    volume_percentage = safe_div(total_volume * 100, avg_daily_volume) if avg_daily_volume > 0 else 0
    oi_percentage = safe_div(total_oi * 100, avg_daily_oi) if avg_daily_oi > 0 else 0
    
    # Load max pain data
    max_pain_by_exp = {}
    try:
        with open(f"json/max-pain/{symbol}.json", "rb") as file:
            max_pain_data = orjson.loads(file.read())
            for entry in max_pain_data:
                exp_date = entry.get("expiration")
                max_pain = entry.get("maxPain", 0)
                max_pain_by_exp[exp_date] = max_pain
    except Exception as e:
        print(f"Error loading max pain data for {symbol}: {e}")
    
    # Build expiration-specific results
    expiration_data = []
    for exp, stats in by_exp.items():
        try:
            calls_vol = stats["volume_calls"]
            puts_vol = stats["volume_puts"]
            calls_oi = stats["oi_calls"]
            puts_oi = stats["oi_puts"]
            
            vol_ratio = safe_div(puts_vol, calls_vol)
            oi_ratio = safe_div(puts_oi, calls_oi)
            
            # Calculate average IV for all contracts with this expiration
            avg_iv = round(statistics.median(stats["iv_all"]) * 100, 2) if stats["iv_all"] else 0
            
            # Get max pain for this expiration
            max_pain = max_pain_by_exp.get(exp, 0)
            
            expiration_data.append({
                "expiration": exp,
                "callVol": calls_vol,
                "putVol": puts_vol,
                "pcVol": vol_ratio,
                "callOI": calls_oi,
                "putOI": puts_oi,
                "pcOI": oi_ratio,
                "avgIV": avg_iv,
                "maxPain": max_pain,
            })
        except:
            pass
    
    # Sort by expiration
    expiration_data.sort(key=lambda x: x["expiration"])
    
    #the idea to compute iv 30d is simple.
    # look at the table data and compute the median based of on the expiration dates that will expire in 30 days
    iv_within_30d = []

    for entry in expiration_data:
        try:
            exp_date = datetime.strptime(entry['expiration'], "%Y-%m-%d").date()
            delta = (exp_date - today).days
            if 0 <= delta <= 30:
                iv_within_30d.append(entry['avgIV'])
        except:
            pass

    iv_30d = round(statistics.median(iv_within_30d),2)

    iv_rank       = calculate_iv_rank(iv_30d/100, historical_ivs)
    iv_percentile = calculate_iv_percentile(iv_30d/100, historical_ivs)
    iv_high, iv_high_date, iv_low, iv_low_date = find_iv_extremes(historical_with_dates)

    # Return comprehensive statistics matching the screenshot
    return {
        "overview": {
            "date": today.strftime("%B %d, %Y"),
            "currentIV": iv_30d,
            "ivRank": iv_rank,
            "totalVolume": int(total_volume),
            "avgDailyVolume": int(avg_daily_volume),
            "volumePercentage": volume_percentage,
            "putCallRatio": overall_volume_pc_ratio,
            "sentiment": volume_sentiment,
            "totalCallVolume": int(total_call_volume),
            "totalPutVolume": int(total_put_volume),
            "totalOpenInterest": int(total_oi),
            "totalCallOI": int(total_call_oi),
            "totalPutOI": int(total_put_oi),
            "openInterestPCRatio": overall_oi_pc_ratio,
            "openInterestSentiment": oi_sentiment,
            "avgDailyOI": int(avg_daily_oi),
            "oiPercentage": oi_percentage
        },
        "impliedVolatility": {
            "current": iv_30d,
            "ivRank": iv_rank,
            "ivPercentile": iv_percentile,
            "historicalVolatility": historical_volatility,
            "ivHigh": iv_high,
            "ivHighDate": iv_high_date,
            "ivLow": iv_low,
            "ivLowDate": iv_low_date
        },
        "openInterest": {
            "total": int(total_oi),
            "calls": int(total_call_oi),
            "puts": int(total_put_oi),
            "putCallRatio": overall_oi_pc_ratio,
            "sentiment": oi_sentiment,
            "avgDaily": int(avg_daily_oi),
            "todayVsAvg": oi_percentage
        },
        "volume": {
            "total": int(total_volume),
            "calls": int(total_call_volume),
            "puts": int(total_put_volume),
            "putCallRatio": overall_volume_pc_ratio,
            "sentiment": volume_sentiment,
            "avgDaily": int(avg_daily_volume),
            "todayVsAvg": volume_percentage
        },
        "table": expiration_data
    }


def process_single_symbol(symbol):
    """Process a single symbol - wrapper for concurrent execution"""
    try:
        data = compute_option_chain_statistics(symbol)
        save_json(data, symbol)
        return f"✓ {symbol}"
    except Exception as e:
        return f"✗ {symbol}: {str(e)}"


def load_symbol_list():
    symbols = []
    db_configs = [
        ("stocks.db", "SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'") ,
        ("etf.db",    "SELECT DISTINCT symbol FROM etfs"),
        ("index.db",  "SELECT DISTINCT symbol FROM indices")
    ]

    for db_file, query in db_configs:
        try:
            con = sqlite3.connect(db_file)
            cur = con.cursor()
            cur.execute(query)
            symbols.extend([r[0] for r in cur.fetchall()])
            con.close()
        except Exception:
            continue

    return symbols

def process_symbols_concurrent(symbols, max_workers=None):
    """Process symbols concurrently with thread pool"""
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)  # Default to reasonable thread count
    
    results = []
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(process_single_symbol, symbol): symbol for symbol in symbols}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(symbols), desc="Processing symbols") as pbar:
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    pbar.update(1)
                    
                    # Optional: Update description with current symbol
                    if completed_count % 10 == 0:  # Update every 10 completions
                        pbar.set_description(f"Processing symbols ({completed_count}/{len(symbols)})")
                        
                except Exception as e:
                    results.append(f"✗ {symbol}: {str(e)}")
                    pbar.update(1)
    
    return results

if __name__ == "__main__":
    symbols = load_symbol_list()
    #symbols = ['AAPL']  # override for testing
    
    print(f"Processing {len(symbols)} symbols...")
    
    # Process symbols concurrently
    results = process_symbols_concurrent(symbols, max_workers=5)  # Adjust max_workers as needed
    
    # Print summary
    successful = sum(1 for r in results if r.startswith("✓"))
    failed = len(results) - successful
    
    print(f"\nCompleted: {successful} successful, {failed} failed")
    
    # Optionally print failed symbols
    if failed > 0:
        print("\nFailed symbols:")
        for result in results:
            if result.startswith("✗"):
                print(f"  {result}")