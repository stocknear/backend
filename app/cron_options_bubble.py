import sqlite3
from datetime import datetime, timedelta, date
import ujson
import asyncio
import os
from dotenv import load_dotenv
from benzinga import financial_data
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import math
from scipy.stats import norm
from scipy.optimize import brentq




load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

fin = financial_data.Benzinga(api_key)


risk_free_rate = 0.05

def black_scholes_price(S, K, T, r, sigma, option_type="CALL"):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "CALL":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "PUT":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Implied volatility function
def implied_volatility(S, K, T, r, market_price, option_type="CALL"):
    def objective_function(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
    
    # Use brentq to solve for the implied volatility
    try:
        return brentq(objective_function, 1e-6, 3)  # Bounds for volatility
    except ValueError:
        return None  # Return None if there's no solution


def calculate_dte(date_expiration):
    expiration_date = datetime.strptime(date_expiration, "%Y-%m-%d")
    return (expiration_date - datetime.today()).days

def calculate_avg_dte(data):
    active_options = [entry for entry in data if calculate_dte(entry['date_expiration']) >= 0]
    
    if active_options:
        total_dte = sum(entry['dte'] for entry in active_options)
        return int(total_dte / len(active_options))
    else:
        return 0

def calculate_put_call_volumes(data):
    put_volume = sum(int(entry['volume']) for entry in data if entry['put_call'] == 'PUT')
    call_volume = sum(int(entry['volume']) for entry in data if entry['put_call'] == 'CALL')
    return put_volume, call_volume

def options_bubble_data(chunk):
    try:
        company_tickers = ','.join(chunk)
        end_date = date.today()
        start_date = end_date - timedelta(365) #look 1 year ago

        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')

        res_list = []
        page = 0
        while True:
            try:
                data = fin.options_activity(company_tickers=company_tickers, page=page, pagesize=1000, date_from=start_date_str, date_to=end_date_str)
                data = ujson.loads(fin.output(data))['option_activity']
                res_list += data
                page +=1
            except:
                break

        res_filtered = [{key: value for key, value in item.items() if key in ['ticker','underlying_price','strike_price','price','date', 'date_expiration', 'put_call', 'volume', 'open_interest']} for item in res_list]

        #================Start computing historical iv60=====================#
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(res_filtered)

        # Ensure correct types for dates and numerical fields
        df['date'] = pd.to_datetime(df['date'])
        df['date_expiration'] = pd.to_datetime(df['date_expiration'])
        df['underlying_price'] = pd.to_numeric(df['underlying_price'], errors='coerce')
        df['strike_price'] = pd.to_numeric(df['strike_price'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
        
        df['days_to_expiration'] = (df['date_expiration'] - df['date']).dt.days
        df_30d = df[(df['days_to_expiration'] >= 40) & (df['days_to_expiration'] <= 80)]
        # Calculate implied volatility for options in the 30-day range
        iv_data = []
        for _, option in df_30d.iterrows():
            S = option['underlying_price']
            K = option['strike_price']
            T = option['days_to_expiration'] / 365
            market_price = option['price']
            option_type = "CALL" if option['put_call'] == "CALL" else "PUT"
            
            # Check for missing values
            if pd.notna(S) and pd.notna(K) and pd.notna(T) and pd.notna(market_price):
                # Calculate IV
                iv = implied_volatility(S, K, T, risk_free_rate, market_price, option_type)
                if iv is not None:
                    iv_data.append({
                        "date": option['date'],
                        "IV": iv,
                        "volume": option['volume']
                    })

        # Create a DataFrame with the calculated IV data
        iv_df = pd.DataFrame(iv_data)

        # Calculate daily IV60 by averaging IVs (weighted by volume)
        def calculate_daily_iv60(group):
            weighted_iv = (group["IV"] * group["volume"]).sum() / group["volume"].sum()
            return weighted_iv

        # Group by date and compute daily IV60
        iv60_history = iv_df.groupby("date").apply(calculate_daily_iv60)

        # Fill NaN values using forward fill to carry the last valid IV60 forward
        iv60_history = iv60_history.ffill()
        iv60_history = iv60_history.to_dict()
        iv60_dict = {k.strftime('%Y-%m-%d'): v for k, v in iv60_history.items()}
        #print(iv60_dict)
        #====================================================================#

        for option_type in ['CALL', 'PUT']:
            for item in res_filtered:
                try:
                    if item['put_call'].upper() == option_type:
                        item['dte'] = calculate_dte(item['date_expiration'])
                        if item['ticker'] in ['BRK.A', 'BRK.B']:
                            item['ticker'] = f"BRK-{item['ticker'][-1]}"
                except:
                    pass

        #Save raw data for each ticker for options page stack bar chart
        result_list = []
        for ticker in chunk:
            try:
                ticker_filtered_data = [entry for entry in res_filtered if entry['ticker'] == ticker]
                if len(ticker_filtered_data) != 0:
                    # Sum up calls and puts for each day for the plot
                    summed_data = {}
                    for entry in ticker_filtered_data:
                        volume = int(entry['volume'])
                        open_interest = int(entry['open_interest'])
                        put_call = entry['put_call']
                        date_str = entry['date']
                        
                        if date_str not in summed_data:
                            summed_data[date_str] = {'CALL': {'volume': 0, 'open_interest': 0}, 'PUT': {'volume': 0, 'open_interest': 0}, 'iv60': None}
                        
                        summed_data[date_str][put_call]['volume'] += volume
                        summed_data[date_str][put_call]['open_interest'] += open_interest
                        
                        if date_str in iv60_dict:
                            summed_data[date_str]['iv60'] = round(iv60_dict[date_str]*100,1)
                    
                    result_list.extend([{'date': date, 'CALL': summed_data[date]['CALL'], 'PUT': summed_data[date]['PUT'], 'iv60': summed_data[date]['iv60']} for date in summed_data])
                    
                    # Reverse the list
                    result_list = result_list[::-1]
                    
                    with open(f"json/options-flow/company/{ticker}.json", 'w') as file:
                        ujson.dump(result_list, file)
            except Exception as e:
                print(e)
                pass



        #Save bubble data for each ticker for overview page
        for ticker in chunk:

            bubble_data = {}
            for time_period, days in {'oneDay': 1, 'oneWeek': 7, 'oneMonth': 30, 'threeMonth': 90, 'sixMonth': 180, 'oneYear': 252}.items():
                start_date = end_date - timedelta(days=days) #end_date is today

                filtered_data = [item for item in res_filtered if start_date <= datetime.strptime(item['date'], '%Y-%m-%d').date() <= end_date]

                ticker_filtered_data = [entry for entry in filtered_data if entry['ticker'] == ticker]
                put_volume, call_volume = calculate_put_call_volumes(ticker_filtered_data)
                avg_dte = calculate_avg_dte(ticker_filtered_data)
                bubble_data[time_period] = {'putVolume': put_volume, 'callVolume': call_volume, 'avgDTE': avg_dte}

            if all(all(value == 0 for value in data.values()) for data in bubble_data.values()):
                bubble_data = {}
                #don't save the json
            else:
                with open(f"json/options-bubble/{ticker}.json", 'w') as file:
                    ujson.dump(bubble_data, file)

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(e)


async def main():
    try:
        stock_con = sqlite3.connect('stocks.db')
        stock_cursor = stock_con.cursor()
        stock_cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
        stock_symbols = [row[0] for row in stock_cursor.fetchall()]

        etf_con = sqlite3.connect('etf.db')
        etf_cursor = etf_con.cursor()
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]

        stock_con.close()
        etf_con.close()
        
        total_symbols = stock_symbols + etf_symbols
        total_symbols = [item.replace("BRK-B", "BRK.B") for item in total_symbols]

        print(len(total_symbols))

        chunk_size = len(total_symbols) // 2000  # Divide the list into N chunks
        chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
        #chunks = [['NVDA']]
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = [loop.run_in_executor(executor, options_bubble_data, chunk) for chunk in chunks]
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                await f

    except Exception as e:
        print(e)

if __name__ == "__main__":
    asyncio.run(main())
