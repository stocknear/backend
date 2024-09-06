import numpy as np
from scipy.stats import norm
from datetime import datetime, date, timedelta
import pandas as pd
from benzinga import financial_data
import ujson
import sqlite3
import os
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')
fin = financial_data.Benzinga(api_key)

def save_json(symbol, data):
    with open(f'json/options-gex/companies/{symbol}.json', 'w') as file:
        ujson.dump(data, file)

def calculate_volatility(prices_df):
    prices_df = prices_df.sort_values(by='date')
    prices_df['return'] = prices_df['close'].pct_change()
    returns = prices_df['return'].dropna()
    return returns.std() * np.sqrt(252)

def black_scholes_d1(S, K, T, r, sigma):
    try:
        if sigma <= 0 or np.sqrt(T) <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    except ZeroDivisionError:
        return 0

def black_scholes_d2(S, K, T, r, sigma):
    return black_scholes_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def delta(S, K, T, r, sigma, option_type='CALL'):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    return norm.cdf(d1) if option_type == 'CALL' else norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    try:
        d1 = black_scholes_d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T)) if S > 0 and sigma > 0 and np.sqrt(T) > 0 else 0
    except ZeroDivisionError:
        return 0

def compute_gex(option_data, r=0.05, sigma=0.2):
    S = float(option_data['underlying_price'])
    K = float(option_data['strike_price'])
    size = float(option_data['open_interest'])
    expiration_date = datetime.strptime(option_data['date_expiration'], "%Y-%m-%d")
    timestamp = datetime.strptime(option_data['date'], "%Y-%m-%d")
    T = (expiration_date - timestamp).days / 365.0
    if T <= 0:
        return 0, timestamp.date()
    
    option_type = option_data['put_call']
    delta_value = delta(S, K, T, r, sigma, option_type)
    gamma_value = gamma(S, K, T, r, sigma)
    notional = size * S
    gex = gamma_value * notional * delta_value
    return gex, timestamp.date()

def compute_daily_gex(option_data_list, volatility):
    gex_data = []
    for option_data in option_data_list:
        gex, trade_date = compute_gex(option_data, sigma=volatility)
        if gex != 0:
            gex_data.append({'date': trade_date, 'gex': gex})
    
    gex_df = pd.DataFrame(gex_data)
    daily_gex = gex_df.groupby('date')['gex'].sum().reset_index()
    daily_gex['gex'] = round(daily_gex['gex'], 0)
    daily_gex['date'] = daily_gex['date'].astype(str)
    return daily_gex

def get_data(ticker):
    res_list = []
    page = 0
    while True:
        try:
            data = fin.options_activity(date_from=start_date_str, date_to=end_date_str, company_tickers=ticker, page=page, pagesize=1000)
            data = ujson.loads(fin.output(data))['option_activity']
            filtered_data = [{key: value for key, value in item.items() if key not in ['description_extended', 'updated']} for item in data]
            res_list += filtered_data
            page += 1
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")
            break
    return res_list

# Define date range
end_date = date.today()
start_date = end_date - timedelta(180)
end_date_str = end_date.strftime('%Y-%m-%d')
start_date_str = start_date.strftime('%Y-%m-%d')

# Connect to SQLite database
stock_con = sqlite3.connect('stocks.db')
stock_cursor = stock_con.cursor()
stock_cursor.execute("PRAGMA journal_mode = wal")
stock_cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND marketCap >= 1E9")
stock_symbols = [row[0] for row in stock_cursor.fetchall()]

query_template = """
    SELECT date, close
    FROM "{ticker}"
    WHERE date BETWEEN ? AND ?
"""

# Process each symbol
for ticker in stock_symbols:
    try:
        query = query_template.format(ticker=ticker)
        df_price = pd.read_sql_query(query, stock_con, params=(start_date_str, end_date_str)).round(2)
        volatility = calculate_volatility(df_price)
        
        ticker_data = get_data(ticker)
        daily_gex = compute_daily_gex(ticker_data, volatility)
        daily_gex = daily_gex.merge(df_price, on='date', how='inner')
        
        if not daily_gex.empty:
            save_json(ticker, daily_gex.to_dict('records'))
    except:
        pass

# Close the database connection
stock_con.close()
