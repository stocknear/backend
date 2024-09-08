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

def save_json(symbol, data, file_path):
    with open(f'{file_path}/{symbol}.json', 'w') as file:
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

    timestamp = datetime.strptime(option_data['date'], "%Y-%m-%d")

    try:
        S = float(option_data['underlying_price'])
        K = float(option_data['strike_price'])
        size = float(option_data['open_interest'])
        expiration_date = datetime.strptime(option_data['date_expiration'], "%Y-%m-%d")
        T = (expiration_date - timestamp).days / 365.0
        if T < 0:
            return 0, timestamp.date()
        elif T == 0:
            T = 1 #Consider 0DTE options

        option_type = option_data['put_call']
        delta_value = delta(S, K, T, r, sigma, option_type)
        gamma_value = gamma(S, K, T, r, sigma)
        notional = size * S
        gex = gamma_value * size * int(option_data['volume']) * S #gamma_value * notional * delta_value

        return gex, timestamp.date()
    except:
        return 0, timestamp.date()

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

def summarize_option_chain(option_data_list):
    summary_data = []

    for option_data in option_data_list:
        try:
            date = datetime.strptime(option_data['date'], "%Y-%m-%d").date()
            open_interest = int(option_data.get('open_interest', 0))
            volume = int(option_data.get('volume', 0))
            price = float(option_data.get('price', 0))
            strike_price = float(option_data.get('strike_price', 0))
            put_call = option_data.get('put_call', 'CALL')
            sentiment = option_data.get('sentiment', 'NEUTRAL')

            # Safely convert premium to float, default to 0 if missing or invalid
            try:
                premium = float(option_data.get('cost_basis', 0))
            except (TypeError, ValueError):
                premium = 0

            # Calculate Bull/Bear/Neutral premiums based on sentiment
            if sentiment == 'BULLISH':
                bull_premium = premium
                bear_premium = 0
                neutral_premium = 0
            elif sentiment == 'BEARISH':
                bull_premium = 0
                bear_premium = premium
                neutral_premium = 0
            else:
                bull_premium = 0
                bear_premium = 0
                neutral_premium = premium

            summary_data.append({
                'date': date,
                'open_interest': open_interest,
                'c_vol': volume if put_call == 'CALL' else 0,
                'p_vol': volume if put_call == 'PUT' else 0,
                'bull_premium': bull_premium,
                'bear_premium': bear_premium,
                'neutral_premium': neutral_premium
            })
        except:
            pass

    # Summarize by date
    df_summary = pd.DataFrame(summary_data)
    daily_summary = df_summary.groupby('date').agg(
        total_oi=('open_interest', 'sum'),
        total_bull_prem=('bull_premium', 'sum'),
        total_bear_prem=('bear_premium', 'sum'),
        total_neutral_prem=('neutral_premium', 'sum'),
        c_vol=('c_vol', 'sum'),
        p_vol=('p_vol', 'sum')
    ).reset_index()

    # Calculate Bull/Bear ratio
    
    try:
        daily_summary['bear_ratio'] = round(daily_summary['total_bear_prem'] / (daily_summary['total_bull_prem']+daily_summary['total_bear_prem']+daily_summary['total_neutral_prem']) * 100, 2)
        daily_summary['bull_ratio'] = round(daily_summary['total_bull_prem'] / (daily_summary['total_bull_prem']+daily_summary['total_bear_prem']+daily_summary['total_neutral_prem']) * 100, 2)
        daily_summary['neutral_ratio'] = round(daily_summary['total_neutral_prem'] / (daily_summary['total_bull_prem']+daily_summary['total_bear_prem']+daily_summary['total_neutral_prem']) * 100, 2)
    except:
        daily_summary['bear_ratio'] = None
        daily_summary['bull_ratio'] = None
        daily_summary['neutral_ratio'] = None
    

    daily_summary['total_volume'] = round(daily_summary['c_vol'] + daily_summary['p_vol'],2)
    daily_summary['total_neutral_prem'] = round(daily_summary['total_neutral_prem'],2)
    daily_summary['date'] = daily_summary['date'].astype(str)
    daily_summary = daily_summary.sort_values(by='date', ascending=False)
    # Return the summarized dataframe
    return daily_summary


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
etf_con = sqlite3.connect('etf.db')

stock_cursor = stock_con.cursor()
stock_cursor.execute("PRAGMA journal_mode = wal")
stock_cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND marketCap >= 500E6")
stock_symbols = [row[0] for row in stock_cursor.fetchall()]

etf_cursor = etf_con.cursor()
etf_cursor.execute("PRAGMA journal_mode = wal")
etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
etf_symbols = [row[0] for row in etf_cursor.fetchall()]

total_symbols = stock_symbols + etf_symbols

query_template = """
    SELECT date, close,change_percent
    FROM "{ticker}"
    WHERE date BETWEEN ? AND ?
"""

# Process each symbol
for ticker in total_symbols:
    try:
        query = query_template.format(ticker=ticker)
        df_price = pd.read_sql_query(query, stock_con if ticker in stock_symbols else etf_con, params=(start_date_str, end_date_str)).round(2)
        df_price = df_price.rename(columns={"change_percent": "changesPercentage"})

        volatility = calculate_volatility(df_price)

        ticker_data = get_data(ticker)
        daily_option_chain = summarize_option_chain(ticker_data)
        daily_option_chain = daily_option_chain.merge(df_price[['date', 'changesPercentage']], on='date', how='inner')
        if not daily_option_chain.empty:
            save_json(ticker, daily_option_chain.to_dict('records'), 'json/options-chain/companies')
        
        daily_gex = compute_daily_gex(ticker_data, volatility)
        daily_gex = daily_gex.merge(df_price[['date', 'close']], on='date', how='inner')
        if not daily_gex.empty:
            save_json(ticker, daily_gex.to_dict('records'),'json/options-gex/companies')

    except Exception as e:
        print(e)
        pass

# Close the database connection
stock_con.close()
etf_con.close()
