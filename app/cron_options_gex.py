import numpy as np
from scipy.stats import norm
from datetime import datetime, date, timedelta
import pandas as pd
from benzinga import financial_data
import ujson
from collections import defaultdict
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


# Define the keys to keep
keys_to_keep = {'time', 'sentiment', 'option_activity_type', 'price', 'underlying_price', 'cost_basis', 'strike_price', 'date', 'date_expiration', 'open_interest', 'put_call', 'volume'}

def filter_data(item):
    # Filter the item to keep only the specified keys and format fields
    filtered_item = {key: value for key, value in item.items() if key in keys_to_keep}
    filtered_item['type'] = filtered_item['option_activity_type'].capitalize()
    filtered_item['sentiment'] = filtered_item['sentiment'].capitalize()
    filtered_item['underlying_price'] = round(float(filtered_item['underlying_price']), 2)
    filtered_item['put_call'] = 'Calls' if filtered_item['put_call'] == 'CALL' else 'Puts'
    return filtered_item


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

def calculate_otm_percentage(option_data_list):
    otm_count = 0
    total_options = len(option_data_list)
    
    for option_data in option_data_list:
        strike_price = float(option_data['strike_price'])
        put_call = option_data['put_call']
        stock_price = float(option_data['stock_price'])  # Get stock price for this option

        # Check if the option is out-of-the-money
        if (put_call == 'CALL' and strike_price > stock_price) or (put_call == 'PUT' and strike_price < stock_price):
            otm_count += 1
    
    if total_options > 0:
        return (otm_count / total_options) * 100
    else:
        return 0


def get_historical_option_data(option_data_list, df_price):
    summary_data = []

    for option_data in option_data_list:
        try:
            date = datetime.strptime(option_data['date'], "%Y-%m-%d").date()
            expiration_date = datetime.strptime(option_data['date_expiration'], "%Y-%m-%d").date()

            open_interest = int(option_data.get('open_interest', 0))
            volume = int(option_data.get('volume', 0))
            strike_price = float(option_data.get('strike_price', 0))
            put_call = option_data.get('put_call', 'CALL')
            sentiment = option_data.get('sentiment', 'NEUTRAL')

            # Safely convert premium to float, default to 0 if missing or invalid
            try:
                premium = float(option_data.get('cost_basis', 0))
            except (TypeError, ValueError):
                premium = 0

            # Determine the stock price based on expiration date
            if expiration_date > date.today():
                stock_price = df_price['close'].iloc[-1]  # Latest stock price
            else:
                # Get the stock price on the option's date
                stock_price_row = df_price[df_price['date'] == str(date)]
                if not stock_price_row.empty:
                    stock_price = stock_price_row['close'].values[0]
                else:
                    continue  # Skip this option if the price isn't available for the date

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

            # Append option data for later summarization
            summary_data.append({
                'date': date,
                'open_interest': open_interest,
                'c_vol': volume if put_call == 'CALL' else 0,
                'p_vol': volume if put_call == 'PUT' else 0,
                'bull_premium': bull_premium,
                'bear_premium': bear_premium,
                'neutral_premium': neutral_premium,
                'put_call': put_call,
                'strike_price': strike_price,
                'stock_price': stock_price
            })

        except Exception as e:
            print(f"Error processing option data: {e}")
            continue

    # Summarize by date
    df_summary = pd.DataFrame(summary_data)

    # Calculate OTM percentage for each day
    def calculate_daily_otm(df):
        return calculate_otm_percentage(df.to_dict('records'))  # Pass the day's options for OTM calculation

    # Apply OTM percentage calculation for each day
    daily_summary = df_summary.groupby('date').agg(
        total_oi=('open_interest', 'sum'),
        total_bull_prem=('bull_premium', 'sum'),
        total_bear_prem=('bear_premium', 'sum'),
        total_neutral_prem=('neutral_premium', 'sum'),
        c_vol=('c_vol', 'sum'),
        p_vol=('p_vol', 'sum'),
    ).reset_index()

    # Calculate OTM percentage for each date and assign it to the daily_summary
    daily_summary['otm_ratio'] = df_summary.groupby('date').apply(lambda df: round(calculate_otm_percentage(df.to_dict('records')), 1)).values


    # Calculate Bull/Bear/Neutral ratios
    try:
        total_prem = daily_summary['total_bull_prem'] + daily_summary['total_bear_prem'] + daily_summary['total_neutral_prem']
        daily_summary['bull_ratio'] = round(daily_summary['total_bull_prem'] / total_prem * 100, 2)
        daily_summary['bear_ratio'] = round(daily_summary['total_bear_prem'] / total_prem * 100, 2)
        daily_summary['neutral_ratio'] = round(daily_summary['total_neutral_prem'] / total_prem * 100, 2)
    except ZeroDivisionError:
        daily_summary['bull_ratio'] = None
        daily_summary['bear_ratio'] = None
        daily_summary['neutral_ratio'] = None

    # Calculate total volume (call + put) and format other fields
    daily_summary['total_volume'] = round(daily_summary['c_vol'] + daily_summary['p_vol'], 2)
    daily_summary['total_neutral_prem'] = round(daily_summary['total_neutral_prem'], 2)
    daily_summary['date'] = daily_summary['date'].astype(str)
    daily_summary = daily_summary.sort_values(by='date', ascending=False)

    # Return the summarized dataframe
    return daily_summary

def get_options_chain(option_data_list):
    # Convert raw data to DataFrame and ensure correct data types
    df = pd.DataFrame(option_data_list)
    type_conversions = {
        'cost_basis': float,
        'volume': int,
        'open_interest': int,
        'strike_price': float,
        'date_expiration': str  # Ensuring date_expiration is initially a string
    }
    for col, dtype in type_conversions.items():
        df[col] = df[col].astype(dtype)
    
    # Convert 'date_expiration' to datetime
    df['date_expiration'] = pd.to_datetime(df['date_expiration'])
    
    # Filter out rows where 'date_expiration' is in the past
    current_date = datetime.now()
    df = df[df['date_expiration'] > current_date]
    
    # Calculate total premium during grouping
    df['total_premium'] = df['cost_basis']
    
    # Group and aggregate data
    grouped = df.groupby(['date_expiration', 'strike_price', 'put_call']).agg(
        total_open_interest=('open_interest', 'sum'),
        total_volume=('volume', 'sum'),
        total_premium=('total_premium', 'sum')
    ).reset_index()
    
    # Pivot the data for puts and calls
    pivoted = grouped.pivot_table(
        index=['date_expiration', 'strike_price'],
        columns='put_call',
        values=['total_open_interest', 'total_volume', 'total_premium'],
        fill_value=0
    ).reset_index()
    
    # Flatten column names
    pivoted.columns = [' '.join(col).strip() for col in pivoted.columns.values]
    
    # Rename columns for clarity
    new_column_names = {
        'total_open_interest CALL': 'total_open_interest_call',
        'total_open_interest PUT': 'total_open_interest_put',
        'total_volume CALL': 'total_volume_call',
        'total_volume PUT': 'total_volume_put',
        'total_premium CALL': 'total_premium_call',
        'total_premium PUT': 'total_premium_put'
    }
    pivoted = pivoted.rename(columns=new_column_names)
    
    # Convert 'date_expiration' to string in ISO format
    pivoted['date_expiration'] = pivoted['date_expiration'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Ensure we capture all relevant columns
    columns_to_keep = ['strike_price'] + [col for col in pivoted.columns if col not in ['strike_price', 'date_expiration']]
    
    # Construct the options chain
    option_chain = pivoted.groupby('date_expiration').apply(
        lambda x: x[columns_to_keep].to_dict(orient='records')
    ).reset_index(name='chain')
    
    return option_chain

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
        
        # Group ticker_data by 'date' and collect all items for each date
        grouped_history = defaultdict(list)
        for item in ticker_data:
            filtered_item = filter_data(item)
            grouped_history[filtered_item['date']].append(filtered_item)

        daily_historical_option_data = get_historical_option_data(ticker_data, df_price)
        daily_historical_option_data = daily_historical_option_data.merge(df_price[['date', 'changesPercentage']], on='date', how='inner')

        # Add "history" column containing all filtered items with the same date
        daily_historical_option_data['history'] = daily_historical_option_data['date'].apply(lambda x: grouped_history.get(x, []))

        if not daily_historical_option_data.empty:
            save_json(ticker, daily_historical_option_data.to_dict('records'), 'json/options-historical-data/companies')


        option_chain_data = get_options_chain(ticker_data)
        if not option_chain_data.empty:
            save_json(ticker, option_chain_data.to_dict('records'), 'json/options-chain/companies')



        daily_gex = compute_daily_gex(ticker_data, volatility)
        daily_gex = daily_gex.merge(df_price[['date', 'close']], on='date', how='inner')
        if not daily_gex.empty:
            save_json(ticker, daily_gex.to_dict('records'), 'json/options-gex/companies')

    except Exception as e:
        print(e)
        pass

# Close the database connection
stock_con.close()
etf_con.close()