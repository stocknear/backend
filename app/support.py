import pandas as pd
import numpy as np
import ujson
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, date, timedelta
from benzinga import financial_data
import os
from dotenv import load_dotenv
import seaborn as sns
import sqlite3

def calculate_volatility(prices_df):
    prices_df = prices_df.sort_values(by='date')
    prices_df['return'] = prices_df['close'].pct_change()
    returns = prices_df['return'].dropna()
    return returns.std() * np.sqrt(252)


# Load API key from environment
load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')
fin = financial_data.Benzinga(api_key)
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


query_template = """
    SELECT date, close,change_percent
    FROM "{ticker}"
    WHERE date BETWEEN ? AND ?
"""


ticker = 'SPY'
end_date = date.today()
start_date = end_date - timedelta(1)
end_date_str = end_date.strftime('%Y-%m-%d')
start_date_str = start_date.strftime('%Y-%m-%d')

query = query_template.format(ticker=ticker)
df_price = pd.read_sql_query(query, stock_con if ticker in stock_symbols else etf_con, params=('2024-01-01', end_date_str)).round(2)
df_price = df_price.rename(columns={"change_percent": "changesPercentage"})

volatility = calculate_volatility(df_price)
print(start_date, end_date)
print('volatility', volatility)

stock_con.close()
etf_con.close()

def get_data(ticker):
    res_list = []
    page = 0
    while True:
        try:
            data = fin.options_activity(date_from=start_date_str, date_to=end_date_str, company_tickers=ticker, page=page, pagesize=1000)
            data = ujson.loads(fin.output(data))['option_activity']
            if not data:
                break  # Stop when no more data is returned
            filtered_data = [{key: value for key, value in item.items() if key not in ['description_extended', 'updated']} for item in data]
            res_list += filtered_data
        except Exception as e:
            print(e)
            break
        page += 1
    return res_list


ticker_data = get_data(ticker)
ticker_data = [
    item for item in ticker_data 
    if datetime.strptime(item['date_expiration'], '%Y-%m-%d') >= datetime.now() and 
    datetime.strptime(item['date_expiration'], '%Y-%m-%d') <= datetime.now() + timedelta(days=5)
]

print(len(ticker_data))

def calculate_option_greeks(S, K, T, r, sigma, option_type='CALL'):
    """
    Calculate option Greeks using Black-Scholes formula
    S: Current stock price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free rate
    sigma: Volatility
    """
    if T <= 0:
        return 0, 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    #d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'CALL':
        delta = norm.cdf(d1)
        #gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    else:  # PUT
        delta = norm.cdf(d1) - 1 #-norm.cdf(-d1)
        #gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)) if S > 0 and sigma > 0 and np.sqrt(T) > 0 else 0
        
    return delta, gamma

def process_options_data(df):
    """
    Process options data and calculate DEX and GEX
    """
    # Convert data types
    df['strike_price'] = pd.to_numeric(df['strike_price'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['open_interest'] = pd.to_numeric(df['open_interest'])
    df['underlying_price'] = pd.to_numeric(df['underlying_price'])
    df['date_expiration'] = pd.to_datetime(df['date_expiration'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate time to expiration in years
    df['T'] = (df['date_expiration'] - df['date']).dt.days / 365.0
    
    # Parameters for calculations
    risk_free_rate = 0.05  # Current approximate risk-free rate
    # Calculate historical volatility (or you could use implied volatility if available)
    sigma = volatility #df['underlying_price'].pct_change().std() * np.sqrt(252) if len(df) > 30 else 0.3
    
    # Calculate Greeks for each option
    greeks = df.apply(lambda row: calculate_option_greeks(
        row['underlying_price'],
        row['strike_price'],
        row['T'],
        risk_free_rate,
        sigma,
        row['put_call']
    ), axis=1)
    
    df['delta'], df['gamma'] = zip(*greeks)
    
    # Calculate DEX (Delta Exposure) and GEX (Gamma Exposure)
    # Convert volume to float if it's not already
    df['volume'] = df['volume'].astype(int)
    df['open_interest'] = df['open_interest'].astype(float)
    
    # Get price per contract
    df['price'] = pd.to_numeric(df['price'])
    
    # Calculate position values
    contract_multiplier = 100  # Standard option contract multiplier
    df['position_value'] = df['price'] * df['open_interest'] * contract_multiplier
    
    # Calculate exposures
    df['gex'] = df['gamma'] * df['volume'] * contract_multiplier #df['gamma'] * df['open_interest'] * df['volume'] * df['underlying_price']
    df['dex'] = df['delta'] * df['volume'] * contract_multiplier * df['underlying_price'] * 0.01 #df['delta'] * df['volume'] * df['underlying_price'] *0.01
    
    return df

def plot_option_exposure(df, current_price):
    """
    Create visualization of DEX and GEX profiles with focused y-axis
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Aggregate exposures by strike price
    dex_by_strike = df.groupby('strike_price')['dex'].sum().reset_index()
    gex_by_strike = df.groupby('strike_price')['gex'].sum().reset_index()
    
    # Filter out strikes with no significant exposure
    significant_exposure = abs(dex_by_strike['dex']) > abs(dex_by_strike['dex']).max() * 0.01
    min_strike = dex_by_strike[significant_exposure]['strike_price'].min()
    max_strike = dex_by_strike[significant_exposure]['strike_price'].max()
    
    # Add some padding to the range
    strike_padding = (max_strike - min_strike) * 0.1
    y_min = max(min_strike - strike_padding, dex_by_strike['strike_price'].min())
    y_max = min(max_strike + strike_padding, dex_by_strike['strike_price'].max())
    
    # Plot DEX bars
    positive_dex = dex_by_strike[dex_by_strike['dex'] > 0]
    negative_dex = dex_by_strike[dex_by_strike['dex'] <= 0]
    
    ax.barh(positive_dex['strike_price'], positive_dex['dex'],
            color='green', alpha=0.7, label='Positive DEX')
    ax.barh(negative_dex['strike_price'], negative_dex['dex'],
            color='#964B00', alpha=0.7, label='Negative DEX')
    
    # Plot GEX profile
    ax.plot(gex_by_strike['gex'], gex_by_strike['strike_price'],
            color='yellow', label='GEX Profile', linewidth=2)
    
    # Calculate and plot support/resistance levels
    significant_gex = gex_by_strike[abs(gex_by_strike['gex']) > abs(gex_by_strike['gex']).mean()]
    resistance_level = significant_gex[significant_gex['strike_price'] > current_price]['strike_price'].min()
    support_level = significant_gex[significant_gex['strike_price'] < current_price]['strike_price'].max()
    
    # Add reference lines
    if pd.notna(resistance_level):
        ax.axhline(y=resistance_level, color='red', linestyle='--', alpha=0.5,
                   label=f'Call Resistance: {resistance_level:.1f}')
    if pd.notna(support_level):
        ax.axhline(y=support_level, color='green', linestyle='--', alpha=0.5,
                   label=f'Put Support: {support_level:.1f}')
    ax.axhline(y=current_price, color='white', linestyle='--', alpha=0.5,
               label=f'Spot Price: {current_price:.1f}')
    
    # Calculate and plot HVL
    hvl = dex_by_strike['strike_price'][abs(dex_by_strike['dex']).idxmin()]
    ax.axhline(y=hvl, color='gray', linestyle='--', alpha=0.5,
               label=f'HVL: {hvl:.1f}')
    
    # Set y-axis limits to focus on relevant region
    ax.set_ylim(y_min, y_max)
    
    # Customize the plot
    ax.set_title(f'Net DEX All Expirations for {df["ticker"].iloc[0]}\nTimestamp: {df["date"].iloc[0]}', pad=20)
    ax.set_xlabel('Exposure (Contract-Adjusted)')
    ax.set_ylabel('Strike Price')
    ax.grid(True, alpha=0.2)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}B' if abs(x) >= 1e6 else f'{x/1e3:.1f}M'))
    
    plt.tight_layout()
    return fig
    
# Use the functions with your data
df = pd.DataFrame(ticker_data)
processed_df = process_options_data(df)
current_price = float(df['underlying_price'].iloc[0])
fig = plot_option_exposure(processed_df, current_price)
plt.show()

# Print analysis
print("\nKey Levels Analysis:")
dex_by_strike = processed_df.groupby('strike_price')['dex'].sum()
gex_by_strike = processed_df.groupby('strike_price')['gex'].sum()

print(f"Largest DEX Positive: Strike ${dex_by_strike.idxmax():.2f} (${dex_by_strike.max()/1e6:.2f}M)")
print(f"Largest DEX Negative: Strike ${dex_by_strike.idxmin():.2f} (${dex_by_strike.min()/1e6:.2f}M)")
print(f"Largest GEX: Strike ${gex_by_strike.idxmax():.2f} (${gex_by_strike.max()/1e6:.2f}M)")
print(f"Net DEX: ${dex_by_strike.sum()/1e6:.2f}M")
print(f"Net GEX: ${gex_by_strike.sum()/1e6:.2f}M")