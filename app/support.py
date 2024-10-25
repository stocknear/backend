import pandas as pd
import numpy as np
import ujson
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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


ticker = 'NVDA'
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
    Calculate option Greeks using Black-Scholes formula with improved accuracy
    S: Current stock price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free rate
    sigma: Volatility
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0, 0
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'CALL':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        else:  # PUT
            delta = -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
        return delta, gamma
    except:
        return 0, 0

def process_options_data_by_expiry(df):
    """
    Process options data with separate calculations for each expiration date
    """
    # Convert data types
    df['strike_price'] = pd.to_numeric(df['strike_price'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['open_interest'] = pd.to_numeric(df['open_interest'])
    df['underlying_price'] = pd.to_numeric(df['underlying_price'])
    df['date_expiration'] = pd.to_datetime(df['date_expiration'])
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = pd.to_numeric(df['price'])
    
    # Calculate time to expiration in years
    df['T'] = (df['date_expiration'] - df['date']).dt.days / 365.0
    
    # Use current risk-free rate
    risk_free_rate = 0.0525
    
    # Calculate Greeks for each option
    greeks = df.apply(lambda row: calculate_option_greeks(
        row['underlying_price'],
        row['strike_price'],
        row['T'],
        risk_free_rate,
        volatility,
        row['put_call']
    ), axis=1)
    
    df['delta'], df['gamma'] = zip(*greeks)
    
    # Calculate exposures
    contract_multiplier = 100
    df['delta_exposure'] = (df['delta'] * df['volume'] * contract_multiplier * 
                          df['underlying_price'])
    
    # Separate calls and puts
    df['option_type'] = np.where(df['put_call'] == 'CALL', 'call', 'put')
    
    return df

def plot_delta_exposure_by_expiry(df, current_price):
    """
    Create a visualization similar to the screenshot with delta exposure by expiration date
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Create custom colormap for calls and puts
    call_colors = ['#90EE90', '#32CD32', '#228B22']  # Light to dark green
    put_colors = ['#FFB6C1', '#DC143C', '#8B0000']   # Light to dark red
    
    # Get unique strike prices and expiration dates
    strike_prices = sorted(df['strike_price'].unique())
    expiry_dates = sorted(df['date_expiration'].unique())
    
    # Create y-axis ticks for strike prices
    y_ticks = np.arange(len(strike_prices))
    
    # Calculate total exposure for each strike and type
    total_exposure = pd.DataFrame()
    
    for expiry in expiry_dates:
        expiry_data = df[df['date_expiration'] == expiry]
        
        # Process calls
        calls = expiry_data[expiry_data['put_call'] == 'CALL']
        call_exposure = calls.groupby('strike_price')['delta_exposure'].sum()
        
        # Process puts
        puts = expiry_data[expiry_data['put_call'] == 'PUT']
        put_exposure = puts.groupby('strike_price')['delta_exposure'].sum()
        
        # Plot calls (positive x-axis)
        if not call_exposure.empty:
            ax.barh(y_ticks, call_exposure.reindex(strike_prices).fillna(0),
                   alpha=0.7, left=total_exposure.get('calls', 0),
                   color=call_colors[expiry_dates.tolist().index(expiry) % len(call_colors)],
                   height=0.8)
        
        # Plot puts (negative x-axis)
        if not put_exposure.empty:
            ax.barh(y_ticks, put_exposure.reindex(strike_prices).fillna(0),
                   alpha=0.7, left=total_exposure.get('puts', 0),
                   color=put_colors[expiry_dates.tolist().index(expiry) % len(put_colors)],
                   height=0.8)
        
        # Update total exposure
        total_exposure['calls'] = total_exposure.get('calls', 0) + call_exposure.reindex(strike_prices).fillna(0)
        total_exposure['puts'] = total_exposure.get('puts', 0) + put_exposure.reindex(strike_prices).fillna(0)
    
    # Add strike price labels
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'${price:.2f}' for price in strike_prices])
    
    # Add current price line
    current_price_idx = np.searchsorted(strike_prices, current_price)
    ax.axhline(y=current_price_idx, color='red', linestyle='--', alpha=0.5,
               label=f'Current Price: ${current_price:.2f}')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'${abs(x/1e6):.1f}M' if abs(x) >= 1e6 else f'${abs(x/1e3):.1f}K'))
    
    # Add labels and title
    ax.set_title(f'Delta Hedging Exposure by Strike and Expiration\n{df["ticker"].iloc[0]}',
                 pad=20)
    ax.set_xlabel('Delta Exposure ($)')
    ax.set_ylabel('Strike Price ($)')
    
    # Add grid
    ax.grid(True, alpha=0.2)
    
    # Add legend for expiration dates
    legend_elements = []
    for i, expiry in enumerate(expiry_dates):
        days_to_expiry = (expiry - df['date'].iloc[0]).days
        legend_elements.append(plt.Rectangle((0,0), 1, 1, 
                             fc=call_colors[i % len(call_colors)],
                             alpha=0.7,
                             label=f'Exp: {expiry.strftime("%Y-%m-%d")} ({days_to_expiry}d)'))
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    # Assuming we have the data loaded in ticker_data
    df = pd.DataFrame(ticker_data)
    
    if not df.empty:
        # Process the data
        processed_df = process_options_data_by_expiry(df)
        current_price = float(df['underlying_price'].iloc[0])
        
        # Create and show the plot
        fig = plot_delta_exposure_by_expiry(processed_df, current_price)
        plt.show()
        
        # Print summary statistics
        print("\nDelta Exposure Summary:")
        total_call_exposure = processed_df[processed_df['put_call'] == 'CALL']['delta_exposure'].sum()
        total_put_exposure = processed_df[processed_df['put_call'] == 'PUT']['delta_exposure'].sum()
        net_exposure = total_call_exposure + total_put_exposure
        
        print(f"Total Call Delta Exposure: ${total_call_exposure/1e6:.2f}M")
        print(f"Total Put Delta Exposure: ${total_put_exposure/1e6:.2f}M")
        print(f"Net Delta Exposure: ${net_exposure/1e6:.2f}M")
    else:
        print("No data available for analysis")