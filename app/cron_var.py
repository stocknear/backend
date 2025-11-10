import pandas as pd
from datetime import datetime
import numpy as np
import ujson
import orjson
import asyncio
import sqlite3
import os
from tqdm import tqdm

async def save_json(symbol, data):
    os.makedirs("json/var", exist_ok=True)  # Ensure directory exists
    with open(f"json/var/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))

def compute_var_monte_carlo(df, num_simulations=10000, trading_days_per_month=21):
    """
    Calculate monthly VaR using Monte Carlo simulation
    
    Args:
        df: DataFrame with price data
        num_simulations: Number of Monte Carlo simulations to run
        trading_days_per_month: Average trading days in a month (typically 21)
    
    Returns:
        Monthly VaR at 95% confidence level (as percentage)
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate daily returns
    df['Returns'] = df['adjClose'].pct_change()
    df = df.dropna()
    
    if len(df) < 2:
        return 0
    
    # Calculate historical statistics
    daily_returns = df['Returns'].values
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    
    # Monte Carlo simulation for monthly returns
    monthly_returns = []
    
    for _ in range(num_simulations):
        # Simulate daily returns for one month
        simulated_daily_returns = np.random.normal(
            mean_return, 
            std_return, 
            trading_days_per_month
        )
        
        # Calculate compounded monthly return
        # (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        monthly_return = np.prod(1 + simulated_daily_returns) - 1
        monthly_returns.append(monthly_return)
    
    # Calculate VaR at 95% confidence level (5th percentile)
    confidence_level = 0.95
    var_percentile = (1 - confidence_level) * 100
    monthly_var = np.percentile(monthly_returns, var_percentile)
    
    # Convert to percentage and return as positive value for loss
    var_percentage = round(abs(monthly_var) * 100, 2)
    
    # Cap at 99% maximum loss
    if var_percentage > 99:
        var_percentage = 99
    
    return var_percentage


async def run():
    start_date = "2015-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    total_symbols = stocks_symbols
    con.close()

    for symbol in tqdm(total_symbols):
        try:
            with open(f"json/historical-price/adj/{symbol}.json","rb") as file:
                df = pd.DataFrame(orjson.loads(file.read()))

            df['date'] = pd.to_datetime(df['date'])

        
            recent_days = 252
            if len(df) > recent_days:
                df_recent = df.tail(recent_days)
            else:
                df_recent = df
            
            # Compute current VaR using all recent data
            current_var = compute_var_monte_carlo(df_recent)
            
            # Create simplified history with just the current VaR
            history = [{'date': end_date, 'var': current_var}]

            res = {'history': history}
            if res:
                await save_json(symbol, res)

        except:
            pass


try:
    asyncio.run(run())
except Exception as e:
    print(e)
