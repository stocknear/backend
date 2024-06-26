import pandas as pd
from datetime import datetime
#import yfinance as yf
import numpy as np
import ujson
import asyncio
import sqlite3
from tqdm import tqdm



async def save_json(symbol, data):
    with open(f"json/var/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

# Define risk rating scale
def assign_risk_rating(var):
    if var >= 25:  # This threshold can be adjusted based on your specific criteria
        return 1
    elif var >= 20:
        return 2
    elif var >= 15:
        return 3
    elif var >= 10:
        return 4
    elif var >= 8:
        return 5
    elif var >= 6:
        return 6
    elif var >= 4:
        return 7
    elif var >= 2:
        return 8
    elif var >= 1:
        return 9
    else:
        return 10

def compute_var(df):
    # Calculate daily returns
    df['Returns'] = df['close'].pct_change()
    df = df.dropna()
    # Calculate VaR at 95% confidence level
    confidence_level = 0.95
    var = abs(np.percentile(df['Returns'], 100 * (1 - confidence_level)))
    var_N_days = round(var * np.sqrt(5)*100,2) # N days

    # Assign risk rating
    risk_rating = assign_risk_rating(var_N_days)
    outlook = 'Neutral'
    if risk_rating < 5:
        outlook = 'Risky'
    elif risk_rating > 5:
        outlook = 'Minimum Risk'

    return {'rating': risk_rating, 'var': -var_N_days, 'outlook': outlook}

    #print(f"The Value at a 95% confidence level is: {var_N_days}%")
    #print(f"The risk rating based on the Value at Risk is: {risk_rating}")

async def run():
    start_date = "2015-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')
    crypto_con = sqlite3.connect('crypto.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    crypto_cursor = crypto_con.cursor()
    crypto_cursor.execute("PRAGMA journal_mode = wal")
    crypto_cursor.execute("SELECT DISTINCT symbol FROM cryptos")
    crypto_symbols = [row[0] for row in crypto_cursor.fetchall()]

    total_symbols = stocks_symbols + etf_symbols + crypto_symbols

    for symbol in tqdm(total_symbols):
        if symbol in etf_symbols:  # Fixed variable name from symbols to symbol
            query_con = etf_con
        elif symbol in crypto_symbols:  # Fixed variable name from symbols to symbol
            query_con = crypto_con
        elif symbol in stocks_symbols:  # Fixed variable name from symbols to symbol
            query_con = con

        query_template = """
                SELECT
                    date, open, high, low, close, volume
                FROM
                    "{symbol}"
                WHERE
                    date BETWEEN ? AND ?
            """
        query = query_template.format(symbol=symbol)
        df = pd.read_sql_query(query, query_con, params=(start_date, end_date))

        try:
            res_dict = compute_var(df)
            
            await save_json(symbol, res_dict)
        except Exception as e:
            print(e)
            

    con.close()
    etf_con.close()
    crypto_con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)

#Test mode
'''

# Download data
ticker = 'TCON'
start_date = datetime(2015, 1, 1)
end_date = datetime.today()

df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
df = df.reset_index()
df = df[['Date', 'Close']]

# Calculate daily returns
df['Returns'] = df['Close'].pct_change()
df = df.dropna()
'''