import pandas as pd
import numpy as np
from ta.utils import *
from ta.volatility import *
from ta.momentum import *
from ta.trend import *
from ta.volume import *
from datetime import datetime
import sqlite3
import concurrent.futures
import json
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

#This is for the stock screener

class TASignals:

    def __init__(self,data):

        self.data = data

    def run(self):
        ta_df = pd.DataFrame()

        ta_df['sma_20'] = sma_indicator(self.data["Close"], window=20)
        ta_df['sma_50'] = sma_indicator(self.data["Close"], window=50)
        ta_df['sma_100'] = sma_indicator(self.data["Close"], window=100)
        ta_df['sma_200'] = sma_indicator(self.data["Close"], window=200)
        ta_df['ema_20'] = ema_indicator(self.data['Close'], window=20)
        ta_df['ema_50'] = ema_indicator(self.data['Close'], window=50)
        ta_df['ema_100'] = sma_indicator(self.data["Close"], window=100)
        ta_df['ema_200'] = sma_indicator(self.data['Close'], window=200)
        ta_df['rsi'] = rsi(self.data['Close'], window=14)
        ta_df['stoch_rsi'] = stochrsi_k(self.data['Close'], window=14, smooth1 = 3, smooth2 =3)*100
        ta_df['atr'] = AverageTrueRange(self.data['High'], self.data['Low'], self.data['Close'], window=14).average_true_range()
        ta_df['cci'] = CCIIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close']).cci()
        ta_df['mfi'] = MFIIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], volume=self.data['Volume']).money_flow_index()


        last_values = {col: [round(ta_df[col].iloc[-1],2)] for col in ta_df.columns} if not ta_df.empty else None
        last_values_df = pd.DataFrame(last_values)

        return last_values_df



def create_columns(con, ta_df):
    """
    Create columns in the table for each indicator if they don't exist.
    """
    cursor = con.cursor()
    existing_columns = cursor.execute(f"PRAGMA table_info(stocks)").fetchall()
    existing_column_names = [col[1] for col in existing_columns]

    for column in ta_df.columns:
        if column not in existing_column_names:
            cursor.execute(f"ALTER TABLE stocks ADD COLUMN {column} REAL")
    con.commit()

def update_database(res, symbol, con):
    """
    Update the database for the given symbol with the indicators' last values.
    """
    if not res.empty:
        # Create a single row update query with all columns
        columns = ', '.join(res.columns)
        placeholders = ', '.join(['?'] * len(res.columns))
        values = res.iloc[0].tolist()
        
        query = f"UPDATE stocks SET ({columns}) = ({placeholders}) WHERE symbol = '{symbol}'"
        con.execute(query, values)
        con.commit()



def process_symbol(ticker):
    try:
        query_template = """
            SELECT
                date, open, high, low, close,volume
            FROM
                "{ticker}"
            WHERE
                date BETWEEN ? AND ?
        """

        query = query_template.format(ticker=ticker)
        df = pd.read_sql_query(query, con, params=(start_date, end_date))


        if not df.empty:
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})

            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            ta_df = TASignals(df).run()
        else:
            ta_df = []

        create_columns(con, ta_df)
        update_database(ta_df, ticker, con)

    except:
        pass
        #print(f"Failed create ta signals for {ticker}: {e}")



con = sqlite3.connect(f'backup_db/stocks.db')


symbol_query = f"SELECT DISTINCT symbol FROM stocks"
symbol_cursor = con.execute(symbol_query)
symbols = [symbol[0] for symbol in symbol_cursor.fetchall()]

#Test mode
#symbols = ['TSLA']

start_date = datetime(2022, 1, 1)
end_date = datetime.today()

# Number of concurrent workers
num_processes = 4 # You can adjust this based on your system's capabilities
futures = []

with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    for symbol in symbols:
        futures.append(executor.submit(process_symbol, symbol))

    # Use tqdm to wrap around the futures for progress tracking
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(symbols), desc="Processing"):
        pass
con.close()


#==============Test mode================
'''
ticker = 'AAPL'
query_template = """
    SELECT
        date, open, high, low, close
    FROM
        {ticker}
    WHERE
        date BETWEEN ? AND ?
"""

start_date = datetime(1970, 1, 1)
end_date = datetime.today()
con = sqlite3.connect('stocks.db')
query = query_template.format(ticker=ticker)
df = pd.read_sql_query(query, con, params=(start_date, end_date))
con.close()

df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')


res = TASignals(df).run()
print(res)
'''