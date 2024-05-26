import pandas as pd
import numpy as np
from ta.utils import *
from ta.volatility import *
from ta.momentum import *
from ta.trend import *
from backtesting import Backtest, Strategy
from datetime import datetime
import sqlite3
import concurrent.futures
import json
from tqdm import tqdm

import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")


def parse_args():
    parser = argparse.ArgumentParser(description='Process stock or ETF data.')
    parser.add_argument('--db', choices=['stocks', 'etf'], required=True, help='Database name (stocks or etf)')
    parser.add_argument('--table', choices=['stocks', 'etfs'], required=True, help='Table name (stocks or etfs)')
    return parser.parse_args()

class MyStrategy(Strategy):
    price_delta = 0.05

    def init(self):
        # Define indicator conditions as functions
        self.buy_conditions = [
            lambda: self.data['sm_5'] > self.data['sm_20'],
            lambda: self.data['ema_10'] > self.data['ema_50'],
            lambda: self.data['macd'] > self.data['signal_line'],
            lambda: self.data['rsi'] <= 30,
            lambda: self.data['stoch_rsi'] <= 30,
            lambda: self.data['aroon_up'] > 50 and self.data['aroon_down'] < 50,
            lambda: self.data['bb_middle'] < self.data['Close'],
            lambda: self.data["adx_ind"] >= 25 and self.data["adx_pos_ind"] > self.data["adx_neg_ind"],
            lambda: self.data['roc'] >= 5,
            lambda: self.data['williams'] >= -20,
        ]
        self.sell_conditions = [
            lambda: self.data['sm_5'] <= self.data['sm_20'],
            lambda: self.data['ema_10'] <= self.data['ema_50'],
            lambda: self.data['macd'] <= self.data['signal_line'],
            lambda: self.data['rsi'] >= 70,
            lambda: self.data['stoch_rsi'] >= 70,
            lambda: self.data['aroon_up'] <= 50 and self.data['aroon_down'] >= 50,
            lambda: self.data['bb_middle'] > self.data['Close'],
            lambda: self.data["adx_ind"] < 25 and self.data["adx_pos_ind"] < self.data["adx_neg_ind"],
            lambda: self.data['roc'] <= -5,
            lambda: self.data['williams'] <= -80,
        ]



    def next(self):

        buy_signal_count = sum(condition() for condition in self.buy_conditions)
        sell_signal_count = sum(condition() for condition in self.sell_conditions)

        # Adjust the threshold according to your requirement (e.g., majority = 2 out of 3 conditions)
        buy_threshold = 8
        sell_threshold = 8

        # Set target take-profit and stop-loss prices to be one price_delta
        # away from the current closing price.

        upper, lower = self.data['Close'][-1] * (1 + np.r_[1, -1]*self.price_delta)

        if not self.position:
            # No existing position
            if buy_signal_count >= buy_threshold:
                self.buy(tp=upper, sl=lower)
        else:
            # There is an existing position
            if sell_signal_count >= sell_threshold:
                self.position.close()
           

class TradingSignals:

    def __init__(self,data):
        data['sm_5'] = sma_indicator(data["Close"], window=5)
        data['sm_20'] = sma_indicator(data["Close"], window=20)

        # Calculate MACD and its signal line
        data['macd'] = macd(data["Close"], window_slow=26, window_fast=12)
        data['signal_line'] = macd_signal(data["Close"], window_slow=26, window_fast=12)

        data['ema_10'] = ema_indicator(data['Close'], window=5)
        data['ema_50'] = sma_indicator(data['Close'], window=20)

        data['rsi'] = rsi(data['Close'], window=14)
        aroon = AroonIndicator(data['Close'], low=data['Low'], window=14)
        data['aroon_up'] = aroon.aroon_up()
        data['aroon_down'] = aroon.aroon_down()
        data['bb_middle'] = BollingerBands(close=data["Close"], window=20, window_dev=2).bollinger_mavg()

        data['roc'] = roc(data['Close'], window=14)

        data['williams'] = WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close']).williams_r()
        data['stoch_rsi'] = StochRSIIndicator(close=data['Close']).stochrsi()

        data['adx_ind'] = adx(data['High'],data['Low'],data['Close'])
        data['adx_pos_ind'] = adx_pos(data['High'], data['Low'], data['Close'])
        data['adx_neg_ind'] = adx_neg(data['High'], data['Low'], data['Close'])

        self.data = data


    def next_pred(self):

        df = self.data.copy()
    
        buy_conditions = [
            lambda: self.data['sm_5'].iloc[-1] > self.data['sm_20'].iloc[-1],
            lambda: self.data['ema_10'].iloc[-1] > self.data['ema_50'].iloc[-1],
            lambda: self.data['macd'].iloc[-1] > self.data['signal_line'].iloc[-1],
            lambda: self.data['rsi'].iloc[-1] <= 30,
            lambda: self.data['stoch_rsi'].iloc[-1] <= 30,
            lambda: self.data['aroon_up'].iloc[-1] > 50 and self.data['aroon_down'].iloc[-1] < 50,
            lambda: self.data['bb_middle'].iloc[-1] < self.data['Close'].iloc[-1],
            lambda: self.data['adx_ind'].iloc[-1] >= 25 and self.data['adx_pos_ind'].iloc[-1] > self.data['adx_neg_ind'].iloc[-1],
            lambda: self.data['roc'].iloc[-1] >= 5,
            lambda: self.data['williams'].iloc[-1] >= -20,
        ]

        sell_conditions = [
            lambda: self.data['sm_5'].iloc[-1] <= self.data['sm_20'].iloc[-1],
            lambda: self.data['ema_10'].iloc[-1] <= self.data['ema_50'].iloc[-1],
            lambda: self.data['macd'].iloc[-1] <= self.data['signal_line'].iloc[-1],
            lambda: self.data['rsi'].iloc[-1] >= 70,
            lambda: self.data['stoch_rsi'].iloc[-1] >= 70,
            lambda: self.data['aroon_up'].iloc[-1] <= 50 and self.data['aroon_down'].iloc[-1] >= 50,
            lambda: self.data['bb_middle'].iloc[-1] > self.data['Close'].iloc[-1],
            lambda: self.data['adx_ind'].iloc[-1] < 25 and self.data['adx_pos_ind'].iloc[-1] < self.data['adx_neg_ind'].iloc[-1],
            lambda: self.data['roc'].iloc[-1] <= -5,
            lambda: self.data['williams'].iloc[-1] <= -80,
        ]

        buy_signal_count = sum(condition() for condition in buy_conditions)
        sell_signal_count = sum(condition() for condition in sell_conditions)

        buy_threshold = 8
        sell_threshold =8

        signal = None
        if buy_signal_count >= buy_threshold and not sell_signal_count >= sell_threshold:
            signal = 'Buy'
        elif sell_signal_count >= sell_threshold and not buy_signal_count >= buy_threshold:
            signal = 'Sell'
        else:
            signal = 'Hold'

        return signal

        

    def run(self):

        df = self.data.copy()

       

        df = df.dropna()

        bt = Backtest(df, MyStrategy, cash=1000000, commission=0, exclusive_orders = True, trade_on_close=True)
        stats = bt.run()
        #print(stats)
        history_sheet = stats['_trades']
        #print(history_sheet)

        stats_output = stats[['Start','End','Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]',\
                    'Duration','Volatility (Ann.) [%]','Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',\
                    'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration','Avg. Drawdown Duration',\
                    '# Trades', 'Win Rate [%]','Best Trade [%]','Worst Trade [%]','Avg. Trade [%]',\
                    'Max. Trade Duration','Avg. Trade Duration','Profit Factor', 'Expectancy [%]','SQN']]

        stats_output = stats_output.to_dict()

        stats_output['Start'] = stats_output['Start'].strftime("%Y-%m-%d")
        stats_output['End'] = stats_output['End'].strftime("%Y-%m-%d")
        stats_output['Duration'] = str(stats_output['Duration']).replace(' days 00:00:00', '')
        stats_output['Avg. Trade Duration'] = str(stats_output['Avg. Trade Duration']).replace(' days 00:00:00', '')
        stats_output['Avg. Drawdown Duration'] = str(stats_output['Avg. Drawdown Duration']).replace(' days 00:00:00', '')

        stats_output['Max. Drawdown Duration'] = str(stats_output['Max. Drawdown Duration']).replace(' days 00:00:00', '')
        stats_output['Max. Trade Duration'] = str(stats_output['Max. Trade Duration']).replace(' days 00:00:00', '')

        stats_output['nextSignal'] = self.next_pred()
        #print(history_sheet)

        output_history_sheet = []

        for i in range(len(history_sheet)):
            output_history_sheet.append(
                        {'time': history_sheet['EntryTime'][i].strftime("%Y-%m-%d"),
                        'position': 'belowBar',
                        'color': '#59B0F6',
                        'shape': 'arrowUp',
                        #'text': 'Buy',
                        'size': 2.0,
                        }
                    )
            output_history_sheet.append(
                    {'time': history_sheet['ExitTime'][i].strftime("%Y-%m-%d"),
                    'position': 'aboveBar',
                    'color': '#E91E63',
                    'shape': 'arrowDown',
                    #'text': 'Sell',
                    'size': 2.0,
                    },
                )

        return [ stats_output , output_history_sheet]




def create_column(con):
    """
    Create the 'tradingSignals' column if it doesn't exist in the 'stocks' table.
    """
    query_check = f"PRAGMA table_info({table_name})"
    cursor = con.execute(query_check)
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'tradingSignals' not in columns:
        query = f"ALTER TABLE {table_name} ADD COLUMN tradingSignals TEXT"
        con.execute(query)
        con.commit()
    '''
    if 'ai_signal' not in columns:
        query = f"ALTER TABLE {table_name} ADD COLUMN ai_signal TEXT"
        con.execute(query)
        con.commit()
    '''


def update_database(res, symbol, con):
    query = f"UPDATE {table_name} SET tradingSignals = ? WHERE symbol = ?"
    res_json = json.dumps(res)  # Convert the pred dictionary to JSON string
    con.execute(query, (res_json, symbol))
    """
    query = f"UPDATE {table_name} SET ai_signal = ? WHERE symbol = ?"
    if res[0]['nextSignal'] == 'Sell':
        signal = 0
    elif res[0]['nextSignal'] == 'Hold':
        signal = 1
    elif res[0]['nextSignal'] == 'Buy':
        signal = 2
    else:
        signal = -1
    con.execute(query, (signal, symbol))
    """
    con.commit()




def process_symbol(ticker):
    try:
        query_template = """
            SELECT
                date, open, high, low, close
            FROM
                "{ticker}"
            WHERE
                date BETWEEN ? AND ?
        """

        query = query_template.format(ticker=ticker)
        df = pd.read_sql_query(query, con, params=(start_date, end_date))


        if not df.empty:
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            res = TradingSignals(df).run()
        else:
            res = []

        create_column(con)
        update_database(res, ticker, con)

    except:
        print(f"Failed create trading signals for {ticker}")


args = parse_args()
db_name = args.db
table_name = args.table
con = sqlite3.connect(f'backup_db/{db_name}.db')

symbol_query = f"SELECT DISTINCT symbol FROM {table_name}"
symbol_cursor = con.execute(symbol_query)
symbols = [symbol[0] for symbol in symbol_cursor.fetchall()]

start_date = datetime(1970, 1, 1)
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

start_date = datetime(2019, 1, 1)
end_date = datetime.today()
con = sqlite3.connect('stocks.db')
query = query_template.format(ticker=ticker)
df = pd.read_sql_query(query, con, params=(start_date, end_date))
con.close()

df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')


res = TradingSignals(df).run()
'''



'''
fig, ax = plt.subplots(figsize=(14,8))
ax.plot(df['Close'] , label = ticker ,linewidth=0.5, color='blue', alpha = 0.9)
#ax.plot(df['sm_5'], label = 'SMA10', alpha = 0.85)
#ax.plot(df['sm_20'], label = 'SMA50' , alpha = 0.85)
ax.scatter(history_sheet['EntryTime'] , history_sheet['EntryPrice'] , label = 'Buy' , marker = '^', color = 'green',alpha =1, s=100)
ax.scatter(history_sheet['ExitTime'] , history_sheet['ExitPrice']  , label = 'Sell' , marker = 'v', color = 'red',alpha =1, s=100)
ax.set_xlabel(f'{start_date} - {end_date}' ,fontsize=18)
ax.set_ylabel('Close Price INR (₨)' , fontsize=18)
legend = ax.legend()
ax.grid()
plt.tight_layout()
plt.show()

'''