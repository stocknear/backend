import pandas as pd
from datetime import datetime
from ta.utils import *
from ta.volatility import *
from ta.momentum import *
from ta.trend import *
from ta.volume import *



class rating_model:
    def __init__(self, df):
        #Results are in the form of
        # Strong Sell => 0
        # Sell => 1
        # Neutral => 2
        # Buy => 3
        # Strong Buy => 4

        self.data = df
    
    def compute_overall_signal(self, data):
        ratingMap = {
            'Strong Sell': 0,
            'Sell': 1,
            'Neutral': 2,
            'Buy': 3,
            'Strong Buy': 4
        }

        # Extract overall ratings from the data
        overallRating = {item['name']: item['signal'] for item in data}

        # Compute mean overall rating
        mean_overall_rating = sum(ratingMap[val] for val in overallRating.values()) / len(overallRating)
        mean_overall_rating /= 4.0
        
        # Determine overall signal based on mean rating
        if 0 < mean_overall_rating <= 0.15:
            overall_signal = "Strong Sell"
        elif 0.15 < mean_overall_rating <= 0.45:
            overall_signal = "Sell"
        elif 0.45 < mean_overall_rating <= 0.55:
            overall_signal = 'Neutral'
        elif 0.55 < mean_overall_rating <= 0.8:
            overall_signal = 'Buy'
        elif 0.8 < mean_overall_rating <= 1.0:
            overall_signal = "Strong Buy"
        else:
            overall_signal = 'n/a'

        return overall_signal

    def ta_rating(self):
        df = pd.DataFrame()
        df['sma_20'] = sma_indicator(self.data['close'], window=20)
        df['sma_50'] = sma_indicator(self.data['close'], window=50)
        df['ema_20'] = ema_indicator(self.data['close'], window=20)
        df['ema_50'] = ema_indicator(self.data['close'], window=50)
        df['wma'] = wma_indicator(self.data['close'], window=20)
        df['adx'] = adx(self.data['high'],self.data['low'],self.data['close'])
        df["adx_pos"] = adx_pos(self.data['high'],self.data['low'],self.data['close'])
        df["adx_neg"] = adx_neg(self.data['high'],self.data['low'],self.data['close'])
        df['williams'] = WilliamsRIndicator(high=self.data['high'], low=self.data['low'], close=self.data['close']).williams_r()

        # Assign ratings based on SMA values
        df['sma_rating'] = 'Neutral'
        if self.data['close'].iloc[-1] < df['sma_50'].iloc[-1] and df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1]:
            df['sma_rating'] = 'Strong Sell'
        elif self.data['close'].iloc[-1] < df['sma_20'].iloc[-1] and df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1]:
            df['sma_rating'] = 'Sell'
        elif df['sma_20'].iloc[-1] <= self.data['close'].iloc[-1] <= df['sma_50'].iloc[-1]:
            df['sma_rating'] = 'Neutral'
        elif self.data['close'].iloc[-1] > df['sma_20'].iloc[-1] and df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]:
            df['sma_rating'] = 'Buy'
        elif self.data['close'].iloc[-1] > df['sma_50'].iloc[-1] and df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]:
            df['sma_rating'] = 'Strong Buy'

        # Assign ratings for ema
        df['ema_rating'] = 'Neutral'

        if self.data['close'].iloc[-1] < df['ema_50'].iloc[-1] and df['ema_20'].iloc[-1] < df['ema_50'].iloc[-1]:
            df['ema_rating'] = 'Strong Sell'
        elif self.data['close'].iloc[-1] < df['ema_20'].iloc[-1] and df['ema_20'].iloc[-1] < df['ema_50'].iloc[-1]:
            df['ema_rating'] = 'Sell'
        elif df['ema_20'].iloc[-1] <= self.data['close'].iloc[-1] <= df['ema_50'].iloc[-1]:
            df['ema_rating'] = 'Neutral'
        elif self.data['close'].iloc[-1] > df['ema_20'].iloc[-1] and df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1]:
            df['ema_rating'] = 'Buy'
        elif self.data['close'].iloc[-1] > df['ema_50'].iloc[-1] and df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1]:
            df['ema_rating'] = 'Strong Buy'

        # Assign ratings based on wma
        df['wma_rating'] = pd.cut(self.data['close'] - df['wma'],
                                  bins=[float('-inf'), -10, -5, 0, 5, 10],
                                  labels=['Strong Sell', 'Sell', 'Neutral', 'Buy', 'Strong Buy'])

        # Assign ratings based on adx
        if df['adx'].iloc[-1] > 50 and df['adx_neg'].iloc[-1] > df['adx_pos'].iloc[-1]:
                df['adx_rating'] = 'Strong Sell'
        elif df['adx'].iloc[-1] >=25 and df['adx'].iloc[-1] <=50 and df['adx_neg'].iloc[1] > df['adx_pos'].iloc[-1]:
                df['adx_rating'] = 'Sell'
        elif df['adx'].iloc[-1] < 25:
                df['adx_rating'] = 'Neutral'
        elif df['adx'].iloc[-1] >=25 and df['adx'].iloc[-1] <=50 and df['adx_pos'].iloc[-1] > df['adx_neg'].iloc[-1]:
                df['adx_rating'] = 'Buy'
        elif df['adx'].iloc[-1] > 50 and df['adx_pos'].iloc[-1] > df['adx_neg'].iloc[-1]:
                df['adx_rating'] = 'Strong Buy'
        else:
            df['adx_rating'] = 'Neutral'

      
        # Assign ratings based on williams
        df['williams_rating'] = 'Neutral'
        df.loc[df["williams"] < -80, 'williams_rating'] = "Strong Sell"
        df.loc[(df["williams"] >= -80) & (df["williams"] < -50), 'williams_rating'] = "Sell"
        df.loc[(df["williams"] >= -50) & (df["williams"] <= -20), 'williams_rating'] = "Buy"
        df.loc[df["williams"] > -20, 'williams_rating'] = "Strong Buy"
                
        #=========Momentum Indicators ============#

      
        aroon = AroonIndicator(self.data['close'], low=self.data['low'], window=14)
        df['rsi'] = rsi(self.data['close'], window=14)
        df['stoch_rsi'] = stochrsi_k(self.data['close'], window=14, smooth1 = 3, smooth2 =3)*100

        df['macd'] = macd(self.data['close'])
        df['macd_signal'] = macd_signal(self.data['close'])
        df['macd_hist'] = 2*macd_diff(self.data['close'])
        df['roc'] = roc(self.data['close'], window=14)
        df['cci'] = CCIIndicator(high=self.data['high'], low=self.data['low'], close=self.data['close']).cci()
        df['mfi'] = MFIIndicator(high=self.data['high'], low=self.data['low'], close=self.data['close'], volume=self.data['volume']).money_flow_index()
        
        # Assign ratings based on MFI values
        df['mfi_rating'] = pd.cut(df['mfi'], 
                                  bins=[-1, 20, 40, 60, 80, 101], 
                                  labels=['Strong Buy', 'Buy', 'Neutral', 'Sell', 'Strong Sell'])

        # Assign ratings based on RSI values
        df['rsi_rating'] = pd.cut(df['rsi'], 
                              bins=[-1, 30, 50, 60, 70, 101], 
                              labels=['Strong Buy', 'Buy', 'Neutral', 'Sell', 'Strong Sell'])

        # Assign ratings based on Stoch RSI values
        df['stoch_rsi_rating'] = pd.cut(df['stoch_rsi'], 
                                bins=[-1, 30, 50, 60, 70, 101], 
                                labels=['Strong Buy', 'Buy', 'Neutral', 'Sell', 'Strong Sell'])


        # Assign ratings for  macd
        if df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and df['macd_hist'].iloc[-1] < 0 \
           and df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2]:
           df['macd_rating'] = 'Strong Sell'
        elif df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and df['macd_hist'].iloc[-1] < 0 \
           and df['macd_hist'].iloc[-1] < df['macd_hist'].iloc[-2]:
           df['macd_rating'] = 'Sell'
        elif abs(df['macd'].iloc[-1] - df['macd_signal'].iloc[-1]) < 0.01 and abs(df['macd_hist'].iloc[-1]) < 0.01:
            df['macd_rating'] = 'Neutral'
        elif df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and df['macd_hist'].iloc[-1] > 0 and df['macd_hist'].iloc[-1] < df['macd_hist'].iloc[-2]:
            df['macd_rating'] = 'Buy'        
        elif df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and df['macd_hist'].iloc[-1] > 0 and df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2]:
            df['macd_rating'] = 'Strong Buy'
        else:
            df['macd_rating'] = 'Neutral'
    

        # Assign ratings for roc
        if df['roc'].iloc[-1] < -10:
            df['roc_rating'] = 'Strong Sell'
        elif df['roc'].iloc[-1] > -10 and df['roc'].iloc[-1] <= -5:
            df['roc_rating'] = 'Sell'
        elif df['roc'].iloc[-1] > -5 and df['roc'].iloc[-1] < 5:
            df['roc_rating'] = 'Neutral'
        elif df['roc'].iloc[-1] >=5 and df['roc'].iloc[-1] < 10:
            df['roc_rating'] = 'Buy'
        elif df['roc'].iloc[-1] >= 10:
            df['roc_rating'] = 'Strong Buy'
        else:
            df['roc_rating'] = 'Neutral'
        


        # Define CCI threshold values for signals
        cci_strong_sell_threshold = -100
        cci_sell_threshold = -50
        cci_buy_threshold = 50
        cci_strong_buy_threshold = 100

        # Assign signals based on CCI values
        if df['cci'].iloc[-1] < cci_strong_sell_threshold:
            df['cci_rating'] = 'Strong Sell'
        elif cci_strong_sell_threshold <= df['cci'].iloc[-1] < cci_sell_threshold:
            df['cci_rating'] = 'Sell'
        elif cci_sell_threshold <= df['cci'].iloc[-1] < cci_buy_threshold:
            df['cci_rating'] = 'Neutral'
        elif cci_buy_threshold <= df['cci'].iloc[-1] < cci_strong_buy_threshold:
            df['cci_rating'] = 'Buy'
        else:
            df['cci_rating'] = 'Strong Buy'



        res_list = [
        {'name': 'Relative Strength Index (14)', 'value': round(df['rsi'].iloc[-1],2), 'signal': df['rsi_rating'].iloc[-1]},
        {'name': 'Stochastic RSI Fast (3,3,14,14)', 'value': round(df['stoch_rsi'].iloc[-1],2), 'signal': df['stoch_rsi_rating'].iloc[-1]},
        {'name': 'Money Flow Index (14)', 'value': round(df['mfi'].iloc[-1],2), 'signal': df['mfi_rating'].iloc[-1]},
        {'name': 'Simple Moving Average (20)', 'value': round(df['sma_20'].iloc[-1],2), 'signal': df['sma_rating'].iloc[-1]},
        {'name': 'Exponential Moving Average (20)', 'value': round(df['ema_20'].iloc[-1],2), 'signal': df['ema_rating'].iloc[-1]},
        {'name': 'Weighted Moving Average (20)', 'value': round(df['wma'].iloc[-1],2), 'signal': df['wma_rating'].iloc[-1]},
        {'name': 'Average Directional Index (14)', 'value': round(df['adx'].iloc[-1],2), 'signal': df['adx_rating'].iloc[-1]},
        {'name': 'Commodity Channel Index (14)', 'value': round(df['cci'].iloc[-1],2), 'signal': df['cci_rating'].iloc[-1]},
        {'name': 'Rate of Change (12)', 'value': round(df['roc'].iloc[-1],2), 'signal': df['roc_rating'].iloc[-1]},
        {'name': 'Moving Average Convergence Divergence (12, 26)', 'value': round(df['macd'].iloc[-1],2), 'signal': df['macd_rating'].iloc[-1]},
        {'name': 'Williams %R (14)', 'value': round(df['williams'].iloc[-1],2), 'signal': df['williams_rating'].iloc[-1]}
        ]

        overall_signal = self.compute_overall_signal(res_list)

        res_dict = {'overallSignal': overall_signal, 'signalList': res_list}
        return res_dict

    # Load the historical stock price data

   



#Testing mode
# Load the data
'''
import sqlite3
start_date = "2015-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
con = sqlite3.connect('stocks.db')
symbol = 'ZTS'

query_template = """
    SELECT
        date, open, high, low, close, volume
    FROM
        "{symbol}"
    WHERE
        date BETWEEN ? AND ?
"""
query = query_template.format(symbol=symbol)
df = pd.read_sql_query(query, con, params=(start_date, end_date))

test = rating_model(df).ta_rating()
print(test)
con.close()
'''