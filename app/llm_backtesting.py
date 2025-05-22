import json
import pandas as pd
from backtesting import Backtest, Strategy

# Step 1: Load and preprocess JSON data
with open('json/historical-price/max/TSLA.json', 'r') as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)
df['date'] = pd.to_datetime(df['time'])
df.set_index('date', inplace=True)

# Rename columns to match backtesting.py expectations
df.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

df.drop(columns=['time'], inplace=True)

print(df)
# Step 2: Define a basic moving average crossover strategy
class SmaCross(Strategy):
    def init(self):
        self.sma1 = self.I(lambda x: x.rolling(10).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: x.rolling(20).mean(), self.data.Close)

    def next(self):
        if self.sma1[-1] > self.sma2[-1] and self.sma1[-2] <= self.sma2[-2]:
            self.buy()
        elif self.sma1[-1] < self.sma2[-1] and self.sma1[-2] >= self.sma2[-2]:
            self.sell()

# Step 3: Run the backtest
bt = Backtest(df, SmaCross, cash=10000, commission=.002)
results = bt.run()
bt.plot()
