import json
import pandas as pd
from backtesting import Backtest, Strategy # Ensure backtesting is installed: pip install backtesting

# Step 1: Load and preprocess JSON data
# Assuming your JSON structure is a list of dictionaries or a dictionary of lists
# that pandas can directly interpret.
# Common issue: The structure of 'raw_data' might not be what pd.DataFrame expects.
# If 'raw_data' is a dictionary with a 'historical' key (common in APIs),
# you might need: df = pd.DataFrame(raw_data['historical'])
try:
    with open('json/historical-price/max/TSLA.json', 'r') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print("Error: The JSON file was not found. Please check the path.")
    exit()
except json.JSONDecodeError:
    print("Error: The JSON file is not formatted correctly.")
    exit()

# Assuming the JSON directly converts to a list of records (dictionaries)
# If 'raw_data' is a dictionary that itself contains the list (e.g., under a key like 'prices' or 'historical')
# you might need something like:
# if isinstance(raw_data, dict) and 'historical' in raw_data:
# df = pd.DataFrame(raw_data['historical'])
# elif isinstance(raw_data, list):
# df = pd.DataFrame(raw_data)
# else:
# print("Error: Unexpected JSON structure.")
# exit()

df = pd.DataFrame(raw_data)

# Check if the DataFrame is empty after loading
if df.empty:
    print("Error: DataFrame is empty after loading JSON. Check the JSON structure and content.")
    exit()

# Verify that 'time' column exists before trying to convert it
if 'time' not in df.columns:
    print("Error: 'time' column not found in the JSON data. Available columns are:", df.columns)
    # Attempt to find a likely date column if 'time' is missing
    # This is a common variation; financial data often uses 'date' directly
    if 'date' in df.columns:
        print("Found 'date' column, will use it instead of 'time'.")
        df.rename(columns={'date': 'original_date_col'}, inplace=True) # temp rename if 'date' is also a target
        df['date'] = pd.to_datetime(df['original_date_col'])
    else:
        # Add more potential date column names if needed
        potential_date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if potential_date_cols:
            print(f"Found potential date column(s): {potential_date_cols}. Please rename one to 'time' or adjust the script.")
        exit()
else:
    df['date'] = pd.to_datetime(df['time'])

df.set_index('date', inplace=True)

# Rename columns to match backtesting.py expectations
# Ensure all expected columns ('open', 'high', 'low', 'close', 'volume') exist
expected_cols_original_names = ['open', 'high', 'low', 'close', 'volume']
missing_cols = [col for col in expected_cols_original_names if col not in df.columns]
if missing_cols:
    print(f"Error: Missing expected columns in DataFrame: {missing_cols}. Available columns are: {df.columns}")
    exit()

df.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Drop the original 'time' column if it exists and is different from the new 'date' index
if 'time' in df.columns:
    df.drop(columns=['time'], inplace=True)
if 'original_date_col' in df.columns: # If we used 'date' as source and renamed it
    df.drop(columns=['original_date_col'], inplace=True)

# Ensure data is sorted by date, which is crucial for time series analysis
df.sort_index(inplace=True)

# Optional: Check for NaNs after processing, especially in 'Close'
if df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any():
    print("Warning: NaN values found in OHLCV data. Consider handling them (e.g., df.dropna(), df.fillna()).")
    # df.dropna(inplace=True) # Example: drop rows with NaNs

print("DataFrame after preprocessing:")
print(df.head())
print(f"\nDataFrame shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Step 2: Define a basic moving average crossover strategy
class SmaCross(Strategy):
    # Optional: Add parameters for SMA periods for flexibility
    n1 = 10 # Short moving average period
    n2 = 20 # Long moving average period

    def init(self):
        # Ensure self.data.Close is a pandas Series
        close_prices = pd.Series(self.data.Close)
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), close_prices)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), close_prices)

    def next(self):
        # Ensure there's enough data for both SMAs to be calculated
        # sma1 and sma2 might be NaN if not enough data points yet
        # Accessing -1 and -2 might cause index out of bounds if data is too short
        # or if SMAs haven't been computed yet (NaNs at the beginning)
        if len(self.sma1) < 2 or len(self.sma2) < 2: # Need at least two points for [-2]
            return

        # Check for NaN values before comparison, common at the start of the series
        if pd.isna(self.sma1[-1]) or pd.isna(self.sma2[-1]) or \
           pd.isna(self.sma1[-2]) or pd.isna(self.sma2[-2]):
            return

        # Original crossover logic
        # Buy signal: short SMA crosses above long SMA
        if self.sma1[-1] > self.sma2[-1] and self.sma1[-2] <= self.sma2[-2]:
            self.buy()
        # Sell signal: short SMA crosses below long SMA
        elif self.sma1[-1] < self.sma2[-1] and self.sma1[-2] >= self.sma2[-2]:
            self.sell()

# Step 3: Run the backtest
# Ensure DataFrame is not empty before passing to Backtest
if df.empty or len(df) < max(SmaCross.n1, SmaCross.n2): # Need enough data for longest SMA
    print(f"Error: DataFrame has insufficient data for the strategy (requires at least {max(SmaCross.n1, SmaCross.n2)} data points).")
    exit()

bt = Backtest(df, SmaCross, cash=10000, commission=.002)
results = bt.run()
print("\nBacktest Results:")
print(results)

# Assuming your previous code has run successfully and you have:
# df: your preprocessed DataFrame
# SmaCross: your strategy class
# bt = Backtest(df, SmaCross, cash=10000, commission=.002)
# results = bt.run()

# To plot ONLY the drawdown using bt.plot():
print("Displaying only the drawdown plot using bt.plot()...")
bt.plot(
    plot_equity=False,       # Don't plot the equity curve
    plot_drawdown=True,      # DO plot the drawdown
    plot_pl=False,           # Don't plot P/L
    superimpose=False,       # Ensure plots are not superimposed if not desired;
                             # False often means separate subplots if multiple were active.
                             # For a single plot like drawdown, its effect might be minimal
                             # but good to be explicit.
    resample=False           # Optional: set to True if you want to resample data (e.g., to 'W' for weekly)
                             # Default is True, which resamples to daily if data is finer.
                             # Set to False to use original data frequency.
)

'''
try:
    bt.plot()
except Exception as e:
    print(f"\nError during plotting: {e}")
    print("Ensure you have matplotlib installed and a suitable environment for plotting (e.g., a Jupyter notebook or a script run with a GUI backend).")
'''