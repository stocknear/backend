import os
import numpy as np
import pandas as pd
import orjson
from typing import List


ALL_SYMBOLS = [
    "SPY", "IWM", "QQQ", "XLB", "XLC", "XLE", "XLF", 
    "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"
]

def save_json(data):
    directory = "json/market-seasonality"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/data.json", 'wb') as file:
        file.write(orjson.dumps(data))

def get_seasonality_data():
    heatmap_data = []

    for y_index, symbol in enumerate(ALL_SYMBOLS):
        file_path = f"json/historical-price/adj/{symbol}.json"
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found for {symbol}")
            continue

        with open(file_path, "rb") as file:
            history = orjson.loads(file.read())

        # 2. Create DataFrame
        if not history:
            continue
            
        df = pd.DataFrame(history)
        
        # Ensure date is datetime object and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        # 3. Resample to Month End ('ME') and take the last 'adjClose' of that month
        # This gives us one price point per month (the closing price)
        monthly_prices = df['adjClose'].resample('ME').last()

        # 4. Calculate Percentage Change
        # (Price_Month_2 - Price_Month_1) / Price_Month_1
        monthly_returns = monthly_prices.pct_change() * 100 # Multiply by 100 for percentage

        # Drop the first NaN value (since it has no previous month to compare to)
        monthly_returns = monthly_returns.dropna()

        # 5. Group by Month Integer (1=Jan, 12=Dec) and calculate Mean
        seasonality = monthly_returns.groupby(monthly_returns.index.month).mean()

        # 6. Format for Highcharts: [x, y, value]
        # x = month index (0-11), y = symbol index, value = avg return
        for month_int, value in seasonality.items():
            x_index = month_int - 1 # Convert 1-12 (Pandas) to 0-11 (Highcharts)
            
            # Round to 2 decimal places for cleaner JSON
            avg_return = round(value, 2)
            
            heatmap_data.append([x_index, y_index, avg_return])

    if heatmap_data:
        save_json(heatmap_data)
    
if __name__ == '__main__':
    get_seasonality_data()