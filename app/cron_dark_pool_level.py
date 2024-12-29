import os
import pandas as pd
import numpy as np
import orjson
from dotenv import load_dotenv
import sqlite3
from datetime import datetime, timedelta
import pytz
from typing import List, Dict


def save_json(data, symbol):
    def convert_numpy(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Convert numpy scalar to Python scalar
        raise TypeError(f"Type is not JSON serializable: {type(obj)}")

    directory = "json/dark-pool/price-level"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory}/{symbol}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data, default=convert_numpy))

# Function to get the last 7 weekdays
def get_last_7_weekdays():
    today = datetime.today()
    weekdays = []
    
    # Start from today and go back until we have 7 weekdays
    while len(weekdays) < 7:
        if today.weekday() < 5:  # Monday to Friday are weekdays (0-4)
            weekdays.append(today)
        today -= timedelta(days=1)
    
    weekdays = [item.strftime("%Y-%m-%d") for item in weekdays]
    return weekdays


def analyze_dark_pool_levels(trades: List[Dict], 
                           size_threshold: float = 0.8, 
                           price_grouping: float = 1.0) -> Dict:
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(trades)
    
    # Convert premium strings to float values
    df['premium'] = df['premium'].apply(lambda x: float(str(x).replace(',', '')))
    
    # Round prices to group nearby levels
    df['price_level'] = (df['price'] / price_grouping).round(2) * price_grouping
    
    # Group by price level and sum volumes
    size_by_price = df.groupby('price_level').agg({
        'size': 'sum',
        'premium': 'sum'
    }).reset_index()
    
    # Calculate volume threshold
    min_size = size_by_price['size'].quantile(size_threshold)
    
    # Identify significant levels
    significant_levels = size_by_price[size_by_price['size'] >= min_size]
    
    # Sort levels by volume to get strongest levels first
    significant_levels = significant_levels.sort_values('size', ascending=False)
    
    # Separate into support and resistance based on current price
    current_price = df['price'].iloc[-1]
    
    support_levels = significant_levels[
        significant_levels['price_level'] < current_price
    ].to_dict('records')
    
    resistance_levels = significant_levels[
        significant_levels['price_level'] > current_price
    ].to_dict('records')
    
    # Calculate additional metrics
    metrics = {
        'avgTradeSize': round(df['size'].mean(),2),
        'totalPrem': round(df['premium'].sum(),2),
        'avgPremTrade': round(df['premium'].mean(),2)
    }
    
    price_level = support_levels+resistance_levels
    price_level = sorted(price_level, key=lambda x: float(x['price_level']))
    return {
        'price_level': price_level,
        'metrics': metrics,
    }

data = []
weekdays = get_last_7_weekdays()
for date in weekdays:
    try:
        with open(f"json/dark-pool/historical-flow/{date}.json", "r") as file:
            raw_data = orjson.loads(file.read())
        data +=raw_data
    except:
        pass

symbol = "GME"
res_list = [item for item in data if item['ticker'] == symbol]


dark_pool_levels = analyze_dark_pool_levels(
    trades=res_list,
    size_threshold=0.9,  # Look for levels with volume in top 20%
    price_grouping=1.0     # Group prices within $1.00
)

print(dark_pool_levels['metrics'])


top_5_elements = [{k: v for k, v in item.items() if k not in ['ticker', 'sector', 'assetType']} for item in sorted(res_list, key=lambda x: float(x['premium']), reverse=True)[:5]]
# Add rank to each item
for rank, item in enumerate(top_5_elements, 1):
    item['rank'] = rank

data = {'hottestTrades': top_5_elements, 'priceLevel': dark_pool_levels['price_level'], 'metrics': dark_pool_levels['metrics']}

if len(data) > 0:
    save_json(data, symbol)
#print(data)