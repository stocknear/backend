import pandas as pd
import requests
from datetime import datetime
from io import StringIO
from dotenv import load_dotenv
import os
import orjson
import numpy as np

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")


def numpy_float_handler(obj):
    if isinstance(obj, (np.float64, np.float32, np.int64, np.int32)):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d')
    raise TypeError(f"Type {type(obj)} not serializable")

def save_json(data):
    path = "json/tracker/potus"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/egg_price.json", "wb") as file:
        file.write(orjson.dumps(data, default=numpy_float_handler))

def fetch_fred_egg_prices():
    api_key = FRED_API_KEY
    series_id = 'APU0000708111'  # FRED series ID for egg prices
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        return df[['date', 'value']]
    except Exception as e:
        print(f"Error fetching data from FRED: {e}")
        return None

def analyze_egg_prices(df):
    """
    Analyzes egg price trends and generates statistics
    """
    if df is None or df.empty:
        return None
    
    # Calculate basic statistics
    stats = {
        'current_price': df['value'].iloc[-1],
        'avg_price': df['value'].mean(),
        'max_price': df['value'].max(),
        'min_price': df['value'].min(),
        'yearly_change': df['value'].iloc[-1] - df['value'].iloc[-13] if len(df) >= 13 else None
    }
    
    # Calculate year-over-year change
    df['YoY_change'] = df['value'].pct_change(periods=12) * 100
    
    return stats, df


def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return value

def main():
    # Fetch data
    df = fetch_fred_egg_prices()
    
    if df is not None:
        # Analyze data
        stats, df_analyzed = analyze_egg_prices(df)
    
        data = df_analyzed.to_dict(orient="records")
        current_date = datetime.now()

        N_years_ago = current_date.replace(year=current_date.year - 5)

        filtered_data = [
            {**entry, 
             'date': entry['date'].strftime('%Y-%m-%d'),
             'price': safe_round(entry['value']),  # Apply safe_round to 'value'
             'yoyChange': safe_round(entry['YoY_change'])  # Apply safe_round to 'YoY_change'
            }
            for entry in data if entry['date'] >= N_years_ago
        ]


        res_dict = {
        'currentPrice': round(stats['current_price'],2),
        'avgPrice': round(stats['avg_price'],2),
        'maxPrice': round(stats['max_price'],2),
        'minPrice': round(stats['min_price'],2),
        'history': filtered_data}

        if filtered_data:
            save_json(res_dict)

if __name__ == "__main__":
    main()