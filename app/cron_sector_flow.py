import os
import pandas as pd
import orjson
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import pytz
import requests  # Add missing import


load_dotenv()
api_key = os.getenv('UNUSUAL_WHALES_API_KEY')
headers = {
    "Accept": "application/json, text/plain",
    "Authorization": api_key
}


def save_json(data):
    directory = "json/sector-flow"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory}/data.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))

def safe_round(value):
    """Attempt to convert a value to float and round it. Return the original value if not possible."""
    try:
        return round(float(value), 2)
    except (ValueError, TypeError):
        return value

def calculate_neutral_premium(data_item):
    """Calculate the neutral premium for a data item."""
    call_premium = float(data_item['call_premium'])
    put_premium = float(data_item['put_premium'])
    bearish_premium = float(data_item['bearish_premium'])
    bullish_premium = float(data_item['bullish_premium'])
    
    total_premiums = bearish_premium + bullish_premium
    observed_premiums = call_premium + put_premium
    neutral_premium = observed_premiums - total_premiums
    
    return safe_round(neutral_premium)

def get_sector_data():
    try:
        url ="https://api.unusualwhales.com/api/market/sector-etfs"
        response = requests.get(url, headers=headers)
        data = response.json().get('data',[])
        res_list = []
        processed_data = []
        for item in data:
            symbol = item['ticker']


            bearish_premium = float(item['bearish_premium'])
            bullish_premium = float(item['bullish_premium'])
            neutral_premium = calculate_neutral_premium(item)
            
            new_item = {
                key if key != 'full_name' else 'name': safe_round(value)
                for key, value in item.items()
                if key != 'in_out_flow'
            }
            new_item['premium_ratio'] = [
                safe_round(bearish_premium),
                neutral_premium,
                safe_round(bullish_premium)
            ]

            with open(f"json/quote/{symbol}.json") as file:
                quote_data = orjson.loads(file.read())
                new_item['price'] = round(quote_data.get('price',0),2)
                new_item['changesPercentage'] = round(quote_data.get('changesPercentage',0),2)

            processed_data.append(new_item)

        return processed_data
    except Exception as e:
        print(e)
        return []

def main():
    sector_data = get_sector_data()

    if len(sector_data) > 0:
        save_json(sector_data)

if __name__ == '__main__':
    main()
