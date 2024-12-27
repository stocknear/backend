import os
import pandas as pd
import orjson
from dotenv import load_dotenv
import sqlite3
from datetime import datetime, timedelta
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

def convert_tape_time(data_list):
    # Iterate through the list and update the 'tape_time' field for each dictionary
    for item in data_list:
        utc_time = datetime.strptime(item['tape_time'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
        new_york_tz = pytz.timezone("America/New_York")
        ny_time = utc_time.astimezone(new_york_tz)
        item['tape_time'] = ny_time.strftime("%Y-%m-%d %H:%M:%S")
    return data_list


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
        url = "https://api.unusualwhales.com/api/market/sector-etfs"
        response = requests.get(url, headers=headers)
        data = response.json().get('data', [])
        res_list = []
        processed_data = []
        
        for item in data:
            symbol = item['ticker']

            bearish_premium = float(item['bearish_premium'])
            bullish_premium = float(item['bullish_premium'])
            neutral_premium = calculate_neutral_premium(item)
            
            # Step 1: Replace 'full_name' with 'name' if needed
            new_item = {
                'name' if key == 'full_name' else key: safe_round(value)
                for key, value in item.items()
                if key != 'in_out_flow'
            }
            
            # Step 2: Replace 'name' values
            if str(new_item.get('name')) == 'Consumer Staples':
                new_item['name'] = 'Consumer Defensive'
            elif str(new_item.get('name')) == 'Consumer Discretionary':
                new_item['name'] = 'Consumer Cyclical'
            elif str(new_item.get('name')) == 'Health Care':
                new_item['name'] = 'Healthcare'
            elif str(new_item.get('name')) == 'Financials':
                new_item['name'] = 'Financial Services'
            elif str(new_item.get('name')) == 'Materials':
                new_item['name'] = 'Basic Materials'

            new_item['premium_ratio'] = [
                safe_round(bearish_premium),
                neutral_premium,
                safe_round(bullish_premium)
            ]

            with open(f"json/quote/{symbol}.json") as file:
                quote_data = orjson.loads(file.read())
                new_item['price'] = round(quote_data.get('price', 0), 2)
                new_item['changesPercentage'] = round(quote_data.get('changesPercentage', 0), 2)

            processed_data.append(new_item)

        return processed_data
    except Exception as e:
        print(e)
        return []


def generate_time_intervals(start_time, end_time):
    """Generate 1-minute intervals from start_time to end_time."""
    intervals = []
    current_time = start_time
    while current_time <= end_time:
        intervals.append(current_time.strftime('%Y-%m-%d %H:%M:%S'))
        current_time += timedelta(minutes=1)
    return intervals

def get_net_prem_ticks(symbol):
    # Fetch data from the API
    url = f"https://api.unusualwhales.com/api/stock/{symbol}/net-prem-ticks"
    response = requests.get(url, headers=headers)
    data = response.json().get('data', [])
    
    # Sort data by date in descending order
    data = sorted(data, key=lambda x: datetime.fromisoformat(x['date'].replace('Z', '+00:00')), reverse=True)
    
    # Convert tape_time if necessary
    data = convert_tape_time(data)
    
    # Load price list
    with open(f"json/one-day-price/{symbol}.json") as file:
        price_list = orjson.loads(file.read())
    
    # Get the start time from the earliest tape_time in data
    if not data:
        return []

    start_time = datetime.strptime(data[0]['tape_time'], '%Y-%m-%d %H:%M:%S')
    end_time = datetime.combine(start_time.date(), datetime.strptime('22:00:00', '%H:%M:%S').time())
    
    # Generate 1-minute intervals
    intervals = generate_time_intervals(start_time, end_time)
    
    # Create a dictionary for fast lookups of existing tape_time
    data_dict = {entry['tape_time']: entry for entry in data}
    
    # Populate data with 1-minute intervals
    populated_data = []
    for time in intervals:
        if time in data_dict:
            populated_data.append(data_dict[time])
        else:
            populated_data.append({
                'date': time.split(' ')[0],
                'net_call_premium': None,
                'net_call_volume': None,
                'net_put_premium': None,
                'net_put_volume': None,
                'tape_time': time,
                'close': None
            })
    
    # Add 'close' values if matches found in price_list
    matched = False
    for entry in populated_data:
        for price in price_list:
            if entry['tape_time'] == price['time']:
                entry['close'] = price['close']
                matched = True
                break  # Exit inner loop once a match is found

    # Return the populated data if matches exist; otherwise, return an empty list
    print(populated_data)
    return populated_data if matched else []

def main():
    #sector_data = get_sector_data()
    sector_data = []
    net_premium_tick_data = get_net_prem_ticks('XLC')

    if len(sector_data) > 0:
        save_json(sector_data)

if __name__ == '__main__':
    main()
