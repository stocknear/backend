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

def generate_time_intervals(start_time, end_time):
    """Generate 1-minute intervals from start_time to end_time."""
    intervals = []
    current_time = start_time
    while current_time <= end_time:
        intervals.append(current_time.strftime('%Y-%m-%d %H:%M:%S'))
        current_time += timedelta(minutes=1)
    return intervals

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

            #get prem tick data:
            '''
            if symbol != 'SPY':
                prem_tick_history = get_net_prem_ticks(symbol)
                #if symbol == 'XLB':
                #    print(prem_tick_history[10])

                new_item['premTickHistory'] = prem_tick_history
            '''

            processed_data.append(new_item)

        return processed_data
    except Exception as e:
        print(e)
        return []

def get_net_prem_ticks(symbol):
    # Fetch data from the API
    url = f"https://api.unusualwhales.com/api/stock/{symbol}/net-prem-ticks"
    response = requests.get(url, headers=headers)
    data = response.json().get('data', [])
    print(data[0])
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
    
    # Initialize aggregated data with cumulative sums
    aggregated_data = {time: {
        'net_call_premium': 0,
        'net_put_premium': 0,
        'net_call_volume': 0,
        'net_put_volume': 0,
        'tape_time': time,
        'close': None
    } for time in intervals}
    
    # Variable to track cumulative sums
    cumulative_net_call_premium = 0
    cumulative_net_put_premium = 0
    cumulative_net_call_volume = 0
    cumulative_net_put_volume = 0
    
    # Aggregate data for each minute, cumulatively adding values
    for time in intervals:
        if time in data_dict:
            entry = data_dict[time]
            # Add current values to cumulative sums
            cumulative_net_call_premium += float(entry.get('net_call_premium', 0))
            cumulative_net_put_premium += float(entry.get('net_put_premium', 0))
            cumulative_net_call_volume += float(entry.get('net_call_volume', 0))
            cumulative_net_put_volume += float(entry.get('net_put_volume', 0))
        
        # Set the aggregated values for this minute
        aggregated_data[time]['net_call_premium'] = cumulative_net_call_premium
        aggregated_data[time]['net_put_premium'] = cumulative_net_put_premium
        aggregated_data[time]['net_call_volume'] = cumulative_net_call_volume
        aggregated_data[time]['net_put_volume'] = cumulative_net_put_volume

    # Populate data with aggregated results
    populated_data = list(aggregated_data.values())

    # Add 'close' values if matches found in price_list
    matched = False
    for entry in populated_data:
        for price in price_list:
            if entry['tape_time'] == price['time']:
                entry['close'] = price['close']
                matched = True
                break  # Exit inner loop once a match is found
    

    return populated_data if matched else []

def get_top_sector_tickers():
    keep_elements = ['price', 'ticker', 'name', 'changesPercentage','netPremium','netCallPremium','netPutPremium','gexRatio','gexNetChange','ivRank']
    sector_list = [
        "Basic Materials",
        "Communication Services",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Energy",
        "Financial Services",
        "Healthcare",
        "Industrials",
        "Real Estate",
        "Technology",
        "Utilities",
    ]
    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": api_key
    }
    url = "https://api.unusualwhales.com/api/screener/stocks"

    res_list = {}

    for sector in sector_list:
        querystring = {
            'order': 'net_premium',
            'order_direction': 'desc',
            'sectors[]': sector
        }

        response = requests.get(url, headers=headers, params=querystring)
        data = response.json().get('data', [])

        updated_data = []
        for item in data[:10]:
            try:
                new_item = {key: safe_round(value) for key, value in item.items()}
                with open(f"json/quote/{item['ticker']}.json") as file:
                    quote_data = orjson.loads(file.read())
                    new_item['name'] = quote_data['name']
                    new_item['price'] = round(float(quote_data['price']), 2)
                    new_item['changesPercentage'] = round(float(quote_data['changesPercentage']), 2)
                    
                    new_item['ivRank'] = int(new_item['iv_rank'])
                    new_item['gexRatio'] = new_item['gex_ratio']
                    new_item['gexNetChange'] = new_item['gex_net_change']
                    new_item['netCallPremium'] = new_item['net_call_premium']
                    new_item['netPutPremium'] = new_item['net_put_premium']

                    new_item['netPremium'] = abs(new_item['netCallPremium'] - new_item['netPutPremium'])
                # Filter new_item to keep only specified elements
                filtered_item = {key: new_item[key] for key in keep_elements if key in new_item}
                updated_data.append(filtered_item)
            except Exception as e:
                print(f"Error processing ticker {item.get('ticker', 'unknown')}: {e}")

        # Add rank to each item
        for rank, item in enumerate(updated_data, 1):
            item['rank'] = rank
        res_list[sector] = updated_data

    return res_list



def main():
    
    sector_data = get_sector_data()
    top_sector_tickers = get_top_sector_tickers()
    data = {'sectorData': sector_data, 'topSectorTickers': top_sector_tickers}
    if len(data) > 0:
        save_json(data)
    

    #get_net_prem_ticks('XLB')


if __name__ == '__main__':
    main()
