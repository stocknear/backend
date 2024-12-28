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
ny_tz = pytz.timezone('America/New_York')


def save_json(data):
    directory = "json/market-flow"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory}/data.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))

# Function to convert and match timestamps
def add_close_to_data(price_list, data):
    for entry in data:
        # Convert timestamp to New York time and desired format
        timestamp = datetime.fromisoformat(entry['timestamp']).astimezone(ny_tz)
        formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Match with price_list
        for price in price_list:
            if price['time'] == formatted_time:
                entry['close'] = price['close']
                break  # Match found, no need to continue searching
    return data

def convert_time(data_list):
    # Iterate through the list and update the 'tape_time' field for each dictionary
    for item in data_list:
        utc_time = datetime.strptime(item['timestamp'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
        new_york_tz = pytz.timezone("America/New_York")
        ny_time = utc_time.astimezone(new_york_tz)
        item['timestamp'] = ny_time.strftime("%Y-%m-%d %H:%M:%S")
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
                prem_tick_history = get_etf_tide(symbol)
                #if symbol == 'XLB':
                #    print(prem_tick_history[10])

                new_item['premTickHistory'] = prem_tick_history
            '''

            processed_data.append(new_item)

        return processed_data
    except Exception as e:
        print(e)
        return []

def get_market_tide():
    # Fetch data from the API
    querystring = {"interval_5m":"false"}
    url = f"https://api.unusualwhales.com/api/market/market-tide"
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json().get('data', [])

    with open(f"json/one-day-price/SPY.json") as file:
        price_list = orjson.loads(file.read())

    data = add_close_to_data(price_list, data)

    return data
  

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
    market_tide = get_market_tide()
    sector_data = get_sector_data()
    top_sector_tickers = get_top_sector_tickers()
    data = {'sectorData': sector_data, 'topSectorTickers': top_sector_tickers, 'marketTide': market_tide}
    if len(data) > 0:
        save_json(data)
    


if __name__ == '__main__':
    main()
