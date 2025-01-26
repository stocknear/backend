import os
import pandas as pd
import orjson
from dotenv import load_dotenv
import sqlite3
from datetime import datetime, timedelta
from GetStartEndDate import GetStartEndDate
import asyncio
import aiohttp
import pytz
import requests  # Add missing import
from collections import defaultdict

load_dotenv()
fmp_api_key = os.getenv('FMP_API_KEY')
ny_tz = pytz.timezone('America/New_York')


def save_json(data):
    directory = "json/market-flow"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory}/data.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))

# Function to convert and match timestamps
def add_close_to_data(price_list, data):
    for entry in data:
        formatted_time = entry['timestamp']
        
        # Match with price_list
        for price in price_list:
            if price['date'] == formatted_time:
                entry['close'] = price['close']
                break  # Match found, no need to continue searching
    return data

def convert_timestamps(data_list):
    ny_tz = pytz.timezone('America/New_York')
    
    for item in data_list:
        try:
            # Get the timestamp and split on '.'
            timestamp = item['timestamp']
            base_time = timestamp.split('.')[0]
            
            # Handle microseconds if present
            if '.' in timestamp:
                microseconds = timestamp.split('.')[1].replace('Z', '')
                microseconds = microseconds.ljust(6, '0')  # Pad with zeros if needed
                base_time = f"{base_time}.{microseconds}"
            
            # Replace 'Z' with '+00:00' (for UTC)
            base_time = base_time.replace('Z', '+00:00')
            
            # Parse the timestamp
            dt = datetime.fromisoformat(base_time)
            
            # Ensure the datetime is timezone-aware (assumed to be UTC initially)
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            
            # Convert the time to New York timezone (automatically handles DST)
            ny_time = dt.astimezone(ny_tz)
            
            # Optionally, format to include date and time
            item['timestamp'] = ny_time.strftime('%Y-%m-%d %H:%M:%S')
            
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format: {item['timestamp']} - Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing timestamp: {item['timestamp']} - Error: {str(e)}")
    
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

async def get_stock_chart_data(ticker):
    start_date_1d, end_date_1d = GetStartEndDate().run()
    start_date = start_date_1d.strftime("%Y-%m-%d")
    end_date = end_date_1d.strftime("%Y-%m-%d")

    url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{ticker}?from={start_date}&to={end_date}&apikey={fmp_api_key}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                data = sorted(data, key=lambda x: x['date'])
                return data
            else:
                return []



def get_market_tide(interval_5m=False):
    ticker_list = ['SPY']
    res_list = []

    for ticker in ticker_list:
        with open("json/options-flow/feed/data.json", "r") as file:
            data = orjson.loads(file.read())

        # Filter and sort data
        ticker_options = [item for item in data if item['ticker'] == ticker]
        ticker_options.sort(key=lambda x: x['time'])

        # Track changes per interval
        delta_data = defaultdict(lambda: {
            'cumulative_net_call_premium': 0,
            'cumulative_net_put_premium': 0,
            'call_ask_vol': 0,
            'call_bid_vol': 0,
            'put_ask_vol': 0,
            'put_bid_vol': 0
        })

        for item in ticker_options:
            try:
                # Parse and standardize timestamp
                dt = datetime.strptime(f"{item['date']} {item['time']}", "%Y-%m-%d %H:%M:%S")
                
                # Truncate to start of minute (for 1m summaries)
                dt = dt.replace(second=0, microsecond=0)
                
                # Adjust for 5-minute intervals if needed
                if interval_5m:
                    dt -= timedelta(minutes=dt.minute % 5)
                
                rounded_ts = dt.strftime("%Y-%m-%d %H:%M:%S")

                # Extract metrics
                cost = float(item.get("cost_basis", 0))
                sentiment = item.get("sentiment", "").lower()
                put_call = item.get("put_call", "").lower()
                vol = int(item.get("volume", 1))

                # Update premium metrics
                if put_call == "calls":
                    if sentiment == "bullish":
                        delta_data[rounded_ts]['cumulative_net_call_premium'] += cost
                        delta_data[rounded_ts]['call_ask_vol'] += vol
                    elif sentiment == "bearish":
                        delta_data[rounded_ts]['cumulative_net_call_premium'] -= cost
                        delta_data[rounded_ts]['call_bid_vol'] += vol
                elif put_call == "puts":
                    if sentiment == "bullish":
                        delta_data[rounded_ts]['cumulative_net_put_premium'] -= cost
                        delta_data[rounded_ts]['put_ask_vol'] += vol
                    elif sentiment == "bearish":
                        delta_data[rounded_ts]['cumulative_net_put_premium'] += cost
                        delta_data[rounded_ts]['put_bid_vol'] += vol

            except Exception as e:
                print(f"Error processing item: {e}")

        # Calculate cumulative values over time
        sorted_ts = sorted(delta_data.keys())
        cumulative = {
            'net_call_premium': 0,
            'net_put_premium': 0,
            'call_ask': 0,
            'call_bid': 0,
            'put_ask': 0,
            'put_bid': 0
        }

        for ts in sorted_ts:
            # Update cumulative values
            cumulative['net_call_premium'] += delta_data[ts]['cumulative_net_call_premium']
            cumulative['net_put_premium'] += delta_data[ts]['cumulative_net_put_premium']
            cumulative['call_ask'] += delta_data[ts]['call_ask_vol']
            cumulative['call_bid'] += delta_data[ts]['call_bid_vol']
            cumulative['put_ask'] += delta_data[ts]['put_ask_vol']
            cumulative['put_bid'] += delta_data[ts]['put_bid_vol']

            # Calculate derived metrics
            call_volume = cumulative['call_ask'] + cumulative['call_bid']
            put_volume = cumulative['put_ask'] + cumulative['put_bid']
            net_volume = (cumulative['call_ask'] - cumulative['call_bid']) - \
                        (cumulative['put_ask'] - cumulative['put_bid'])

            res_list.append({
                'timestamp': ts,
                'ticker': ticker,
                'net_call_premium': cumulative['net_call_premium'],
                'net_put_premium': cumulative['net_put_premium'],
                'call_volume': call_volume,
                'put_volume': put_volume,
                'net_volume': net_volume
            })

    res_list.sort(key=lambda x: x['timestamp'])

    price_list = asyncio.run(get_stock_chart_data(ticker))
    if len(price_list) == 0:
        with open(f"json/one-day-price/{ticker}.json") as file:
            price_list = orjson.loads(file.read())

    data = add_close_to_data(price_list, res_list)


    return res_list


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
                    
                    new_item['ivRank'] = round(float(new_item['iv_rank']),2)
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


def get_top_spy_tickers():
    keep_elements = ['price', 'ticker', 'name', 'changesPercentage','netPremium','netCallPremium','netPutPremium','gexRatio','gexNetChange','ivRank']

    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": api_key
    }
    url = "https://api.unusualwhales.com/api/screener/stocks"

    querystring = {"is_s_p_500":"true"}
        

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
                
                new_item['ivRank'] = round(float(new_item['iv_rank']),2)
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

    return updated_data



def main():
    
    market_tide = get_market_tide()
    data = {'marketTide': market_tide}
    '''
    sector_data = get_sector_data()
    top_sector_tickers = get_top_sector_tickers()
    top_spy_tickers = get_top_spy_tickers()
    top_sector_tickers['SPY'] = top_spy_tickers
    data = {'sectorData': sector_data, 'topSectorTickers': top_sector_tickers, 'marketTide': market_tide}
    '''
    if len(data) > 0:
        save_json(data)
    

if __name__ == '__main__':
    main()
