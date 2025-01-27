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
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
from GetStartEndDate import GetStartEndDate
from tqdm import tqdm

import re


load_dotenv()
fmp_api_key = os.getenv('FMP_API_KEY')
api_key = os.getenv('INTRINIO_API_KEY')

intrinio.ApiClient().set_api_key(api_key)
intrinio.ApiClient().allow_retries(True)


ny_tz = pytz.timezone('America/New_York')

today,_ =  GetStartEndDate().run()
today = today.strftime("%Y-%m-%d")


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

def parse_contract_data(option_symbol):
    # Define regex pattern to match the symbol structure
    match = re.match(r"([A-Z]+)(\d{6})([CP])(\d+)", option_symbol)
    if not match:
        raise ValueError(f"Invalid option_symbol format: {option_symbol}")
    
    ticker, expiration, option_type, strike_price = match.groups()
    
    return option_type
    

async def get_intrinio_data(ticker):
    url=f"https://api-v2.intrinio.com/options/unusual_activity/{ticker}/intraday?page_size=1000&api_key={api_key}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
    
    data = data.get('trades',[])
    if data:
        res_list = []
        for item in data:
            try:
                iso_timestamp = item['timestamp'].replace('Z', '+00:00')
                # Parse timestamp and convert to New York time
                timestamp = datetime.fromisoformat(iso_timestamp).astimezone(ny_tz)
                formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                put_call = parse_contract_data(item['contract'].replace("___","").replace("__","").replace("_",''))
                if put_call == 'C':
                    put_call = 'calls'
                else:
                    put_call = 'puts'

                res_list.append({'timestamp': formatted_time, 'put_call': put_call, 'cost_basis': item['total_value'], 'volume': item['total_size'], 'sentiment': item['sentiment']})
            except:
                pass

        res_list.sort(key=lambda x: x['timestamp'])
        return res_list
    else:
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
    with open(f"json/stocks-list/sp500_constituent.json","r") as file:
        ticker_list = orjson.loads(file.read())
        ticker_list = [item['symbol'] for item in ticker_list][:10]
    
    
    res_list = []

    # Track changes per interval
    delta_data = defaultdict(lambda: {
        'cumulative_net_call_premium': 0,
        'cumulative_net_put_premium': 0,
        'call_ask_vol': 0,
        'call_bid_vol': 0,
        'put_ask_vol': 0,
        'put_bid_vol': 0
    })

    for ticker in tqdm(['SPY']):
        '''
        with open("json/options-flow/feed/data.json", "r") as file:
            data = orjson.loads(file.read())
        '''
        data = asyncio.run(get_intrinio_data(ticker))
        

        ticker_options = [item for item in data if item['timestamp'].startswith(today)]
        ticker_options.sort(key=lambda x: x['timestamp'])


        for item in ticker_options:
            try:
                # Parse and standardize timestamp
                dt = datetime.strptime(f"{item['timestamp']}", "%Y-%m-%d %H:%M:%S")
                
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

    price_list = asyncio.run(get_stock_chart_data('SPY'))
    if len(price_list) == 0:
        with open(f"json/one-day-price/'SPY'.json") as file:
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
