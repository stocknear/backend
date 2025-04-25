import os
import pandas as pd
import orjson
from dotenv import load_dotenv
import sqlite3
from datetime import datetime, timedelta
import asyncio
import aiohttp
import pytz
import requests  # Add missing import
from collections import defaultdict
from GetStartEndDate import GetStartEndDate
from tqdm import tqdm

import re


load_dotenv()
fmp_api_key = os.getenv('FMP_API_KEY')


ny_tz = pytz.timezone('America/New_York')



def save_json(data, filename):
    directory = "json/market-flow"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory}/{filename}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))


def safe_round(value):
    try:
        return round(float(value), 2)
    except (ValueError, TypeError):
        return value

def add_close_to_data(price_list, data):
    for entry in data:
        formatted_time = entry['time']
        # Match with price_list
        for price in price_list:
            # Check if 'time' key exists; if not, try 'date'
            if 'time' in price:
                price_time = price['time']
            elif 'date' in price:
                price_time = price['date']
            else:
                continue  # Skip if neither key is present

            if price_time == formatted_time:
                entry['close'] = price['close']
                break  # Match found, no need to continue searching
    return data




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




def get_sector_data(sector_ticker,interval_1m=True):
    res_list = []
    
    # Load the options flow data.
    with open("json/options-flow/feed/data.json", "r") as file:
        all_data = orjson.loads(file.read())
    
    # Load ETF holdings data and extract ticker weights.
    with open(f"json/etf/holding/{sector_ticker}.json", "r") as file:
        holdings_data = orjson.loads(file.read())
        # Build a dictionary mapping ticker symbols to their weightPercentage.
        ticker_weights = {item['symbol']: item['weightPercentage'] for item in holdings_data['holdings']}
    
    # Use a common dictionary to accumulate flows across all tickers.
    delta_data = defaultdict(lambda: {
        'cumulative_net_call_premium': 0,
        'cumulative_net_put_premium': 0,
        'call_ask_vol': 0,
        'call_bid_vol': 0,
        'put_ask_vol': 0,
        'put_bid_vol': 0
    })
    
    # Process each ticker's data using its weight.
    for ticker in tqdm(ticker_weights.keys()):
        # Convert the weight percentage to a fraction.
        weight = 1 #ticker_weights[ticker] / 100.0 #ignore weights of sector
        # Filter data for the current ticker.
        ticker_data = [item for item in all_data if item.get('ticker') == ticker]
        ticker_data.sort(key=lambda x: x['time'])
        
        for item in ticker_data:
            try:
                # Combine date and time, then truncate seconds and microseconds.
                dt = datetime.strptime(f"{item['date']} {item['time']}", "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(second=0, microsecond=0)
                
                # Adjust to the start of the minute if using 1-minute intervals.
                if interval_1m:
                    minute = dt.minute - (dt.minute % 1)
                    dt = dt.replace(minute=minute)
                
                rounded_ts = dt.strftime("%Y-%m-%d %H:%M:%S")
                
                # Extract metrics.
                cost = float(item.get("cost_basis", 0))
                sentiment = item.get("sentiment", "")
                put_call = item.get("put_call", "")
                vol = int(item.get("volume", 0))
    
                # Update metrics, scaled by the ticker's weight.
                if put_call == "Calls":
                    if sentiment == "Bullish":
                        delta_data[rounded_ts]['cumulative_net_call_premium'] += cost * weight
                        delta_data[rounded_ts]['call_ask_vol'] += vol * weight
                    elif sentiment == "Bearish":
                        delta_data[rounded_ts]['cumulative_net_call_premium'] -= cost * weight
                        delta_data[rounded_ts]['call_bid_vol'] += vol * weight
                elif put_call == "Puts":
                    if sentiment == "Bullish":
                        delta_data[rounded_ts]['cumulative_net_put_premium'] += cost * weight
                        delta_data[rounded_ts]['put_ask_vol'] += vol * weight
                    elif sentiment == "Bearish":
                        delta_data[rounded_ts]['cumulative_net_put_premium'] -= cost * weight
                        delta_data[rounded_ts]['put_bid_vol'] += vol * weight
    
            except Exception as e:
                print(f"Error processing item: {e}")
    
    # Calculate cumulative values over time.
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
        cumulative['net_call_premium'] += delta_data[ts]['cumulative_net_call_premium']
        cumulative['net_put_premium'] += delta_data[ts]['cumulative_net_put_premium']
        cumulative['call_ask'] += delta_data[ts]['call_ask_vol']
        cumulative['call_bid'] += delta_data[ts]['call_bid_vol']
        cumulative['put_ask'] += delta_data[ts]['put_ask_vol']
        cumulative['put_bid'] += delta_data[ts]['put_bid_vol']
    
        call_volume = cumulative['call_ask'] + cumulative['call_bid']
        put_volume = cumulative['put_ask'] + cumulative['put_bid']
        net_volume = (cumulative['call_ask'] - cumulative['call_bid']) - (cumulative['put_ask'] - cumulative['put_bid'])
    
        res_list.append({
            'time': ts,
            'net_call_premium': round(cumulative['net_call_premium']),
            'net_put_premium': round(cumulative['net_put_premium']),
            'call_volume': round(call_volume),
            'put_volume': round(put_volume),
            'net_volume': round(net_volume),
        })
    
    # Sort the results list by time.
    res_list.sort(key=lambda x: x['time'])
    
    # Get the price list for the sector ticker.
    price_list = asyncio.run(get_stock_chart_data(sector_ticker))
    if len(price_list) == 0:
        with open(f"json/one-day-price/{sector_ticker}.json", "r") as file:
            price_list = orjson.loads(file.read())
    
    # Append closing prices to the data.
    data = add_close_to_data(price_list, res_list)
    
    # Ensure that each minute until the specified end time (e.g., 16:01:00) is present.
    fields = ['net_call_premium', 'net_put_premium', 'call_volume', 'put_volume', 'net_volume', 'close']
    last_time = datetime.strptime(data[-1]['time'], "%Y-%m-%d %H:%M:%S")
    end_time = last_time.replace(hour=16, minute=1, second=0)
    
    while last_time < end_time:
        last_time += timedelta(minutes=1)
        data.append({
            'time': last_time.strftime("%Y-%m-%d %H:%M:%S"),
            **{field: None for field in fields}
        })
    
    return data


def get_top_tickers(sector_ticker):
    with open(f"json/etf/holding/{sector_ticker}.json", "r") as file:
        holdings_data = orjson.loads(file.read())
        # Build a dictionary mapping ticker symbols to their weightPercentage.
        data = [item['symbol'] for item in holdings_data['holdings']]

    res_list = []
    for symbol in data:
        try:
            with open(f"json/options-stats/companies/{symbol}.json","r") as file:
                stats_data = orjson.loads(file.read())
            
            new_item = {key: safe_round(value) for key, value in stats_data.items()}
            
            with open(f"json/quote/{symbol}.json") as file:
                quote_data = orjson.loads(file.read())
                new_item['symbol'] = symbol
                new_item['name'] = quote_data['name']
                new_item['price'] = round(float(quote_data['price']), 2)
                new_item['changesPercentage'] = round(float(quote_data['changesPercentage']), 2)
                
            if new_item['net_premium']:
                res_list.append(new_item)
        except:
            pass

    # Add rank to each item
    res_list = [item for item in res_list if 'net_call_premium' in item and 'net_put_premium' in item]
    res_list = sorted(res_list, key=lambda item: item['net_premium'], reverse=True)

    for rank, item in enumerate(res_list, 1):
        item['rank'] = rank

    return res_list



def get_market_flow():
    market_tide = get_sector_data(sector_ticker="SPY")
    top_pos_tickers = get_top_tickers(sector_ticker="SPY")
    #top_neg_tickers = sorted(get_top_tickers(sector_ticker="SPY"), key=lambda item: item['net_premium'])
    #for rank, item in enumerate(top_neg_tickers, 1):
    #    item['rank'] = rank

    data = {'marketTide': market_tide, 'topPosNetPremium': top_pos_tickers[:20]}
    if data:
        save_json(data, 'overview')


def get_sector_flow():
    sector_dict = {}
    top_pos_tickers_dict = {}

    for sector_ticker in ["XLB", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLRE", "XLK", "XLU"]:
        sector_data = get_sector_data(sector_ticker=sector_ticker)  
        top_pos_tickers = get_top_tickers(sector_ticker=sector_ticker)
        #top_neg_tickers = sorted(get_top_tickers(sector_ticker=sector_ticker), key=lambda item: item['net_premium'])

    
        sector_dict[sector_ticker] = sector_data
        top_pos_tickers_dict[sector_ticker] = top_pos_tickers[:20]
        #top_neg_tickers_dict[sector_ticker] = top_neg_tickers[:5]


    data = {
        'sectorFlow': sector_dict,
        'topPosNetPremium': top_pos_tickers_dict,
    }

    if data:
        save_json(data, 'sector')


def main():

    get_market_flow()
    get_sector_flow()
        
    
    

if __name__ == '__main__':
    main()