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


def get_overview_data(sector_ticker):
    """
    Collect overview data for tickers in holdings with put/call size, open interest, and ratios
    Returns summed results for all tickers
    """
    # Load the options flow data
    with open("json/options-flow/feed/data.json", "r") as file:
        all_data = orjson.loads(file.read())
    
    # Load ETF holdings data
    with open(f"json/etf/holding/{sector_ticker}.json", "r") as file:
        holdings_data = orjson.loads(file.read())
        ticker_weights = {item['symbol']: item['weightPercentage'] for item in holdings_data['holdings']}
    
    # Initialize total aggregation variables
    total_put_size = 0
    total_call_size = 0
    total_put_oi = 0
    total_call_oi = 0
    
    # Process each ticker in holdings
    for ticker in tqdm(ticker_weights.keys(), desc="Processing overview data"):
        # Filter data for the current ticker
        ticker_data = [item for item in all_data if item.get('ticker') == ticker]
        
        if not ticker_data:
            continue
        
        # Aggregate data for the ticker
        for item in ticker_data:
            try:
                volume = int(item.get("volume", 0))
                open_interest = int(item.get("open_interest", 0))
                put_call = item.get("put_call", "")
                
                if put_call == "Calls":
                    total_call_size += volume
                    total_call_oi += open_interest
                elif put_call == "Puts":
                    total_put_size += volume
                    total_put_oi += open_interest
                    
            except Exception as e:
                print(f"Error processing item for ticker {ticker}: {e}")
                continue
    
    # Calculate put/call ratios for the totals
    put_call_size_ratio = total_put_size / total_call_size if total_call_size > 0 else 0
    put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
    
    # Return summed overview data
    overview_data = {
        'putVol': total_put_size,
        'callVol': total_call_size,
        'putOI': total_put_oi,
        'callOI': total_call_oi,
        'pcVol': round(put_call_size_ratio, 2),
        'pcOI': round(put_call_oi_ratio, 2),
        'date': datetime.today().strftime("%Y-%m-%d"),
    }
    
    return overview_data


def get_30_day_average_data(sector_ticker):
    """
    Calculate 30-day average for total size (put+call) and open interest for holdings tickers
    """
    # Load ETF holdings data
    with open(f"json/etf/holding/{sector_ticker}.json", "r") as file:
        holdings_data = orjson.loads(file.read())
        ticker_weights = {item['symbol']: item['weightPercentage'] for item in holdings_data['holdings']}
    
    # Generate list of dates for the last 30 days
    end_date = datetime.now()
    date_list = []
    for i in range(30):
        date = end_date - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        date_list.append(date_str)
    
    daily_totals = []
    
    # Process each date
    for date_str in tqdm(date_list, desc="Processing 30-day historical data"):
        file_path = f"json/options-historical-data/flow-data/{date_str}.json"
        
        try:
            with open(file_path, "r") as file:
                daily_data = orjson.loads(file.read())
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
        
        # Initialize daily aggregation variables
        daily_total_size = 0
        daily_total_oi = 0
        
        # Process each ticker in holdings for this date
        for ticker in ticker_weights.keys():
            # Filter data for the current ticker
            ticker_data = [item for item in daily_data if item.get('ticker') == ticker]
            
            # Aggregate data for the ticker
            for item in ticker_data:
                try:
                    volume = int(item.get("volume", 0))
                    open_interest = int(item.get("open_interest", 0))
                    
                    daily_total_size += volume
                    daily_total_oi += open_interest
                    
                except Exception as e:
                    print(f"Error processing item for ticker {ticker} on {date_str}: {e}")
                    continue
        
        if daily_total_size > 0 or daily_total_oi > 0:
            daily_totals.append({
                'date': date_str,
                'total_size': daily_total_size,
                'total_oi': daily_total_oi
            })
    
    # Calculate 30-day averages
    if daily_totals:
        avg_total_size = sum(day['total_size'] for day in daily_totals) / len(daily_totals)
        avg_total_oi = sum(day['total_oi'] for day in daily_totals) / len(daily_totals)
        
        avg_data = {
            'avg30Vol': round(avg_total_size, 0),
            'avg30OI': round(avg_total_oi, 0),
        }
    else:
        avg_data = {
            'avg30Vol': 0,
            'avg30OI': 0,
        }
    
    return avg_data


def get_sector_flow_analysis():
    """Calculate premium flow by sector using SP500 tickers and stockdeck sector data"""
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
    
    # Load SP500 ticker list
    with open("json/stocks-list/list/sp500.json", "r") as file:
        sp500_data = orjson.loads(file.read())
        sp500_tickers = [item['symbol'] for item in sp500_data]
    
    # Load the options flow data
    with open("json/options-flow/feed/data.json", "r") as file:
        all_data = orjson.loads(file.read())
    
    # Group SP500 tickers by sector using stockdeck data
    sector_tickers = defaultdict(list)
    
    for ticker in tqdm(sp500_tickers, desc="Loading sector data"):
        try:
            # Load stockdeck data for this ticker to get sector information
            with open(f"json/stockdeck/{ticker}.json", "r") as file:
                stockdeck_data = orjson.loads(file.read())
                sector = stockdeck_data.get('sector')
                if sector and sector in sector_list:
                    sector_tickers[sector].append(ticker)
        except:
            # Skip if stockdeck file doesn't exist
            continue
    
    sector_premium_data = {}
    
    # Calculate premium flow for each sector
    for sector in sector_list:
        tickers_in_sector = sector_tickers.get(sector, [])
        total_call_premium = 0
        total_put_premium = 0
        
        # Calculate call and put premiums separately for all tickers in this sector
        for ticker in tickers_in_sector:
            # Filter options flow data for this ticker
            ticker_data = [item for item in all_data if item.get('ticker') == ticker]
            
            for item in ticker_data:
                try:
                    cost = float(item.get("cost_basis", 0))
                    put_call = item.get("put_call", "")
                    
                    if put_call == "Calls":
                        total_call_premium += cost
                    elif put_call == "Puts":
                        total_put_premium += cost
                except:
                    continue
        
        total_premium = total_call_premium + total_put_premium
        
        sector_premium_data[sector] = {
            'sector': sector,
            'callPrem': int(total_call_premium),
            'putPrem': int(total_put_premium),
            'totalPremium': int(total_premium),
        }
    
    # Sort sectors by total premium (highest to lowest)
    sector_flow = sorted(sector_premium_data.values(), key=lambda x: x['totalPremium'], reverse=True)
    
    return sector_flow

def get_market_flow():
    market_tide = get_sector_data(sector_ticker="SPY")
    time_str = market_tide[0]['time']
    date_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

    overview = get_overview_data(sector_ticker="SPY")
    avg_30_day = get_30_day_average_data(sector_ticker="SPY")
    sector_flow = get_sector_flow_analysis()
    
    data = {
        'date': date_obj.strftime("%b %d, %Y"),
        'marketTide': market_tide, 
        'overview': {**overview,**avg_30_day},
        'sectorFlow': sector_flow
    }
    if data:
        save_json(data, 'data')

    

if __name__ == '__main__':
    get_market_flow()