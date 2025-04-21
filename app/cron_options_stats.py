from __future__ import print_function
import asyncio
import time
from datetime import datetime, timedelta
import orjson
from tqdm import tqdm
import sqlite3
from dotenv import load_dotenv
import os
import re
from statistics import mean
from collections import defaultdict


# Database connection and symbol retrieval
def get_total_symbols():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
        stocks_symbols = [row[0] for row in cursor.fetchall()]

    with sqlite3.connect('etf.db') as etf_con:
        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]
    
    with sqlite3.connect('index.db') as index_con:
        index_cursor = index_con.cursor()
        index_cursor.execute("PRAGMA journal_mode = wal")
        index_cursor.execute("SELECT DISTINCT symbol FROM indices")
        index_symbols = [row[0] for row in index_cursor.fetchall()]

    return stocks_symbols + etf_symbols +index_symbols


def save_json(data, symbol, directory):
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{symbol}.json", 'wb') as file:
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
            if price['time'] == formatted_time:
                entry['close'] = price['close']
                break  # Match found, no need to continue searching
    return data


def get_market_flow_data(ticker,interval_1m=True):
    res_list = []
    
    # Load the options flow data.
    with open("json/options-flow/feed/data.json", "r") as file:
        all_data = orjson.loads(file.read())
    
    # Load ETF holdings data and extract ticker weights.
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
                    delta_data[rounded_ts]['cumulative_net_call_premium'] += cost
                    delta_data[rounded_ts]['call_ask_vol'] += vol
                elif sentiment == "Bearish":
                    delta_data[rounded_ts]['cumulative_net_call_premium'] -= cost
                    delta_data[rounded_ts]['call_bid_vol'] += vol
            elif put_call == "Puts":
                if sentiment == "Bullish":
                    delta_data[rounded_ts]['cumulative_net_put_premium'] += cost
                    delta_data[rounded_ts]['put_ask_vol'] += vol
                elif sentiment == "Bearish":
                    delta_data[rounded_ts]['cumulative_net_put_premium'] -= cost
                    delta_data[rounded_ts]['put_bid_vol'] += vol

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
    with open(f"json/one-day-price/{ticker}.json", "r") as file:
        price_list = orjson.loads(file.read())

    # Append closing prices to the data.
    data = add_close_to_data(price_list, res_list)
    fields = ['net_call_premium', 'net_put_premium', 'call_volume', 'put_volume', 'net_volume', 'close']
    last_time = datetime.strptime(data[-1]['time'], "%Y-%m-%d %H:%M:%S")
    end_time = last_time.replace(hour=16, minute=0, second=0)
    
    while last_time < end_time:
        last_time += timedelta(minutes=1)
        data.append({
            'time': last_time.strftime("%Y-%m-%d %H:%M:%S"),
            **{field: None for field in fields}
        })
    
    return data


async def main():
    
    with open(f"json/options-flow/feed/data.json", "r") as file:
        data = orjson.loads(file.read())

    total_symbols = get_total_symbols()
    
    for symbol in tqdm(total_symbols):
        try:
            #Start of daily stats
            call_premium = 0
            put_premium = 0
            call_open_interest = 0
            put_open_interest = 0
            call_volume = 0
            put_volume = 0
            bearish_premium = 0
            bullish_premium = 0
            neutral_premium = 0

            net_call_premium = 0
            net_put_premium = 0
            net_premium = 0
            
            for item in data:
                try:
                    if item['ticker'] == symbol:
                        if item['put_call'] == 'Calls':
                            call_premium += item['cost_basis']
                            call_open_interest += int(item['open_interest'])
                            call_volume += int(item['volume'])
                        elif item['put_call'] == 'Puts':
                            put_premium += item['cost_basis']
                            put_open_interest += int(item['open_interest'])
                            put_volume += int(item['volume'])

                        if item['sentiment'] == 'Bullish':
                            bullish_premium +=item['cost_basis']
                            if item['put_call'] == 'Calls':
                                net_call_premium +=item['cost_basis']
                            elif item['put_call'] == 'Puts':
                                net_put_premium +=item['cost_basis']
                        
                        if item['sentiment'] == 'Bearish':
                            bearish_premium +=item['cost_basis']
                            if item['put_call'] == 'Calls':
                                net_call_premium -=item['cost_basis']
                            elif item['put_call'] == 'Puts':
                                net_put_premium -=item['cost_basis']

                        if item['sentiment'] == 'Neutral':
                            neutral_premium +=item['cost_basis']
                except:
                    pass

            with open(f"json/options-historical-data/companies/{symbol}.json", "r") as file:
                past_data = orjson.loads(file.read())[0]
                #previous_open_interest = past_data['total_open_interest']
                iv_rank = past_data['iv_rank']
                iv = past_data['iv']

            total_open_interest = call_open_interest+put_open_interest

            #changesPercentageOI = round((total_open_interest/previous_open_interest-1)*100, 2) if previous_open_interest > 0 else 0
            #changeOI = total_open_interest - previous_open_interest
            put_call_ratio = round(put_volume/call_volume,2) if call_volume > 0 else 0

            net_premium = net_call_premium - net_put_premium
            premium_ratio = [
                safe_round(bearish_premium),
                safe_round(neutral_premium),
                safe_round(bullish_premium)
            ]
            aggregate = {
                "call_premium": round(call_premium,0),
                "call_open_interest": round(call_open_interest,0),
                "call_volume": round(call_volume,0),
                "put_premium": round(put_premium,0),
                "put_open_interest": round(put_open_interest,0),
                "put_volume": round(put_volume,0),
                "putCallRatio": round(put_volume/call_volume,0),
                "total_open_interest": round(total_open_interest,0),
                "iv": round(iv,2),
                "iv_rank": round(iv_rank,2),
                "putCallRatio": put_call_ratio,
                "premium_ratio": premium_ratio,
                "net_call_premium": round(net_call_premium),
                "net_put_premium": round(net_put_premium),
                "net_premium": round(net_premium),
            }

            if aggregate:
                save_json(aggregate, symbol,"json/options-stats/companies")
            else:
                os.remove(f"json/options-stats/companies/{symbol}.json")

            #End of daily stats
            flow_data = get_market_flow_data(symbol)
            if flow_data:
                save_json(flow_data, symbol,"json/market-flow/companies")
            else:
                os.remove(f"json/market-flow/companies/{symbol}.json")

        except:
            try:
                os.remove(f"json/options-stats/companies/{symbol}.json")
                os.remove(f"json/market-flow/companies/{symbol}.json")
            except:
                pass

if __name__ == "__main__":
    asyncio.run(main())
