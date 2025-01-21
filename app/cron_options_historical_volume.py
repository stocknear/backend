import requests
import orjson
import re
from datetime import datetime,timedelta
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
import time
from tqdm import tqdm
from collections import defaultdict


def prepare_data(data, symbol):
    res_list = []
    #data = [entry for entry in data if datetime.strptime(entry['date'], "%Y-%m-%d") >= N_days_ago]
    
    start_date_str = data[-1]['date']
    end_date_str = data[0]['date']

    query = query_template.format(ticker=symbol)
    df_price = pd.read_sql_query(query, con if symbol in stocks_symbols else etf_con, params=(start_date_str, end_date_str)).round(2)
    df_price = df_price.rename(columns={"change_percent": "changesPercentage"})

    # Convert the DataFrame to a dictionary for quick lookups by date
    df_change_dict = df_price.set_index('date')['changesPercentage'].to_dict()
    df_close_dict = df_price.set_index('date')['close'].to_dict()

    for item in data:
        try:
            # Round numerical and numerical-string values
            new_item = {
                key: safe_round(value) if isinstance(value, (int, float, str)) else value
                for key, value in item.items()
            }

            # Add parsed fields
            new_item['volume'] = round(new_item['call_volume'] + new_item['put_volume'], 2)
            new_item['putCallRatio'] = round(new_item['put_volume']/new_item['call_volume'],2)
            new_item['avgVolumeRatio'] = round(new_item['volume'] / (round(new_item['avg_30_day_call_volume'] + new_item['avg_30_day_put_volume'], 2)), 2)
            new_item['total_premium'] = round(new_item['call_premium'] + new_item['put_premium'], 2)
            new_item['net_premium'] = round(new_item['net_call_premium'] - new_item['net_put_premium'],2)
            new_item['total_open_interest'] = round(new_item['call_open_interest'] + new_item['put_open_interest'], 2)
            
            bearish_premium = float(item['bearish_premium'])
            bullish_premium = float(item['bullish_premium'])
            neutral_premium = calculate_neutral_premium(item)

            new_item['premium_ratio'] = [
                safe_round(bearish_premium),
                neutral_premium,
                safe_round(bullish_premium)
            ]


            # Add changesPercentage if the date exists in df_change_dict
            if item['date'] in df_change_dict:
                new_item['changesPercentage'] = df_change_dict[item['date']]
            if item['date'] in df_close_dict:
                new_item['price'] = df_close_dict[item['date']]

            res_list.append(new_item)
        except:
            pass
    
    res_list = sorted(res_list, key=lambda x: x['date'])
    for i in range(1, len(res_list)):
        try:
            current_open_interest = res_list[i]['total_open_interest']
            previous_open_interest = res_list[i-1]['total_open_interest']
            changes_percentage_oi = round((current_open_interest/previous_open_interest -1)*100,2)
            res_list[i]['changesPercentageOI'] = changes_percentage_oi
        except:
            res_list[i]['changesPercentageOI'] = None

    res_list = sorted(res_list, key=lambda x: x['date'],reverse=True)

    if res_list:
        save_json(res_list, symbol)


def get_contracts_from_directory(directory: str):
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        
        # Get all tickers from filenames
        return [file.replace(".json", "") for file in os.listdir(directory) if file.endswith(".json")]
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



def aggregate_data_by_date():
    total_symbols = ['AA']
    data_by_date = defaultdict(lambda: {"volume": 0, "open_interest": 0})
    contracts_processed = 0
    
    for symbol in tqdm(total_symbols, desc="Processing symbols"):
        try:
            contract_list = get_contracts_from_directory(f"json/all-options-contracts/{symbol}")
            
            for item in tqdm(contract_list, desc=f"Processing {symbol} contracts", leave=False):
                try:
                    with open(f"json/all-options-contracts/{symbol}/{item}.json", "r") as file:
                        data = orjson.loads(file.read())
                        
                        # Process historical data
                        for entry in data.get('history', []):
                            date = entry.get('date')
                            volume = entry.get('volume')
                            open_interest = entry.get('open_interest')
                            
                            if date:
                                # Aggregate volume
                                if volume is not None:
                                    data_by_date[date]["volume"] += int(volume)
                                
                                # Aggregate open interest
                                if open_interest is not None:
                                    data_by_date[date]["open_interest"] += int(open_interest)
                                
                    contracts_processed += 1
                    
                except Exception as e:
                    print(f"Error processing contract {item} for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
            continue
    
    # Sort results by date
    sorted_results = {date: metrics for date, metrics in sorted(data_by_date.items())}
    
    return sorted_results, contracts_processed

if __name__ == '__main__':
    # Run the aggregation
    results, total_processed = aggregate_data_by_date()
    
    print("\nData by date:")
    for date, metrics in results.items():
        print(f"{date}: Volume = {metrics['volume']:,}, Open Interest = {metrics['open_interest']:,}")
    
    print(f"\nTotal contracts processed: {total_processed}")