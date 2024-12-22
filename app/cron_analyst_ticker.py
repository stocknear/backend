from benzinga import financial_data
import requests
from datetime import datetime, timedelta, date
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import norm
import time
import sqlite3
import ujson
import math
import statistics
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

fin = financial_data.Benzinga(api_key)


query_template = """
    SELECT date,close
    FROM "{ticker}"
    WHERE date BETWEEN ? AND ?
"""
end_date = datetime.today().date()
start_date_12m = end_date - timedelta(days=365)

def filter_latest_entries(data):
    latest_entries = {}
    
    for entry in data:
        try:
            # Combine 'analyst' and 'name' to create a unique key
            key = (entry['analyst'], entry['name'])
            
            # Convert date to a comparable format (datetime object)
            date_time_str = f"{entry['date']} {entry['time']}"
            date_time = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
            
            # If this combination is not in latest_entries or if it's a newer date, update the dictionary
            if key not in latest_entries or date_time > latest_entries[key][0]:
                latest_entries[key] = (date_time, entry)
        except Exception as e:
            print(f"Error processing entry: {e}")
            pass

    # Return only the latest entries
    return [entry for _, entry in latest_entries.values()]

# Example usage
# filtered_list = remove_duplicate_names(your_original_list)
# Example usage
# filtered_list = remove_duplicate_ids(your_original_list)


# Define a function to remove duplicates based on a key
def remove_duplicates(data, key):
    seen = set()
    new_data = []
    for item in data:
        if item[key] not in seen:
            seen.add(item[key])
            new_data.append(item)
    return new_data



def get_summary(res_list):
    # Get the latest summary of ratings from the last 12 months
    end_date = date.today()
   

   
    # Filter the data for the last 12 months and consider the last N ratings
    #Furthermore consider only the last rating of the analyst if he provided multiple in the last 12 months
    #filtered data is needed for the recommendation list
    filtered_data = [item for item in res_list if start_date_12m <= datetime.strptime(item['date'], '%Y-%m-%d').date() <= end_date]
    #unique list is needed for analyst summary rating
    unique_filtered_data = filter_latest_entries(filtered_data)[:30]

    # Initialize dictionary to store the latest price target for each analyst
    latest_pt_current = defaultdict(list)
    
    # Iterate through the filtered data to collect pt_current for each analyst
    for item in unique_filtered_data:
        if 'adjusted_pt_current' in item and item['adjusted_pt_current']:
            analyst_name = item['analyst_name']
            try:
                pt_current_value = float(item['adjusted_pt_current'])
                # Collect all pt_current values for each analyst
                latest_pt_current[analyst_name].append(pt_current_value)
            except (ValueError, TypeError):
                print(f"Invalid pt_current value for analyst '{analyst_name}': {item['adjusted_pt_current']}")
    
    # Compute statistics for price targets
    pt_current_values = [val for sublist in latest_pt_current.values() for val in sublist]
    #remove outliers to keep high and low price target reasonable
    q1, q3 = np.percentile(pt_current_values, [25, 75])
    iqr = q3 - q1
    pt_current_values = [x for x in pt_current_values if (q1 - 1.5 * iqr) <= x <= (q3 + 1.5 * iqr)]

    # Compute different price target metrics if there are values, otherwise set to 0
    if pt_current_values:
        median_pt_current = statistics.median(pt_current_values)
        avg_pt_current = statistics.mean(pt_current_values)
        low_pt_current = min(pt_current_values)
        high_pt_current = max(pt_current_values)
    else:
        median_pt_current = avg_pt_current = low_pt_current = high_pt_current = 0
    
    # Initialize recommendation tracking
    rating_hierarchy = {'Strong Sell': 0, 'Sell': 1, 'Hold': 2, 'Buy': 3, 'Strong Buy': 4}
    
    # Track monthly recommendations
    monthly_recommendations = {}
    
    # Iterate through the filtered data to track monthly recommendations
    for item in filtered_data:
        # Extract month from the date
        item_date = datetime.strptime(item['date'], '%Y-%m-%d')
        month_key = item_date.strftime('%Y-%m-01')
        
        # Initialize month's recommendation counts if not exists
        if month_key not in monthly_recommendations:
            monthly_recommendations[month_key] = {
                'Strong Sell': 0,
                'Sell': 0,
                'Hold': 0,
                'Buy': 0,
                'Strong Buy': 0
            }
        
        # Check and increment recommendation count for the month
        if 'rating_current' in item and item['rating_current'] in rating_hierarchy:
            monthly_recommendations[month_key][item['rating_current']] += 1
    
    # Convert monthly recommendations to a sorted list
    recommendation_list = []
    for month in sorted(monthly_recommendations.keys()):
        month_data = monthly_recommendations[month]
        recommendation_list.append({
            'date': month,
            'Strong Sell': month_data['Strong Sell'],
            'Sell': month_data['Sell'],
            'Hold': month_data['Hold'],
            'Buy': month_data['Buy'],
            'Strong Buy': month_data['Strong Buy']
        })
    
    # Compute consensus ratings (similar to previous implementation)
    consensus_ratings = defaultdict(str)
    for item in unique_filtered_data:
        if 'rating_current' in item and item['rating_current'] and 'analyst_name' in item and item['analyst_name']:
            try:
                analyst_name = item['analyst_name']
                current_rating = item['rating_current']
                if current_rating in rating_hierarchy:
                    consensus_ratings[analyst_name] = current_rating
            except:
                pass
    
    # Compute the consensus rating based on the most frequent rating among analysts
    consensus_rating_counts = defaultdict(int)
    for rating in consensus_ratings.values():
        consensus_rating_counts[rating] += 1
    consensus_rating = max(consensus_rating_counts, key=consensus_rating_counts.get)
    
    # Sum up all Buy, Sell, Hold for the progress bar in sveltekit
    data_dict = dict(consensus_rating_counts)
    buy_total = data_dict.get('Strong Buy', 0) + data_dict.get('Buy', 0)
    sell_total = data_dict.get('Strong Sell', 0) + data_dict.get('Sell', 0)
    hold_total = data_dict.get('Hold', 0)
    
    # Count unique analysts
    unique_analyst_names = set()
    numOfAnalyst = len(unique_filtered_data)
    '''
    for item in filtered_data:
        if item['analyst_name'] not in unique_analyst_names:
            unique_analyst_names.add(item['analyst_name'])
            numOfAnalyst += 1
    '''
    
    # Update stats dictionary with new keys including recommendationList
    stats = {
        'numOfAnalyst': numOfAnalyst, 
        'consensusRating': consensus_rating, 
        'medianPriceTarget': round(median_pt_current, 2),
        'avgPriceTarget': round(avg_pt_current, 2),
        'lowPriceTarget': round(low_pt_current, 2),
        'highPriceTarget': round(high_pt_current, 2),
        'recommendationList': recommendation_list
    }
    
    categorical_ratings = {'Buy': buy_total, 'Sell': sell_total, 'Hold': hold_total}
    
    res = {**stats, **categorical_ratings}
    return res

def run(chunk, analyst_list, con):
    start_date = datetime(2015, 1, 1)
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    company_tickers = ','.join(chunk)
    res_list = []
    
    for page in range(0, 500):
        try:
            data = fin.ratings(company_tickers=company_tickers, page=page, pagesize=1000, date_from=start_date_str, date_to=end_date_str)
            data = ujson.loads(fin.output(data))['ratings']
            res_list += data
        except:
            break

    res_list = [item for item in res_list if item.get('analyst_name')]
    for ticker in chunk:
        try:
            ticker_filtered_data = [item for item in res_list if item['ticker'] == ticker]
            if len(ticker_filtered_data) != 0:
                for item in ticker_filtered_data:
                    try:
                        if item['rating_current'] == 'Strong Sell' or item['rating_current'] == 'Strong Buy':
                            pass
                        elif item['rating_current'] == 'Accumulate' and item['rating_prior'] == 'Buy':
                            item['rating_current'] = 'Buy'
                        elif item['rating_current'] == 'Neutral':
                            item['rating_current'] = 'Hold'
                        elif item['rating_current'] == 'Equal-Weight' or item['rating_current'] == 'Sector Weight' or item['rating_current'] == 'Sector Perform':
                            item['rating_current'] = 'Hold'
                        elif item['rating_current'] == 'In-Line':
                            item['rating_current'] = 'Hold'
                        elif item['rating_current'] == 'Outperform' and item['action_company'] == 'Downgrades':
                            item['rating_current'] = 'Hold'
                        elif item['rating_current'] == 'Negative':
                            item['rating_current'] = 'Sell'
                        elif (item['rating_current'] == 'Outperform' or item['rating_current'] == 'Overweight') and (item['action_company'] == 'Reiterates' or item['action_company'] == 'Initiates Coverage On'):
                            item['rating_current'] = 'Buy'
                            item['action_company'] = 'Initiates'
                        elif item['rating_current'] == 'Market Outperform' and (item['action_company'] == 'Maintains' or item['action_company'] == 'Reiterates'):
                            item['rating_current'] = 'Buy'
                        elif item['rating_current'] == 'Outperform' and (item['action_company'] == 'Maintains' or item['action_pt'] == 'Announces' or item['action_company'] == 'Upgrades'):
                            item['rating_current'] = 'Buy'
                        elif item['rating_current'] == 'Buy' and (item['action_company'] == 'Raises' or item['action_pt'] == 'Raises'):
                            item['rating_current'] = 'Strong Buy'
                        elif item.get("rating_prior",None) == "Buy" and item.get("rating_current",None) == "Buy" and (float(item.get("adjusted_pt_prior", 0)) < float(item.get('adjusted_pt_current', 0))):
                            item["rating_current"] = "Strong Buy"
                        elif item['rating_current'] == 'Overweight' and (item['action_company'] == 'Maintains' or item['action_company'] == 'Upgrades' or item['action_company'] == 'Reiterates' or item['action_pt'] == 'Raises'):
                            item['rating_current'] = 'Buy'
                        elif item['rating_current'] == 'Positive' or item['rating_current'] == 'Sector Outperform':
                            item['rating_current'] = 'Buy'
                        elif item['rating_current'] == 'Underperform' or item['rating_current'] == 'Underweight':
                            item['rating_current'] = 'Sell'
                        elif item['rating_current'] == 'Reduce' and (item['action_company'] == 'Downgrades' or item['action_pt'] == 'Lowers'):
                            item['rating_current'] = 'Sell'
                        elif item['rating_current'] == 'Sell' and item['action_pt'] == 'Announces':
                            item['rating_current'] = 'Strong Sell'
                        elif item['rating_current'] == 'Market Perform':
                            item['rating_current'] = 'Hold'
                        elif item['rating_prior'] == 'Outperform' and item['action_company'] == 'Downgrades':
                            item['rating_current'] = 'Hold'
                        elif item['rating_current'] == 'Peer Perform' and item['rating_prior'] == 'Peer Perform':
                            item['rating_current'] = 'Hold'
                        elif item['rating_current'] == 'Peer Perform' and item['action_pt'] == 'Announces':
                            item['rating_current'] = 'Hold'
                            item['action_company'] = 'Initiates'
                    except:
                        pass

                summary = get_summary(ticker_filtered_data)
                
                try:
                    # Add historical price for the last 12 months
                    query = query_template.format(ticker=ticker)
                    df_12m = pd.read_sql_query(query, con, params=(start_date_12m, end_date)).round(2)
                    df_12m['date'] = pd.to_datetime(df_12m['date'])

                    df_12m_last_per_month = df_12m.groupby(df_12m['date'].dt.to_period('M')).tail(1)
                    past_price_list = [{"date": row['date'].strftime('%Y-%m-%d'), "close": row['close']} for _, row in df_12m_last_per_month.iterrows()]
                    summary["pastPriceList"] = past_price_list
                except:
                    summary["pastPriceList"] = []

                # Get ratings of each analyst
                with open(f"json/analyst/summary/{ticker}.json", 'w') as file:
                    ujson.dump(summary, file)

                for item1 in ticker_filtered_data:
                    for item2 in analyst_stats_list:
                        if item1['analyst'] == item2['companyName'] and item1['analyst_name'] == item2['analystName']:
                            item1['analystId'] = item2['analystId']
                            item1['analystScore'] = item2['analystScore']
                            break
                        elif item1['analyst_name'] == item2['analystName']:
                            item1['analystId'] = item2['analystId']
                            item1['analystScore'] = item2['analystScore']
                            break

                desired_keys = ['date', 'action_company', 'rating_current', 'adjusted_pt_current', 'adjusted_pt_prior', 'analystId', 'analystScore', 'analyst', 'analyst_name']

                ticker_filtered_data = [
                    {key: item[key] if key in item else None for key in desired_keys}
                    for item in ticker_filtered_data
                ]

                with open(f"json/analyst/history/{ticker}.json", 'w') as file:
                    ujson.dump(ticker_filtered_data, file)
        except Exception as e:
            print(e)



try:
    con = sqlite3.connect('stocks.db')
    stock_cursor = con.cursor()
    stock_cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols =[row[0] for row in stock_cursor.fetchall()]

    	
    #Save all analyst data in raw form for the next step
    with open(f"json/analyst/all-analyst-data.json", 'r') as file:
    	analyst_stats_list = ujson.load(file)

    chunk_size = len(stock_symbols) // 300  # Divide the list into N chunks
    chunks = [stock_symbols[i:i + chunk_size] for i in range(0, len(stock_symbols), chunk_size)]
    #chunks = [['NVDA']]
    for chunk in chunks:
        run(chunk, analyst_stats_list, con)

except Exception as e:
    print(e)

finally:
	con.close()