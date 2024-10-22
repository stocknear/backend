import requests
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import time
import sqlite3
import ujson
import os
from dotenv import load_dotenv
from tqdm import tqdm 
import pandas as pd
from collections import Counter
import aiohttp
import asyncio
import statistics


load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

headers = {"accept": "application/json"}

query_template = """
    SELECT date, close
    FROM "{ticker}"
    WHERE date BETWEEN ? AND ?
"""
buy_ratings = ['Outperform', 'Overweight', 'Market Outperform', 'Buy', 'Positive', 'Sector Outperform']

sell_ratings = ['Negative', 'Underperform', 'Underweight', 'Reduce', 'Sell']

ticker_price_cache = {} # Dictionary to store cached price data for tickers

# Define a function to remove duplicates based on a key
def remove_duplicates(data, key):
    seen = set()
    new_data = []
    for item in data:
        if item[key] not in seen:
            seen.add(item[key])
            new_data.append(item)
    return new_data

def extract_sector(ticker, con):
    
    query_template = f"""
    SELECT 
        sector
    FROM 
        stocks
    WHERE
        symbol = ?
    """
    try:
        df = pd.read_sql_query(query_template, con, params=(ticker,))
        sector = df['sector'].iloc[0]
    except:
        sector = None

    return sector

def calculate_rating(data):
    overall_average_return = float(data['avgReturn'])
    overall_success_rate = float(data['successRate'])
    total_ratings = int(data['totalRatings'])
    last_rating = data['lastRating']

    try:
        last_rating_date = datetime.strptime(last_rating, "%Y-%m-%d")
        difference = (datetime.now() - last_rating_date).days
    except:
        difference = 1000  # In case of None or invalid date

    if total_ratings == 0 or difference >= 600:
        return 0
    else:
        # Define weights for each factor
        weight_return = 0.4
        weight_success_rate = 0.3
        weight_total_ratings = 0.1
        weight_difference = 0.2  # Reduced weight for difference

        # Calculate weighted sum
        weighted_sum = (weight_return * overall_average_return +
                        weight_success_rate * overall_success_rate +
                        weight_total_ratings * total_ratings +
                        weight_difference * (1 / (1 + difference)))  # Adjusted weight for difference

        # Normalize the weighted sum to get a rating between 0 and 5
        min_rating = 0
        max_rating = 5
        normalized_rating = min(max(weighted_sum / (weight_return + weight_success_rate + weight_total_ratings + weight_difference), min_rating), max_rating)

        # Apply additional conditions based on total ratings and average return
        if normalized_rating >= 4:
            if total_ratings < 10:
                normalized_rating -= 2.4
            elif total_ratings < 15:
                normalized_rating -= 2.5
            elif total_ratings < 20:
                normalized_rating -= 0.75
            elif total_ratings < 30:
                normalized_rating -= 1
            elif overall_average_return <= 10:
                normalized_rating -= 1.1

        if overall_average_return <= 0:
            normalized_rating = max(normalized_rating - 2, 0)

        # Cap the rating if the last rating is older than 30 days
        if difference > 30:
            normalized_rating = min(normalized_rating, 4.5)

        if overall_success_rate < 50:
            normalized_rating = min(normalized_rating, 4.6)

        if overall_average_return < 30:
            normalized_rating = min(normalized_rating, 4.6)

        return round(normalized_rating, 2)

def get_top_stocks():
    with open(f"json/analyst/all-analyst-data.json", 'r') as file:
        analyst_stats_list = ujson.load(file)

    filtered_data = [item for item in analyst_stats_list if item['analystScore'] >= 4]

    res_list = []
    # Define the date range for the past 12 months
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)

    res_list = []
    for item in filtered_data:
        ticker_list = item['ratingsList']
        # Filter by 'Strong Buy' and ensure the rating is within the last 12 months
        ticker_list = [{'ticker': i['ticker'], 'adjusted_pt_current': i['adjusted_pt_current'], 'date': i['date']} 
                       for i in ticker_list 
                       if i['rating_current'] == 'Strong Buy' 
                       and start_date <= datetime.strptime(i['date'], '%Y-%m-%d').date() <= end_date]
        if len(ticker_list) > 0:
            res_list += ticker_list

    # Create a dictionary to store ticker occurrences and corresponding pt_current values
    ticker_data = {}
    for item in res_list:
        ticker = item['ticker']
        pt_current_str = item['adjusted_pt_current']
        if pt_current_str:  # Skip empty strings
            pt_current = float(pt_current_str)
            if ticker in ticker_data:
                ticker_data[ticker]['pt_list'].append(pt_current)
            else:
                ticker_data[ticker] = {'pt_list': [pt_current]}

    for ticker, info in ticker_data.items():
        try:
            with open(f"json/quote/{ticker}.json", 'r') as file:
                res = ujson.load(file)
            info['price'] = res.get('price', None)
            info['name'] = res.get('name', None)
            info['marketCap'] = res.get('marketCap', None)
        except:
            info['price'] = None
            info['name'] = None
            info['marketCap'] = None

    # Calculate median pt_current for each ticker
    for ticker, info in ticker_data.items():
        if info['pt_list']:
            info['median'] = round(statistics.median(info['pt_list']), 2)
    
    # Convert the dictionary back to a list format
    result = [{'ticker': ticker, 
               'upside': round((info['median']/info.get('price')-1)*100, 2) if info.get('price') else None, 
               'priceTarget': info['median'], 
               'price': info['price'], 
               'counter': len(info['pt_list']), 
               'name': info['name'], 
               'marketCap': info['marketCap']} 
              for ticker, info in ticker_data.items()]
    
    result = [item for item in result if item['upside'] is not None and item['upside'] >= 5 and item['upside'] <= 250]  # Filter outliers

    result_sorted = sorted(result, key=lambda x: x['counter'] if x['counter'] is not None else float('-inf'), reverse=True)

    for rank, item in enumerate(result_sorted):
        item['rank'] = rank + 1

    with open(f"json/analyst/top-stocks.json", 'w') as file:
        ujson.dump(result_sorted, file)


async def get_analyst_ratings(analyst_id, session):
    url = "https://api.benzinga.com/api/v2.1/calendar/ratings"
    res_list = []
    
    for page in range(5):
        try:
            querystring = {
                "token": api_key,
                "parameters[analyst_id]": analyst_id,
                "page": str(page),
                "pagesize": "1000"
            }
            async with session.get(url, headers=headers, params=querystring) as response:
                data = await response.json()
                ratings = data.get('ratings', [])
                if not ratings:
                    break  # Stop fetching if no more ratings
                res_list += ratings
        except Exception as e:
            #print(f"Error fetching page {page} for analyst {analyst_id}: {e}")
            break


    # Date filter: only include items with 'date' >= '2015-01-01'
    filtered_data = [
        {key: value for key, value in item.items() if key not in {'url_news', 'url', 'url_calendar', 'updated', 'time', 'currency'}}
        for item in res_list
        if datetime.strptime(item['date'], '%Y-%m-%d') >= datetime(2015, 1, 1)
    ]

    return filtered_data

async def get_all_analyst_stats():
    url = "https://api.benzinga.com/api/v2.1/calendar/ratings/analysts"
    res_list = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            session.get(url, headers=headers, params={"token": api_key, "page": str(page), 'pagesize': "1000"})
            for page in range(100)
        ]

        # Gather responses concurrently
        responses = await asyncio.gather(*tasks)
        
        # Process each response
        for response in responses:
            if response.status == 200:  # Check for successful response
                try:
                    data = ujson.loads(await response.text())['analyst_ratings_analyst']
                    res_list += data
                except Exception as e:
                    pass

    # Remove duplicates of analysts and filter based on ratings accuracy
    res_list = remove_duplicates(res_list, 'id')
    res_list = [item for item in res_list if item.get('ratings_accuracy', {}).get('total_ratings', 0) != 0]

    # Construct the final result list
    final_list = [{
        'analystName': item['name_full'],
        'companyName': item['firm_name'],
        'analystId': item['id'],
        'firmId': item['firm_id'],
        'avgReturn': item['ratings_accuracy'].get('overall_average_return', 0),
        'successRate': item['ratings_accuracy'].get('overall_success_rate', 0),
        'totalRatings': item['ratings_accuracy'].get('total_ratings', 0),
    } for item in res_list]

    return final_list

async def process_analyst(item, con, session, start_date, end_date):
    # Fetch analyst ratings
    data = await get_analyst_ratings(item['analystId'], session)
    item['ratingsList'] = data
    item['totalRatings'] = len(data)
    item['lastRating'] = data[0]['date'] if data else None
    item['numOfStocks'] = len({d['ticker'] for d in data})

    total_return = 0
    valid_ratings_count = 0
    success_count = 0  # To track successful ratings for success rate calculation

    for stock in data:
        try:
            ticker = stock['ticker']
            rating_date = stock['date']
            rating_current = stock['rating_current']

            # Skip neutral or undefined ratings
            if rating_current not in buy_ratings and rating_current not in sell_ratings:
                continue

            # Check if the ticker data is already cached
            if ticker not in ticker_price_cache:
                # If not cached, query the stock data and cache it
                query = query_template.format(ticker=ticker)
                df = pd.read_sql_query(query, con, params=(start_date, end_date))
                ticker_price_cache[ticker] = df
            else:
                # Use cached data
                df = ticker_price_cache[ticker]

            # Ensure we have data for the rating date
            rating_date_data = df[df['date'] == rating_date]
            if rating_date_data.empty:
                # Try finding the closest date within a few days if exact date is missing
                for days_offset in range(1, 5):
                    closest_date = (pd.to_datetime(rating_date) - pd.Timedelta(days=days_offset)).strftime('%Y-%m-%d')
                    rating_date_data = df[df['date'] == closest_date]
                    if not rating_date_data.empty:
                        break

            if rating_date_data.empty:
                continue  # Skip if no close price data found

            # Get close price on rating date
            close_price_on_rating = rating_date_data['close'].values[0]

            # Calculate the date 12 months later
            future_date = (pd.to_datetime(rating_date) + pd.DateOffset(months=12)).strftime('%Y-%m-%d')

            # Try to find the close price 12 months later
            future_date_data = df[df['date'] == future_date]
            if future_date_data.empty:
                # If 12 months price isn't available, use the latest available price
                future_date_data = df.iloc[-1]  # Use the last available price
                if future_date_data.empty:
                    continue  # If no future data, skip this rating

            close_price_in_future = future_date_data['close'] if isinstance(future_date_data, pd.Series) else future_date_data['close'].values[0]

            # Calculate return
            stock_return = (close_price_in_future - close_price_on_rating) / close_price_on_rating
            total_return += stock_return
            valid_ratings_count += 1

            # Determine if the rating was successful
            if rating_current in buy_ratings:
                if close_price_in_future > close_price_on_rating:
                    success_count += 1  # Success for buy ratings
            elif rating_current in sell_ratings:
                if close_price_in_future < close_price_on_rating:
                    success_count += 1  # Success for sell ratings
        except:
            pass

    # Calculate average return if there are valid ratings
    if valid_ratings_count > 0:
        item['avgReturn'] = round(total_return / valid_ratings_count * 100, 2)  # Percentage format
    else:
        item['avgReturn'] = 0

    # Calculate success rate
    if valid_ratings_count > 0:
        item['successRate'] = round(success_count / valid_ratings_count * 100, 2)  # Success rate in percentage
    else:
        item['successRate'] = 0

    # Populate other stats and score
    stats_dict = {
        'avgReturn': item.get('avgReturn', 0),
        'successRate': item.get('successRate', 0),
        'totalRatings': item['totalRatings'],
        'lastRating': item['lastRating'],
    }

    item['analystScore'] = calculate_rating(stats_dict)

async def get_single_analyst_data(analyst_list, con):
    start_date = '2015-01-01'
    end_date = datetime.today().strftime("%Y-%m-%d")

    async with aiohttp.ClientSession() as session:
        tasks = [process_analyst(item, con, session, start_date, end_date) for item in analyst_list]
        for task in tqdm(asyncio.as_completed(tasks), total=len(analyst_list)):
            await task

async def run():
    # Step1: Get all analyst id's and stats
    con = sqlite3.connect('stocks.db')
    analyst_list = await get_all_analyst_stats()
    print('Number of analysts:', len(analyst_list))
    
    #Test Mode
    #analyst_list = [ item for item in analyst_list if item['analystId'] =='597f5b95c1f5580001ef542a']

    # Step2: Get rating history for each individual analyst and score the analyst
    await get_single_analyst_data(analyst_list, con)
    try:
        print('Start extracting main sectors')
        for item in tqdm(analyst_list):
            ticker_list = [entry['ticker'] for entry in item['ratingsList']]
            sector_list = []
            for ticker in ticker_list:
                sector = extract_sector(ticker, con)
                sector_list.append(sector)

            sector_counts = Counter(sector_list)
            main_sectors = sector_counts.most_common(3)
            main_sectors = [item[0] for item in main_sectors if item[0] is not None]
            item['mainSectors'] = main_sectors

    except Exception as e:
        print(e)

    # Sort analysts by score
    analyst_list = sorted(analyst_list, key=lambda x: (float(x['analystScore']), float(x['avgReturn']), float(x['successRate'])), reverse=True)
    number_of_all_analysts = len(analyst_list)

    # Assign rank and other metrics to analysts
    for rank, item in enumerate(analyst_list):
        item['rank'] = rank + 1
        item['numOfAnalysts'] = number_of_all_analysts
        item['avgReturn'] = round(float(item['avgReturn']), 2)
        item['successRate'] = round(float(item['successRate']), 2)
        with open(f"json/analyst/analyst-db/{item['analystId']}.json", 'w') as file:
            ujson.dump(item, file)

    # Save top 100 analysts
    top_analysts_list = []
    for item in analyst_list[0:100]:
        top_analysts_list.append({
            'analystName': item['analystName'],
            'analystId': item['analystId'],
            'rank': item['rank'],
            'analystScore': item['analystScore'],
            'companyName': item['companyName'],
            'successRate': item['successRate'],
            'avgReturn': item['avgReturn'],
            'totalRatings': item['totalRatings'],
            'lastRating': item['lastRating']
        })

    with open(f"json/analyst/top-analysts.json", 'w') as file:
        ujson.dump(top_analysts_list, file)

    # Save all analyst data in raw form for the next step
    with open(f"json/analyst/all-analyst-data.json", 'w') as file:
        ujson.dump(analyst_list, file)

    # Save top stocks with strong buys from 5-star analysts
    get_top_stocks()

    # Close the connection
    con.close()


if __name__ == "__main__":
    asyncio.run(run())
    