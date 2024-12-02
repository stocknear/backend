import requests
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import time
import sqlite3
import orjson
import os
from dotenv import load_dotenv
from tqdm import tqdm 
import pandas as pd
from collections import Counter
import aiohttp
import asyncio
import statistics
import math

load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

headers = {"accept": "application/json"}

# Load stock screener data
with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

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



async def get_data(ticker_list, item):
    """Extract specified columns data for a given symbol."""
    columns = ['sector', 'industry', 'price']

    sector_list = []
    industry_list = []

    for ticker in ticker_list:
        ticker_data = stock_screener_data_dict.get(ticker, {})
        
        # Extract specified columns data for each ticker
        sector = ticker_data.get('sector',None)
        industry = ticker_data.get('industry',None)
        price = ticker_data.get('price',None)
        price = round(price, 2) if price is not None else None


        # Append data to relevant lists if values are present
        if sector:
            sector_list.append(sector)
        if industry:
            industry_list.append(industry)

        # Update ratingsList in item with price for the corresponding ticker
        try:
            if len(item.get('ratingsList')) > 0:
                for rating in item.get('ratingsList', []):
                    try:
                        if rating.get('ticker') == ticker:
                            upside = round((float(rating['adjusted_pt_current'])/price-1)*100,2)
                            rating['price'] = price
                            rating['upside'] = upside
                    except:
                        rating['price'] = None
                        rating['upside'] = None
        except:
            pass

    # Get the top 3 most common sectors and industries
    sector_counts = Counter(sector_list)
    industry_counts = Counter(industry_list)
    main_sectors = [item[0] for item in sector_counts.most_common(3)]
    main_industries = [item[0] for item in industry_counts.most_common(3)]

    # Add main sectors and industries to the item dictionary
    item['mainSectors'] = main_sectors
    item['mainIndustries'] = main_industries

    return item


def smooth_scale(value, max_value=100, curve_factor=2):
    """
    Create a smooth, non-linear scaling that prevents extreme values
    while allowing nuanced differentiation.
    """
    # Ensure inputs are valid
    if max_value <= 0:
        raise ValueError("max_value must be greater than 0")
    
    # Clamp the value to a non-negative range
    normalized = max(min(value / max_value, 1), 0)
    
    return math.pow(normalized, curve_factor)


def calculate_rating(data):
    overall_average_return = float(data['avgReturn'])
    overall_success_rate = float(data['successRate'])
    total_ratings = int(data['totalRatings'])
    last_rating = data['lastRating']
    average_return_percentile = float(data['avgReturnPercentile'])
    total_ratings_percentile = float(data['totalRatingsPercentile'])

    try:
        last_rating_date = datetime.strptime(last_rating, "%Y-%m-%d")
        difference = (datetime.now() - last_rating_date).days
    except:
        difference = 1000  # In case of None or invalid date

    if total_ratings == 0 or difference >= 600:
        return 0
    else:
        # Define weights for each factor
        weights = {
            'return': 0.35,
            'success_rate': 0.35,
            'total_ratings': 0.1,
            'recency': 0.1,
            'returnPercentile': 0.05,
            'ratingsPercentile': 0.05,
        }

        # Calculate weighted sum
        weighted_components = [
            weights['return'] * smooth_scale(overall_average_return, max_value=50, curve_factor=1.8),
            weights['success_rate'] * smooth_scale(overall_success_rate, max_value=100, curve_factor=1.5),
            weights['total_ratings'] * smooth_scale(min(total_ratings, 100), max_value=100, curve_factor=1.3),
            weights['recency'] * (1 / (1 + math.log1p(difference))),
            weights['returnPercentile'] * smooth_scale(average_return_percentile),
            weights['ratingsPercentile'] * smooth_scale(total_ratings_percentile),
        ]

        # Calculate base rating
        base_rating = sum(weighted_components)
        normalized_rating = min(max(base_rating / sum(weights.values()) * 5, 0), 5)
        # Encourage higher ratings for sufficient data and good performance
        if total_ratings > 50 and overall_success_rate > 60 and overall_average_return > 60:
            normalized_rating += 1.0

        elif total_ratings > 30 and overall_success_rate > 50 and overall_average_return > 50:
            normalized_rating += 0.5

        elif total_ratings > 20 and overall_success_rate >= 50 and overall_average_return >= 15:
            normalized_rating += 0.3

        # Apply additional conditions based on return and success rate thresholds
        if overall_average_return <= 5:
            normalized_rating = max(normalized_rating - 1.5, 0)
        elif overall_average_return <= 10:
            normalized_rating = max(normalized_rating - 1.0, 0)

        if overall_success_rate < 50:
            normalized_rating = min(normalized_rating, 3.5)

        # Cap the rating for older ratings
        if difference > 30:
            normalized_rating = min(normalized_rating, 4.8)

        # Ensure final rating remains in valid bounds
        #print(round(min(max(normalized_rating, 0), 5), 2))
        return round(min(max(normalized_rating, 0), 5), 2)

def get_top_stocks():
    with open(f"json/analyst/all-analyst-data.json", 'r') as file:
        analyst_stats_list = orjson.loads(file.read())

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
                res = orjson.loads(file.read())
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
    result = [{'symbol': ticker, 
               'upside': round((info['median']/info.get('price')-1)*100, 2) if info.get('price') else None, 
               'priceTarget': info['median'], 
               'price': info['price'], 
               'counter': len(info['pt_list']), 
               'name': info['name']} 
              for ticker, info in ticker_data.items()]
    
    result = [item for item in result if item['upside'] is not None and item['upside'] >= 20 and item['upside'] <= 250]  # Filter outliers

    result_sorted = sorted(result, key=lambda x: x['counter'] if x['counter'] is not None else float('-inf'), reverse=True)
    
    #top 100 stocks
    result_sorted = result_sorted[:100]

    for rank, item in enumerate(result_sorted):
        item['rank'] = rank + 1

    with open(f"json/analyst/top-stocks.json", 'w') as file:
        file.write(orjson.dumps(result_sorted).decode('utf-8'))


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

    # If prior rating and current rating is "Buy" we interpret it as "Strong Buy"
    for item in filtered_data:
        try:
            if item.get("rating_prior",None) == "Buy" and item.get("rating_current",None) == "Buy":
                if float(item.get("adjusted_pt_prior", 0)) < float(item.get('adjusted_pt_current', 0)):
                    item["rating_current"] = "Strong Buy"
        except:
            pass

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
                    data = orjson.loads(await response.text())['analyst_ratings_analyst']
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
        'totalRatingsPercentile': item['ratings_accuracy'].get('total_ratings_percentile', 0),
        'avgReturnPercentile': item['ratings_accuracy'].get('avg_return_percentile', 0),
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
        item['successRate'] = round((success_count / valid_ratings_count) * 100, 2)  # Success rate in percentage
    else:
        item['successRate'] = 0

    # Populate other stats and score
    stats_dict = {
        'avgReturn': item.get('avgReturn', 0),
        'successRate': item.get('successRate', 0),
        'totalRatings': item['totalRatings'],
        'lastRating': item['lastRating'],
        'totalRatingsPercentile': item['totalRatingsPercentile'],
        'avgReturnPercentile': item['avgReturnPercentile']
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
    
    #Test Modes
    #analyst_list = [ item for item in analyst_list if item['analystId'] =='5a02da51efacff00010633d2']

    # Step2: Get rating history for each individual analyst and score the analyst
    await get_single_analyst_data(analyst_list, con)
    try:
        print('Start extracting data')
        for item in tqdm(analyst_list):
            ticker_list = [entry['ticker'] for entry in item['ratingsList']]
            if len(ticker_list) > 0:
                await get_data(ticker_list, item)

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
            file.write(orjson.dumps(item).decode('utf-8'))

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
        file.write(orjson.dumps(top_analysts_list).decode('utf-8'))

    # Save all analyst data in raw form for the next step
    with open(f"json/analyst/all-analyst-data.json", 'w') as file:
        file.write(orjson.dumps(analyst_list).decode('utf-8'))

    # Save top stocks with strong buys from 5-star analysts
    get_top_stocks()

    # Close the connection
    con.close()


if __name__ == "__main__":
    asyncio.run(run())
    