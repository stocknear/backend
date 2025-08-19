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
        # Adjusted weights for more balanced scoring
        weights = {
            'return': 0.30,         # Reduced from 0.35
            'success_rate': 0.30,   # Reduced from 0.35
            'total_ratings': 0.15,  # Increased from 0.1
            'recency': 0.10,        # Same
            'returnPercentile': 0.075,  # Increased from 0.05
            'ratingsPercentile': 0.075, # Increased from 0.05
        }

        # More generous scaling for returns (adjusted max_value and curve_factor)
        return_component = weights['return'] * smooth_scale(
            overall_average_return + 10,  # Shift up to make 0% return more neutral
            max_value=40,  # Reduced from 50 for more generous scaling
            curve_factor=1.5  # Reduced from 1.8 for smoother curve
        )
        
        # Calculate weighted sum with adjusted components
        weighted_components = [
            return_component,
            weights['success_rate'] * smooth_scale(overall_success_rate, max_value=100, curve_factor=1.3),  # Reduced from 1.5
            weights['total_ratings'] * smooth_scale(min(total_ratings, 80), max_value=80, curve_factor=1.2),  # Adjusted threshold
            weights['recency'] * (1 / (1 + math.log1p(difference/7))),  # Scaled by week instead of day
            weights['returnPercentile'] * smooth_scale(average_return_percentile, max_value=100, curve_factor=1.2),
            weights['ratingsPercentile'] * smooth_scale(total_ratings_percentile, max_value=100, curve_factor=1.2),
        ]

        # Calculate base rating with a higher baseline
        base_rating = sum(weighted_components)
        # Start with a base of 2.5 to ensure decent analysts get reasonable ratings
        normalized_rating = 2.5 + (base_rating / sum(weights.values()) * 3.5)  # Base 2.5 + up to 3.5 from performance
        
        # More generous bonus conditions with lower thresholds
        if total_ratings >= 35 and overall_success_rate >= 58 and overall_average_return >= 18:
            normalized_rating += 0.8  # Top tier bonus
        elif total_ratings >= 20 and overall_success_rate >= 54 and overall_average_return >= 12:
            normalized_rating += 0.5  # Strong performer bonus
        elif total_ratings >= 12 and overall_success_rate >= 51 and overall_average_return >= 8:
            normalized_rating += 0.35  # Good performer bonus
        elif total_ratings >= 8 and overall_success_rate >= 49 and overall_average_return >= 4:
            normalized_rating += 0.25  # Decent performer bonus
        elif total_ratings >= 5 and overall_success_rate >= 47 and overall_average_return >= 0:
            normalized_rating += 0.15  # Basic qualifier bonus

        # Less aggressive penalties  
        if overall_average_return <= -20:
            normalized_rating = max(normalized_rating - 1.2, 0)
        elif overall_average_return <= -15:
            normalized_rating = max(normalized_rating - 0.8, 0)  
        elif overall_average_return <= -10:
            normalized_rating = max(normalized_rating - 0.4, 0)  
        elif overall_average_return <= -5:
            normalized_rating = max(normalized_rating - 0.2, 0)  
        elif overall_average_return < 0:
            normalized_rating = max(normalized_rating - 0.05, 0)  # Very minor penalty for slightly negative returns

        # More lenient success rate cap
        if overall_success_rate < 45:
            normalized_rating = min(normalized_rating, 3.7)  # Increased from 3.5
        elif overall_success_rate < 50:
            normalized_rating = min(normalized_rating, 4.2)  # Allow 4+ ratings with decent success

        # Adjusted recency cap - more forgiving
        if difference > 60:
            normalized_rating = min(normalized_rating, 4.7)  # Only cap if older than 2 months
        elif difference > 30:
            normalized_rating = min(normalized_rating, 4.9)  # Very minor cap for 1+ month old

        # Percentile boost for top performers (new addition)
        if average_return_percentile >= 70 and total_ratings_percentile >= 50:
            normalized_rating += 0.2  # Boost for analysts in top percentiles
        elif average_return_percentile >= 60 and total_ratings_percentile >= 40:
            normalized_rating += 0.1

        # Ensure final rating remains in valid bounds
        return round(min(max(normalized_rating, 0), 5), 2)

def get_top_stocks():
    with open(f"json/analyst/all-analyst-data.json", 'r') as file:
        analyst_stats_list = orjson.loads(file.read())

    # Filter analysts with a score >= 4
    filtered_data = [item for item in analyst_stats_list if item['analystScore'] >= 4]
    end_date = datetime.now().date()
    # Define the date range for the past 12 months
    start_date = end_date - timedelta(days=365)

    # Track unique analyst-stock pairs and get the latest Strong Buy rating within the past 12 months
    # from each unique analyst with a 4-star or higher rating
    res_list = []
    
    for analyst in filtered_data:
        analyst_id = analyst['analystId']
        ticker_ratings = {}

        for rating in analyst['ratingsList']:
            rating_date = datetime.strptime(rating['date'], '%Y-%m-%d').date()
            ticker = rating['ticker']
            
            if rating['rating_current'] == 'Strong Buy' and start_date <= rating_date:
                # Keep the latest rating for each stock by this analyst
                if ticker not in ticker_ratings or rating_date > ticker_ratings[ticker]['date']:
                    ticker_ratings[ticker] = {
                        'ticker': ticker,
                        'adjusted_pt_current': rating['adjusted_pt_current'],
                        'date': rating_date,
                        'analystId': analyst_id
                    }

        # Add the latest ratings to the result list
        res_list.extend(ticker_ratings.values())

    # Create a dictionary to store ticker occurrences and corresponding pt_current values
    ticker_data = {}
    for item in res_list:
        ticker = item['ticker']
        pt_current_str = item['adjusted_pt_current']
        analyst_id = item['analystId']

        if pt_current_str:  # Skip empty strings
            pt_current = float(pt_current_str)

            if ticker not in ticker_data:
                ticker_data[ticker] = {
                    'pt_list': [],
                    'analyst_ids': set()
                }

            # Only count unique analysts per ticker (each analyst counted once per stock)
            # This ensures we're considering only the latest rating from each unique analyst
            if analyst_id not in ticker_data[ticker]['analyst_ids']:
                ticker_data[ticker]['pt_list'].append(pt_current)
                ticker_data[ticker]['analyst_ids'].add(analyst_id)

    # Fetch additional data (price, name, marketCap) for each ticker
    for ticker, info in ticker_data.items():
        try:
            with open(f"json/quote/{ticker}.json", 'r') as file:
                res = orjson.loads(file.read())
            info['price'] = res.get('price', None)
            info['name'] = res.get('name', None)
            info['marketCap'] = res.get('marketCap', None)
        except:
            info['name'] = None
            info['marketCap'] = None

    # Calculate median pt_current for each ticker
    for ticker, info in ticker_data.items():
        if info['pt_list']:
            info['median'] = round(statistics.median(info['pt_list']), 2)

    # Convert the dictionary back to a list format
    result = [
        {
            'symbol': ticker,
            'topAnalystUpside': round((info['median'] / info.get('price') - 1) * 100, 2) if info.get('price') else None,
            'topAnalystPriceTarget': info['median'],
            'topAnalystCounter': len(info['analyst_ids']),
            'topAnalystRating': "Strong Buy",
            'marketCap': info['marketCap'],
            'name': info['name']
        }
        for ticker, info in ticker_data.items()
    ]

    # Filter outliers with upside between 20% and 250%
    result = [item for item in result if item['topAnalystUpside'] is not None and 10 <= item['topAnalystUpside'] <= 250]

    # Sort results by the number of unique analysts (analystCounter) in descending order
    result_sorted = sorted(result, key=lambda x: x['topAnalystCounter'] if x['topAnalystCounter'] is not None else float('-inf'), reverse=True)

    # Top 50 stocks
    result_sorted = result_sorted[:50]

    # Add rank to each item
    for rank, item in enumerate(result_sorted):
        item['rank'] = rank + 1

    # Save results to a JSON file
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
    
    if len(analyst_list) < 4000:
        return

    #Test Modes
    #analyst_list = [ item for item in analyst_list if item['analystId'] =='5e720d1d5f2b9b000114e970']

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
    