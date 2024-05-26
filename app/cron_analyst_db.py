from benzinga import financial_data
import requests
from datetime import datetime, timedelta, date
from collections import defaultdict
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

load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

fin = financial_data.Benzinga(api_key)

headers = {"accept": "application/json"}


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
        difference = 1000  # In case of None

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

        if normalized_rating >= 4:
            if total_ratings < 10:
                normalized_rating -= 2.4
            elif total_ratings < 15:
                normalized_rating -= 2.5
            elif total_ratings < 20:
                normalized_rating -= 0.75
            elif total_ratings < 30:
                normalized_rating -= 1
            elif overall_average_return <=10:
            	normalized_rating -=1.1
        '''
        if overall_average_return <= 0 and overall_average_return >= -5:
            normalized_rating = min(normalized_rating - 2, 0)
        elif overall_average_return < -5 and overall_average_return >= -10:
            normalized_rating = min(normalized_rating - 3, 0)
        else:
        	normalized_rating = min(normalized_rating - 4, 0)
       	'''
       	if overall_average_return <= 0:
            normalized_rating = min(normalized_rating - 2, 0)

        normalized_rating = max(normalized_rating, 0)

        return round(normalized_rating, 2)

def get_analyst_ratings(analyst_id):
	
	url = "https://api.benzinga.com/api/v2.1/calendar/ratings"
	res_list = []
	
	for page in range(0,5):
		try:
			querystring = {"token":api_key,"parameters[analyst_id]": analyst_id, "page": str(page), "pagesize":"1000"}
			response = requests.request("GET", url, headers=headers, params=querystring)
			data = ujson.loads(response.text)['ratings']
			res_list +=data
			time.sleep(2)
		except:
			break

	return res_list

def get_all_analyst_stats():
	url = "https://api.benzinga.com/api/v2.1/calendar/ratings/analysts"
	res_list = []
	for _ in range(0,20): #Run the api N times because not all analyst are counted Bug from benzinga
		for page in range(0,100):
			try:
				querystring = {"token":api_key,"page": f"{page}", 'pagesize': "1000"}
				response = requests.request("GET", url, headers=headers, params=querystring)

				data = ujson.loads(response.text)['analyst_ratings_analyst']
				res_list+=data
			except:
				break
		time.sleep(5)
	
	res_list = remove_duplicates(res_list, 'id') # remove duplicates of analyst
	res_list = [item for item in res_list if item.get('ratings_accuracy', {}).get('total_ratings', 0) != 0]

	final_list = []
	for item in res_list:
		analyst_dict = {
			'analystName': item['name_full'],
		    'companyName': item['firm_name'],
		    'analystId': item['id'],
		    'firmId': item['firm_id']
		}

		stats_dict = {
			'avgReturn': item['ratings_accuracy'].get('overall_average_return', 0),
		    'successRate': item['ratings_accuracy'].get('overall_success_rate', 0),
		    'totalRatings': item['ratings_accuracy'].get('total_ratings', 0),
		}

		final_list.append({**analyst_dict,**stats_dict})


	return final_list

def get_top_stocks():
	with open(f"json/analyst/all-analyst-data.json", 'r') as file:
		analyst_stats_list = ujson.load(file)

	filtered_data = [item for item in analyst_stats_list if item['analystScore'] >= 5]

	res_list = []
	for item in filtered_data:
	    ticker_list = item['ratingsList']
	    ticker_list = [{'ticker': i['ticker'], 'pt_current': i['pt_current']} for i in ticker_list if i['rating_current'] == 'Strong Buy']
	    if len(ticker_list) > 0:
	        #res_list += list(set(ticker_list))
	        res_list += ticker_list
	        
	# Create a dictionary to store ticker occurrences and corresponding pt_current values
	ticker_data = {}
	for item in res_list:
	    ticker = item['ticker']
	    pt_current_str = item['pt_current']
	    if pt_current_str:  # Skip empty strings
	        pt_current = float(pt_current_str)
	        if ticker in ticker_data:
	            ticker_data[ticker]['sum'] += pt_current
	            ticker_data[ticker]['counter'] += 1
	        else:
	            ticker_data[ticker] = {'sum': pt_current, 'counter': 1}

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

	# Calculate average pt_current for each ticker
	for ticker, info in ticker_data.items():
	    info['average'] = round(info['sum'] / info['counter'],2)

	# Convert the dictionary back to a list format
	result = [{'ticker': ticker, 'upside': round((info['average']/info.get('price')-1)*100, 2) if info.get('price') else None, 'priceTarget': info['average'], 'price': info['price'], 'counter': info['counter'], 'name': info['name'], 'marketCap': info['marketCap']} for ticker, info in ticker_data.items()]
	result = [item for item in result if item['upside'] is not None and item['upside'] >= 5 and item['upside'] <= 250] #filter outliners

	result_sorted = sorted(result, key=lambda x: x['counter'] if x['counter'] is not None else float('-inf'), reverse=True)

	for rank, item in enumerate(result_sorted):
		item['rank'] = rank+1

	with open(f"json/analyst/top-stocks.json", 'w') as file:
	    ujson.dump(result_sorted, file)


if __name__ == "__main__":
	#Step1 get all analyst id's and stats
	analyst_list = get_all_analyst_stats()
	print('Number of analyst:', len(analyst_list))
	#Step2 get rating history for each individual analyst and score the analyst
	for item in tqdm(analyst_list):
		data = get_analyst_ratings(item['analystId'])
		item['ratingsList'] = data
		item['totalRatings'] = len(data) #true total ratings, which is important for the score
		item['lastRating'] = data[0]['date'] if len(data) > 0 else None
		item['numOfStocks'] = len({item['ticker'] for item in data})
		stats_dict = {
			'avgReturn': item.get('avgReturn', 0),
		    'successRate': item.get('successRate', 0),
		    'totalRatings': item.get('totalRatings', 0),
		    'lastRating': item.get('lastRating', None),
		}
		item['analystScore'] = calculate_rating(stats_dict)

	try:
		con = sqlite3.connect('stocks.db')
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
		con.close()
	except Exception as e:
		print(e)

	analyst_list = sorted(analyst_list, key=lambda x: float(x['analystScore']), reverse=True)
	number_of_all_analysts = len(analyst_list)

	for rank, item in enumerate(analyst_list):
		item['rank'] = rank+1
		item['numOfAnalysts'] = number_of_all_analysts
		item['avgReturn'] = round(float(item['avgReturn']),2)
		item['successRate'] = round(float(item['successRate']),2)
		with open(f"json/analyst/analyst-db/{item['analystId']}.json", 'w') as file:
			ujson.dump(item, file)


	#Save top 100 analysts
	top_analysts_list = []
	#Drop the element ratingsList for the top 100 analysts list
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
	    	'lastRating': item['lastRating'],
	    	'mainSectors': item['mainSectors']
	    })

	with open(f"json/analyst/top-analysts.json", 'w') as file:
		ujson.dump(top_analysts_list, file)

	#Save all analyst data in raw form for the next step
	with open(f"json/analyst/all-analyst-data.json", 'w') as file:
		ujson.dump(analyst_list, file)

	#Save top stocks with strong buys from 5 star analysts
	get_top_stocks()