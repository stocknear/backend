from benzinga import financial_data
import requests
from datetime import datetime, timedelta, date
from collections import defaultdict
import numpy as np
from scipy.stats import norm
import time
import sqlite3
import ujson
import math
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

fin = financial_data.Benzinga(api_key)



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
	#Get Latest Summary of ratings from the last 12 months
	# -Number of Analyst, -Price Target, -Consensus Rating
	end_date = date.today()
	start_date = end_date - timedelta(days=365) #end_date is today
	filtered_data = [item for item in res_list if start_date <= datetime.strptime(item['date'], '%Y-%m-%d').date() <= end_date]

	#Compute Average Price Target
	latest_pt_current = defaultdict(int)
	# Iterate through the data to update the latest pt_current for each analyst
	for item in filtered_data:
		if 'adjusted_pt_current' in item and item['adjusted_pt_current']:
			analyst_name = item['analyst_name']
			latest_pt_current[analyst_name] = max(latest_pt_current[analyst_name], float(item['pt_current']))

	# Compute the average pt_current based on the latest values
	pt_current_values = list(latest_pt_current.values())
	average_pt_current = sum(pt_current_values) / len(pt_current_values) if pt_current_values else 0

	#print("Average pt_current:", round(average_pt_current, 2))




	# Compute Consensus Rating
	consensus_ratings = defaultdict(str)
	# Define the rating hierarchy
	rating_hierarchy = {'Strong Sell': 0, 'Sell': 1, 'Hold': 2, 'Buy': 3, 'Strong Buy': 4}

	# Iterate through the data to update the consensus rating for each analyst
	for item in filtered_data:
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
	#print("Consensus Rating:", consensus_rating)

	#Sum up all Buy,Sell,Hold for the progress bar in sveltekit
	# Convert defaultdict to regular dictionary
	data_dict = dict(consensus_rating_counts)

	# Sum up 'Strong Buy' and 'Buy'
	buy_total = data_dict.get('Strong Buy', 0) + data_dict.get('Buy', 0)

	# Sum up 'Strong Sell' and 'Sell'
	sell_total = data_dict.get('Strong Sell', 0) + data_dict.get('Sell', 0)
	hold_total = data_dict.get('Hold', 0)


	unique_analyst_names = set()
	numOfAnalyst = 0

	for item in filtered_data:
	    if item['analyst_name'] not in unique_analyst_names:
	        unique_analyst_names.add(item['analyst_name'])
	        numOfAnalyst += 1
	#print("Number of unique analyst names:", numOfAnalyst)

	stats = {'numOfAnalyst': numOfAnalyst, 'consensusRating': consensus_rating, 'priceTarget': round(average_pt_current, 2)}
	categorical_ratings = {'Buy': buy_total, 'Sell': sell_total, 'Hold': hold_total}
	
	res = {**stats, **categorical_ratings}
	return res

def run(chunk,analyst_list):
	end_date = date.today()
	start_date = datetime(2015,1,1)
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
	#print(res_list[-15])
	for ticker in chunk:
		try:
			ticker_filtered_data = [item for item in res_list if item['ticker'] == ticker]
			if len(ticker_filtered_data) != 0:
				for item in ticker_filtered_data:
					if item['rating_current'] == 'Strong Sell' or item['rating_current'] == 'Strong Buy':
						pass
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
						item['action_comapny'] = 'Initiates'
					elif item['rating_current'] == 'Market Outperform' and (item['action_company'] == 'Maintains' or item['action_company'] == 'Reiterates'):
						item['rating_current'] = 'Buy'
					elif item['rating_current'] == 'Outperform' and (item['action_company'] == 'Maintains' or item['action_pt'] == 'Announces' or item['action_company'] == 'Upgrades'):
						item['rating_current'] = 'Buy'
					elif item['rating_current'] == 'Buy' and (item['action_company'] == 'Raises' or item['action_pt'] == 'Raises'):
						item['rating_current'] = 'Strong Buy'
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
					elif item['rating_current'] == 'Peer Perform' and item['rating_prior'] == 'Peer Perfrom':
						item['rating_current'] = 'Hold'
					elif item['rating_current'] == 'Peer Perform' and item['action_pt'] == 'Announces':
						item['rating_current'] = 'Hold'
						item['action_comapny'] = 'Initiates'

				summary = get_summary(ticker_filtered_data)

				#get ratings of each analyst
				with open(f"json/analyst/summary/{ticker}.json", 'w') as file:
					ujson.dump(summary, file)

				for item1 in ticker_filtered_data:
					#item1['analystId'] = ''
					#item1['analystScore'] = 0
					#item1['adjusted_pt_current'] = 0
					#item1['adjusted_pt_prior'] = 0
					for item2 in analyst_stats_list:
						if item1['analyst'] == item2['companyName'] and item1['analyst_name'] == item2['analystName']:
							item1['analystId'] = item2['analystId']
							item1['analystScore'] = item2['analystScore']
							break
						elif item1['analyst_name'] == item2['analystName']:
							item1['analystId'] = item2['analystId']
							item1['analystScore'] = item2['analystScore']
							break
					#Bug: Benzinga does not give me reliable all analyst names and hence score. 
					# Compute in those cases the analyst score separately for each analyst
					
					'''
					if 'analystScore' not in item1: #or item1['analystScore'] == 0:
						one_sample_list = get_one_sample_analyst_data(item1['analyst_name'], item1['analyst'])
						item1['analystId'] = one_sample_list[0]['id']
						item1['analystScore'] = one_sample_list[0]['analystScore']
					'''

				desired_keys = ['date', 'action_company', 'rating_current', 'adjusted_pt_current', 'adjusted_pt_prior', 'analystId', 'analystScore', 'analyst', 'analyst_name']

				ticker_filtered_data = [
				    {key: item[key] if key in item else None for key in desired_keys}
				    for item in ticker_filtered_data
				]


				#print(ticker_filtered_data[0])
				#time.sleep(10000)
				with open(f"json/analyst/history/{ticker}.json", 'w') as file:
					ujson.dump(ticker_filtered_data, file)

		except Exception as e:
			print(e)




try:
    stock_con = sqlite3.connect('stocks.db')
    stock_cursor = stock_con.cursor()
    stock_cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in stock_cursor.fetchall()]

    stock_con.close()
    	
    #Save all analyst data in raw form for the next step
    with open(f"json/analyst/all-analyst-data.json", 'r') as file:
    	analyst_stats_list = ujson.load(file)

    chunk_size = len(stock_symbols) // 40  # Divide the list into N chunks
    chunks = [stock_symbols[i:i + chunk_size] for i in range(0, len(stock_symbols), chunk_size)]
    #chunks = [['AMD','NVDA','MSFT']]
    for chunk in chunks:
        run(chunk, analyst_stats_list)

except Exception as e:
    print(e)
