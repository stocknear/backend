import requests
import pandas as pd
import time
import ujson
from datetime import datetime
from tqdm import tqdm 
from collections import defaultdict



start_date = '2015-01-01'
end_date = datetime.today().strftime("%Y-%m-%d")

# API endpoint for spending by award
url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

# Headers
headers = {
    "Content-Type": "application/json",
}


def save_json(symbol, data):
    with open(f"json/government-contract/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

# Define a function to remove duplicates based on a key
def remove_duplicates(data, key):
    seen = set()
    new_data = []
    for item in data:
        if item[key] not in seen:
            seen.add(item[key])
            new_data.append(item)
    return new_data


def sum_contract(symbol, data):
    aggregated_data = {}
    for entry in data:
        year = entry['date'][:4]
        expenses = entry.get('amount')  # Retrieve expenses or default to None
        if expenses is not None:  # Check if expenses is not None
            if year not in aggregated_data:
                aggregated_data[year] = {
                    'year': year,
                    'amount': 0,
                    'numOfContracts': 0,
                }
            aggregated_data[year]['amount'] += int(expenses)
            aggregated_data[year]['numOfContracts'] += 1

    data = list(aggregated_data.values())
    save_json(symbol, data)
    

def get_data(symbol, name):
    res = []
    for page in tqdm(range(1,2000)):
        try:
            data = {
                "filters": {
                    "recipient_search_text": [name],
                    "time_period": [{"start_date": start_date, "end_date": end_date}],
                    "award_type_codes": ["A", "B", "C", "D"],  # Contract award types
                },
                "fields": ["Award ID", "Recipient Name", "Award Amount", "Last Modified Date"],
                "page": page,
                "limit": 100  # Adjust as needed
            }
            response = requests.post(url, json=data, headers=headers)
            response_data = (response.json())['results']
            res += [{'id': item['Award ID'], 'amount': item['Award Amount'], 'date': item['Last Modified Date']} for item in response_data]
            time.sleep(1) #avoid api limit
        except Exception as e:
            break

    sorted_res = sorted(res, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
    sorted_res = remove_duplicates(sorted_res, 'id')
    
    if len(sorted_res) > 0:
        sum_contract(symbol, sorted_res)

try:
    company_data = [{'symbol': 'LMT', 'name': 'Lockheed Martin'},{'symbol': 'J', 'name': 'Jacobs Engineering'},{'symbol': 'CRWD', 'name': 'CrowdStrike'},{'symbol': 'FLR', 'name': 'Fluor'},{'symbol': 'GD', 'name': 'General Dynamics'},{'symbol': 'NOC', 'name': 'Northrop Grumman'},{'symbol': 'RTX', 'name': 'Raytheon Technologies'},{'symbol': 'LHX', 'name': 'L3Harris Technologies'},{'symbol': 'CAT', 'name': 'Caterpillar'},{'symbol': 'JNJ', 'name': 'Johnson & Johnson'},{'symbol': 'CVX', 'name': 'Chevron'},{'symbol': 'XOM', 'name': 'Exxon Mobil'},{'symbol': 'UNH', 'name': 'UnitedHealth'},{'symbol': 'PFE', 'name': 'Pfizer'},{'symbol': 'BAH', 'name': 'Booz Allen Hamilton'},{'symbol': 'NEE', 'name': 'NextEra'},{'symbol': 'LDOS', 'name': 'Leidos'},{'symbol': 'PLTR', 'name': 'Palantir'},{'symbol': 'HII', 'name': 'Huntington Ingalls'},{'symbol': 'CACI', 'name': 'CACI International'},{'symbol': 'SAIC', 'name': 'Science Applications'},{'symbol': 'BA', 'name': 'Boeing'}]
    for item in company_data:
        symbol = item['symbol']
        name = item['name']
        get_data(symbol, name)
        time.sleep(100) #avoid api limit

except Exception as e:
    print(e)


