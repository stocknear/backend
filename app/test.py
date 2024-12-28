import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('UNUSUAL_WHALES_API_KEY')

url = "https://api.unusualwhales.com/api/screener/stocks"

querystring = {
    'order': 'net_premium',
    'order_direction': 'desc',
    'sectors[]': 'Technology'
}

headers = {
    "Accept": "application/json, text/plain",
    "Authorization": api_key
}

response = requests.get(url, headers=headers, params=querystring)

data = response.json()['data']


print(data[0])
print(len(data))

