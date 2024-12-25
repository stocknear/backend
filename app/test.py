import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('UNUSUAL_WHALES_API_KEY')

querystring = {"limit":"200"}

url = "https://api.unusualwhales.com/api/darkpool/recent"

headers = {
    "Accept": "application/json, text/plain",
    "Authorization": api_key
}

response = requests.get(url, headers=headers, params=querystring)

print(len(response.json()['data']))
print(response.json()['data'][0])