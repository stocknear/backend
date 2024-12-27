import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('UNUSUAL_WHALES_API_KEY')

url = 'https://api.unusualwhales.com/api/stock/XLB/net-prem-ticks'
headers = {
    'Accept': 'application/json',
    'Authorization': api_key
}

response = requests.get(url, headers=headers)
data = response.json()['data']

fields_to_sum = [
    "net_call_premium",
    "net_call_volume",
    "net_put_premium",
    "net_put_volume"
]

result = []
for idx, e in enumerate(data):
    e['net_call_premium'] = float(e['net_call_premium'])
    e['net_put_premium'] = float(e['net_put_premium'])

    #e['net_call_volume'] = float(e['net_call_volume'])
    #e['net_put_volume'] = float(e['net_put_volume'])
    
    
    if idx != 0:
        for field in fields_to_sum:
            e[field] += result[idx-1].get(field, 0)
    
    result.append(e)

#print(result)
print(result[-1]['net_put_volume']*result[-1]['net_put_premium']*10**(-6))