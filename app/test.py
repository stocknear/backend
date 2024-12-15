import requests

url = "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json?filter=top"
response = requests.get(url)
print(response)