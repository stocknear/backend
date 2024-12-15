import requests
from bs4 import BeautifulSoup

url = "https://twitter.com/search?q=%24AAPL&src=typed_query"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

print(soup)