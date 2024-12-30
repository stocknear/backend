import requests
from dotenv import load_dotenv
import os
import sqlite3

load_dotenv()
api_key = os.getenv('UNUSUAL_WHALES_API_KEY')

con = sqlite3.connect('stocks.db')
etf_con = sqlite3.connect('etf.db')

cursor = con.cursor()
cursor.execute("PRAGMA journal_mode = wal")
cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
stocks_symbols = [row[0] for row in cursor.fetchall()]

etf_cursor = etf_con.cursor()
etf_cursor.execute("PRAGMA journal_mode = wal")
etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
etf_symbols = [row[0] for row in etf_cursor.fetchall()]

con.close()
etf_con.close()

total_symbols = stocks_symbols[:1000] #+ etf_symbols
total_symbols = ",".join(total_symbols)
print(total_symbols)
url = "https://api.unusualwhales.com/api/screener/stocks"

querystring = {"ticker": total_symbols}

headers = {
    "Accept": "application/json, text/plain",
    "Authorization": api_key
}

response = requests.get(url, headers=headers, params=querystring)

data = response.json()['data']


print(len(data))
print(data[-1]['ticker'])

