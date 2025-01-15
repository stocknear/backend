import ujson
import asyncio
import aiohttp
import os
import sqlite3
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
import requests

# Load environment variables
load_dotenv()

today = datetime.today().date()

api_key = os.getenv('UNUSUAL_WHALES_API_KEY')

url = "https://api.unusualwhales.com/api/market/fda-calendar"

headers = {
    "Accept": "application/json, text/plain",
    "Authorization": api_key
}



async def save_json(data):
    with open(f"json/fda-calendar/data.json", 'w') as file:
        ujson.dump(data, file)


async def get_data():

    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    try:
        response = requests.get(url, headers=headers)
        data = response.json()['data']
        data = [
            entry for entry in data
            if datetime.strptime(entry['start_date'], '%Y-%m-%d').date() >= today
        ]
        
        res_list = []
        for item in data:
            try:
                symbol = item['ticker']
                if symbol in stock_symbols:
                    res_list.append({**item})
            except:
                pass
        
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []


async def run():
    data = await get_data()
    if len(data) > 0:
        await save_json(data)

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(f"An error occurred: {e}")