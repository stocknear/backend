from datetime import datetime, timedelta
import orjson
import time
import sqlite3
import asyncio
import aiohttp
import random
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

async def save_json(data):
    with open(f"json/sentiment-tracker/data.json", 'wb') as file:
        file.write(orjson.dumps(data))


async def get_data(session, total_symbols):
    sources = ["twitter", "stocktwits"]  # Sources to loop through
    result_data = {}  # Dictionary to store results from both sources

    for source in sources:
        # Construct the API URL with the source parameter
        url = f"https://financialmodelingprep.com/api/v4/social-sentiments/trending?type=bullish&source={source}&apikey={api_key}"

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if len(data) > 0:
                        res_list = []
                        for item in data:
                            symbol = item['symbol']
                            item['sentiment'] = round(item['sentiment']*100)
                            item['lastSentiment'] = round(item['lastSentiment']*100)
                            if symbol in total_symbols:
                                try:
                                    with open(f"json/quote/{symbol}.json", 'r') as file:
                                        res = orjson.loads(file.read())
                                        item['price'] = round(res['price'], 2)
                                        item['changesPercentage'] = round(res['changesPercentage'], 2)
                                        item['marketCap'] = round(res['marketCap'], 2)
                                    res_list.append({**item})
                                except:
                                    pass
                        result_data[source] = res_list  # Store the result list for the current source
        except Exception as e:
            print(f"Error fetching data from {source}: {e}")
            pass
    
    # Save the combined result as a single JSON file
    if result_data:
        await save_json(result_data)



async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    total_symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    
    async with aiohttp.ClientSession() as session:
        await get_data(session, total_symbols)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())