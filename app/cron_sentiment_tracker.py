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
    with open(f"json/tracker/sentiment/data.json", 'wb') as file:
        file.write(orjson.dumps(data))


async def get_data(session, total_symbols):
    sources = ["twitter", "stocktwits"]  # Sources to loop through
    result_data = {}  # Dictionary to store the final combined results

    for source in sources:
        # Construct the API URL with the source parameter
        url = f"https://financialmodelingprep.com/api/v4/social-sentiments/trending?type=bullish&source={source}&apikey={api_key}"

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if len(data) > 0:
                        for item in data:
                            symbol = item['symbol']
                            item['sentiment'] = round(item['sentiment']*100)

                            if symbol in total_symbols:
                                try:
                                    with open(f"json/quote/{symbol}.json", 'r') as file:
                                        res = orjson.loads(file.read())
                                        item['price'] = round(res['price'], 2)
                                        item['changesPercentage'] = round(res['changesPercentage'], 2)
                                        item['marketCap'] = round(res['marketCap'], 2)

                                    # If the symbol already exists, keep the one with the highest sentiment
                                    if symbol in result_data:
                                        if item['sentiment'] > result_data[symbol]['sentiment']:
                                            result_data[symbol] = item
                                    else:
                                        result_data[symbol] = item

                                except Exception as e:
                                    print(f"Error reading data for {symbol}: {e}")
                                    pass

        except Exception as e:
            print(f"Error fetching data from {source}: {e}")
            pass




    # Convert the result_data dictionary to a list of items
    final_result = list(result_data.values())
    final_result = [{k: v for k, v in item.items() if k != 'lastSentiment'} for item in final_result]

    final_result = sorted(final_result, key=lambda x: x['sentiment'], reverse=True)
    
    for index, stock in enumerate(final_result, start=1):
        stock['rank'] = index


    # Save the combined result as a single JSON file
    if final_result:
        await save_json(final_result)


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