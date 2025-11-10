from datetime import datetime
import orjson
import sqlite3
import asyncio
import aiohttp
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

today = datetime.today().strftime('%Y-%m-%d')

async def save_json(symbol, data):
    os.makedirs("json/market-cap/companies", exist_ok=True)
    with open(f"json/market-cap/companies/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))
        print(data)


async def get_data(session, symbol):
    url = (
        f"https://financialmodelingprep.com/stable/historical-market-capitalization"
        f"?symbol={symbol}&from=2000-01-01&to={today}&limit=5000&apikey={api_key}"
    )

    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if data:
                    # Remove duplicates by date
                    unique_res_list = {item['date']: item for item in data}.values()
                    unique_res_list = sorted(unique_res_list, key=lambda x: x['date'])

                    # Filter out 'symbol'
                    filtered_data = [
                        {k: v for k, v in item.items() if k != 'symbol'}
                        for item in unique_res_list
                    ]

                    if filtered_data:
                        await save_json(symbol, filtered_data)
            else:
                print(f"Failed {symbol}: status {response.status}, url={url}")
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")



async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, symbol in enumerate(tqdm(symbols), 1):
            tasks.append(get_data(session, symbol))
            if i % 100 == 0:
                await asyncio.gather(*tasks)
                tasks = []
                print(f'sleeping mode: {i}')
                await asyncio.sleep(60)  # Pause for 60 seconds

        if tasks:
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(run())
