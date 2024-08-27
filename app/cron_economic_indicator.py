from datetime import datetime, timedelta
import ujson
import asyncio
import aiohttp
import os
from dotenv import load_dotenv
from collections import defaultdict  # Import defaultdict

# Load environment variables
load_dotenv()
api_key = os.getenv('FMP_API_KEY')



# Function to save JSON data
async def save_json(data, name):
    os.makedirs('json/economic-indicator', exist_ok=True)
    with open(f'json/economic-indicator/{name}.json', 'w') as file:
        ujson.dump(data, file)

# Function to fetch data from the API
async def get_data(session, url):
    #url = f"https://financialmodelingprep.com/api/v4/treasury?from={start_date}&to={end_date}&apikey={api_key}"
    async with session.get(url) as response:
        data = await response.json()
        return data


async def get_treasury():
    treasury_data = []
    start_date = datetime.now() - timedelta(days=365*10)
    today = datetime.now()
    async with aiohttp.ClientSession() as session:
        while start_date <= today:
            # Calculate the next end_date, ensuring it doesn't go beyond today
            end_date = min(start_date + timedelta(days=30), today)

            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            url = f"https://financialmodelingprep.com/api/v4/treasury?from={start_str}&to={end_str}&apikey={api_key}"
            data = await get_data(session,url)
            if data:
                treasury_data += data

            # Update start_date for the next loop iteration
            start_date = end_date + timedelta(days=1)

    
    treasury_data = sorted(treasury_data, key=lambda x: x['date'])
    #await save_json(treasury_data, 'treasury')
    return treasury_data

async def get_cpi():
    start_date = datetime(2000,1,1).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v4/economic?name=CPI&from={start_date}&to={end_date}&apikey={api_key}"
        data = await get_data(session,url)
        #if data:
        #    await save_json(data, 'cpi')
        data = sorted(data, key=lambda x: x['date'])
    return data

# Main function to manage the date iteration and API calls
async def run():
    cpi = await get_cpi()
    treasury = await get_treasury()
    data = {'cpi': cpi, 'treasury': treasury}
    await save_json(data, 'data')


# Run the asyncio event loop
loop = asyncio.get_event_loop()
loop.run_until_complete(run())
