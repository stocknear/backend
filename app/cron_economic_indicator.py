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
async def save_json(data):
    os.makedirs('json/economic-indicator', exist_ok=True)
    with open(f'json/economic-indicator/data.json', 'w') as file:
        ujson.dump(data, file)

# Function to fetch data from the API
async def get_data(session, url):
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

async def get_gdp():
    start_date = datetime(2000,1,1).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v4/economic?name=GDP&from={start_date}&to={end_date}&apikey={api_key}"
        data = await get_data(session,url)
        data = sorted(data, key=lambda x: x['date'])

    return data

async def get_real_gdp():
    start_date = datetime(2000,1,1).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v4/economic?name=realGDP&from={start_date}&to={end_date}&apikey={api_key}"
        data = await get_data(session,url)
        data = sorted(data, key=lambda x: x['date'])
        
    return data

async def get_real_gdp_per_capita():
    start_date = datetime(2000,1,1).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v4/economic?name=realGDPPerCapita&from={start_date}&to={end_date}&apikey={api_key}"
        data = await get_data(session,url)
        data = sorted(data, key=lambda x: x['date'])
        
    return data

async def get_unemployment_rate():
    start_date = datetime(2000,1,1).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v4/economic?name=unemploymentRate&from={start_date}&to={end_date}&apikey={api_key}"
        data = await get_data(session,url)
        data = sorted(data, key=lambda x: x['date'])
        
    return data

async def get_recession_probability():
    start_date = datetime(2000,1,1).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v4/economic?name=smoothedUSRecessionProbabilities&from={start_date}&to={end_date}&apikey={api_key}"
        data = await get_data(session,url)
        data = sorted(data, key=lambda x: x['date'])
        
    return data

async def get_inflation_rate():
    start_date = datetime(2000,1,1).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v4/economic?name=inflationRate&from={start_date}&to={end_date}&apikey={api_key}"
        data = await get_data(session,url)
        data = sorted(data, key=lambda x: x['date'])
        data = [entry for entry in data if datetime.strptime(entry['date'], "%Y-%m-%d").day == 1]
    return data

async def get_fed_fund_rate():
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v4/economic?name=federalFunds&apikey={api_key}"
        data = await get_data(session,url)
        data = sorted(data[:300], key=lambda x: x['date'])
    return data


# Main function to manage the date iteration and API calls
async def run():
    cpi = await get_cpi()
    treasury = await get_treasury()
    unemployment_rate = await get_unemployment_rate()
    #recession_probability = await get_recession_probability()
    gdp = await get_gdp()
    real_gdp = await get_real_gdp()
    real_gdp_per_capita = await get_real_gdp_per_capita()
    inflation_rate = await get_inflation_rate()
    fed_fund_rate = await get_fed_fund_rate()
    data = {
        'cpi': cpi,
        'treasury': treasury,
        'unemploymentRate': unemployment_rate,
        #'recessionProbability': recession_probability,
        'gdp': gdp,
        'realGDP': real_gdp,
        'realGDPPerCapita': real_gdp_per_capita,
        'inflationRate': inflation_rate,
        'fedFundRate': fed_fund_rate,
    }

    await save_json(data)


# Run the asyncio event loop
loop = asyncio.get_event_loop()
loop.run_until_complete(run())
