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


def get_sector_path(sector):
    sector_paths = {
        'basicMaterialsChangesPercentage': "basic-materials",
        'communicationServicesChangesPercentage': "communication-services",
        'consumerCyclicalChangesPercentage': "consumer-cyclical",
        'consumerDefensiveChangesPercentage': "consumer-defensive",
        'financialServicesChangesPercentage': "financial",
        'industrialsChangesPercentage': "industrials",
        'energyChangesPercentage': "energy",
        'utilitiesChangesPercentage': "utilities",
        'realEstateChangesPercentage': "real-estate",
        'technologyChangesPercentage': "technology",
        'healthcareChangesPercentage': 'healthcare',
    }
    return sector_paths.get(sector, None)

# Function to save JSON data
async def save_json(data, name):
    os.makedirs('json/sector', exist_ok=True)
    with open(f'json/sector/{name}.json', 'w') as file:
        ujson.dump(data, file)

# Function to fetch data from the API
async def get_data(session, start_date, end_date):
    url = f"https://financialmodelingprep.com/api/v3/historical-sectors-performance?from={start_date}&to={end_date}&apikey={api_key}"
    async with session.get(url) as response:
        data = await response.json()
        return data

# Main function to manage the date iteration and API calls
async def run():
    sector_data = defaultdict(list)
    start_date = datetime.now() - timedelta(days=180)
    today = datetime.now()

    async with aiohttp.ClientSession() as session:
        while start_date <= today:
            # Calculate the next end_date, ensuring it doesn't go beyond today
            end_date = min(start_date + timedelta(days=30), today)

            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            data = await get_data(session, start_str, end_str)
            if data:
                for item in data:
                    date = item['date']
                    for sector_key, sector_value in item.items():
                        if sector_key == 'date':
                            continue
                        sector_name = get_sector_path(sector_key)
                        if sector_name:
                            sector_data[sector_name].append({
                                'date': date,
                                'changesPercentage': round(sector_value,3)
                            })

            # Update start_date for the next loop iteration
            start_date = end_date + timedelta(days=1)

    # Save each sector's data as a separate JSON file
    for sector, records in sector_data.items():
        records = sorted(records, key=lambda x: x['date'])
        await save_json(records, sector)

    return sector_data

# Run the asyncio event loop
loop = asyncio.get_event_loop()
sector_results = loop.run_until_complete(run())
