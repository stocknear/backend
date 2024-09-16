import ujson
import asyncio
import aiohttp
import os
import sqlite3
from tqdm import tqdm
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()
benzinga_api_key = os.getenv('benzinga_api_key')
fmp_api_key = os.getenv('FMP_API_KEY')


url = "https://api.benzinga.com/api/v2.1/calendar/fda"
querystring = {"token":benzinga_api_key}
headers = {"accept": "application/json"}


async def save_json(data):
    with open(f"json/fda-calendar/data.json", 'w') as file:
        ujson.dump(data, file)

async def get_quote_of_stocks(ticker):
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={fmp_api_key}" 
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {}

async def get_data():

    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    try:
        response = requests.request("GET", url, headers=headers, params=querystring)
        data = ujson.loads(response.text)['fda']
        # New list to store the extracted information
        extracted_data = []

        # Iterate over the original data to extract required fields
        for entry in tqdm(data):
            try:
                symbol = entry['companies'][0]['securities'][0]['symbol']
                if symbol in stock_symbols:
                    name = entry['companies'][0]['name']
                    drug_name = entry['drug']['name'].capitalize()
                    indication = entry['drug']['indication_symptom']
                    outcome = entry['outcome']
                    source_type = entry['source_type']
                    status = entry['status']
                    target_date = entry['target_date']
                    
                    changes_percentage = round((await get_quote_of_stocks(symbol))[0]['changesPercentage'] ,2)

                    # Create a new dictionary with the extracted information
                    new_entry = {
                        'symbol': symbol,
                        'name': name,
                        'drugName': drug_name,
                        'indication': indication,
                        'outcome': outcome,
                        'sourceType': source_type,
                        'status': status,
                        'targetDate': target_date,
                        'changesPercentage': changes_percentage
                    }
                
                    # Append the new dictionary to the new list
                    extracted_data.append(new_entry)
            except:
                pass

        # Output the new list
        return extracted_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []


async def run():
    data = await get_data()
    await save_json(data)

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(f"An error occurred: {e}")