from datetime import datetime, timedelta
import ujson
import time
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
import time 
import asyncio
import aiohttp
from faker import Faker
from tqdm import tqdm

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')


def generate_female_names(num_names):
    fake = Faker()
    female_names = []

    for _ in range(num_names):
        female_names.append(fake.first_name_female())

    return female_names

def generate_male_names(num_names):
    fake = Faker()
    male_names = []
    for _ in range(num_names):
        male_names.append(fake.first_name_male())

    return male_names


# Specify the number of female names you want in the list
number_of_names = 20_000
female_names_list = generate_female_names(number_of_names)
male_names_list = generate_male_names(number_of_names)


def custom_sort(entry):
    # Ensure "Chief Executive Officer" appears first, then sort by name
    if "Chief Executive Officer" in entry['title']:
        return (0, entry['name'])
    else:
        return (1, entry['name'])


async def save_executives(session, symbol):
    url = f"https://financialmodelingprep.com/api/v3/key-executives/{symbol}?apikey={api_key}"
    async with session.get(url) as response:
        data = await response.json()
    unique_names = set()
    filtered_data = []

    for entry in sorted(data, key=custom_sort):
        name = entry['name']
        if name not in unique_names:
            unique_names.add(name)
            filtered_data.append(entry)
    
    for entry in filtered_data:
        if entry['gender'] == '' or entry['gender'] == None:
            if any(substring.lower() in entry['name'].lower() for substring in female_names_list):
                #print(entry['name'])
                entry['gender'] = 'female'
            elif any(substring.lower() in entry['name'].lower() for substring in male_names_list):
                #print(entry['name'])
                entry['gender'] = 'male'

    if len(filtered_data) > 0:
        with open(f"json/executives/{symbol}.json", 'w') as file:
            ujson.dump(filtered_data, file)

async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    async with aiohttp.ClientSession() as session:
        tasks = []
        i = 0
        for symbol in tqdm(symbols):
            tasks.append(save_executives(session, symbol))

            i += 1
            if i % 800 == 0:
                await asyncio.gather(*tasks)
                tasks = []
                print('sleeping mode: ', i)
                await asyncio.sleep(60)  # Pause for 60 seconds

        #tasks.append(self.save_ohlc_data(session, "%5EGSPC"))
        
        if tasks:
            await asyncio.gather(*tasks)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())