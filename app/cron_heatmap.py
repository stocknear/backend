from mixpanel_utils import MixpanelUtils
import ujson
import asyncio
import aiohttp
from datetime import datetime, timedelta
from collections import Counter, OrderedDict

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')



async def get_quote_of_stocks(ticker_list):
    ticker_str = ','.join(ticker_list)
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v3/quote/{ticker_str}?apikey={api_key}" 
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return []

async def run():
    index_list = ['sp500', 'nasdaq', 'dowjones']
    for index in index_list:
        url = f"https://financialmodelingprep.com/api/v3/{index}_constituent?apikey={api_key}"
        res_list = []

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()

        for item in data:
            res_list.append({'symbol': item['symbol'], 'sector': item['sector']})

        ticker_list = [item['symbol'] for item in res_list]
        latest_quote = await get_quote_of_stocks(ticker_list)
        for quote in latest_quote:
            symbol = quote['symbol']
            for item in res_list:
                if item['symbol'] == symbol:
                    item['changesPercentage'] = round(quote['changesPercentage'],2)
                    item['marketCap'] = quote['marketCap']
        
        # Create a dictionary to store sectors and their corresponding symbols and percentages
        sector_dict = {}

        for item in res_list:
            sector = item['sector']
            symbol = item['symbol']
            percentage = item['changesPercentage']
            marketCap = item['marketCap']
            
            if sector not in sector_dict:
                sector_dict[sector] = {'name': sector, 'value': 0, 'children': []}
            
            sector_dict[sector]['value'] += marketCap
            sector_dict[sector]['children'].append({'name': symbol, 'value': marketCap, 'changesPercentage': percentage})

        # Convert the dictionary to a list
        result_list = list(sector_dict.values())

        # Optionally, if you want to add the 'value' for each sector
        for sector in result_list:
            sector['value'] = round(sector['value'], 2)
        #print(result_list)

        with open(f"json/heatmaps/{index}.json", 'w') as file:
            ujson.dump(result_list, file)


asyncio.run(run())