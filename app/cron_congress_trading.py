import orjson
import asyncio
import aiohttp
import aiofiles
import sqlite3
import pandas as pd
import time
import hashlib
from collections import defaultdict, Counter
from tqdm import tqdm
from dotenv import load_dotenv
import os
from utils.helper import replace_representative

with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}


load_dotenv()
api_key = os.getenv('FMP_API_KEY')


async def save_json_data(symbol, data):
    folder_path = "json/congress-trading/company"
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    file_path = os.path.join(folder_path, f"{symbol}.json")
    async with aiofiles.open(file_path, 'wb') as file:
        await file.write(orjson.dumps(data))



async def get_congress_data(symbols, session):
    tasks = []
    politician_list = []
    for symbol in symbols:
        task = asyncio.create_task(get_endpoints(symbol, session))
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    
    for symbol, response in zip(symbols, responses):
        if len(response) > 0:
            await save_json_data(symbol, response)
            politician_list +=response

    return politician_list

def generate_id(name):
    hashed = hashlib.sha256(name.encode()).hexdigest()
    return hashed[:10]

async def get_endpoints(symbol, session):
    res_list = []
    amount_mapping = {
    '$1,001 -': '$1K-$15K',
    '$1,001 - $15,000': '$1K-$15K',
    '$15,001 - $50,000': '$15K-$50K',
    '$15,001 -': '$15K-$50K',
    '$50,001 - $100,000': '$50K-$100K',
    '$100,001 - $250,000': '$100K-$250K',
    '$100,001 - $500,000': '$100K-$500K',
    '$250,001 - $500,000': '$250K-$500K',
    '$500,001 - $1,000,000': '$500K-$1M',
    '$1,000,001 - $5,000,000': '$1M-$5M',
    'Spouse/DC Over $1,000,000': 'Over $1M'
    }

    congressional_districts = {"UT": "Utah","CA": "California","NY": "New York","TX": "Texas","FL": "Florida","IL": "Illinois","PA": "Pennsylvania","OH": "Ohio","GA": "Georgia","MI": "Michigan","NC": "North Carolina","AZ": "Arizona","WA": "Washington","CO": "Colorado","OR": "Oregon","VA": "Virginia","NJ": "New Jersey","TN": "Tennessee","MA": "Massachusetts","WI": "Wisconsin","SC": "South Carolina","KY": "Kentucky","LA": "Louisiana","AR": "Arkansas","AL": "Alabama","MS": "Mississippi","NDAL": "North Dakota","SDAL": "South Dakota","MN": "Minnesota","IA": "Iowa","OK": "Oklahoma","ID": "Idaho","NH": "New Hampshire","NE": "Nebraska","MTAL": "Montana","WYAL": "Wyoming","WV": "West Virginia","VTAL": "Vermont","DEAL": "Delaware","RI": "Rhode Island","ME": "Maine","HI": "Hawaii","AKAL": "Alaska","NM": "New Mexico","KS": "Kansas","MS": "Mississippi","CT": "Connecticut","MD": "Maryland","NV": "Nevada",}

    try:
        # Form API request URLs
        url_senate = f"https://financialmodelingprep.com/stable/senate-trades?symbol={symbol}&apikey={api_key}"
        url_house = f"https://financialmodelingprep.com/stable/house-trades?symbol={symbol}&apikey={api_key}"
        
        async with session.get(url_senate) as response_senate, session.get(url_house) as response_house:
            data = []
            for count, response in enumerate([response_senate, response_house]):
                data = await response.json()
                for item in data:
                    if count == 0:
                        item['congress'] = 'Senate'
                    elif count == 1:
                        item['congress'] = 'House'

                    item['amount'] = amount_mapping.get(item['amount'], item['amount'])
                    if any('sale' in word.lower() for word in item['type'].split()):
                        item['type'] = 'Sold'
                    if any('purchase' in word.lower() for word in item['type'].split()):
                        item['type'] = 'Bought'
                    if any('exchange' in word.lower() for word in item['type'].split()):
                        item['type'] = 'Exchange'


                    if 'representative' in item:
                        item['representative'] = replace_representative(item['representative'])

                    if 'office' in item:
                        item['representative'] = replace_representative(item['office'])

                    item['id'] = generate_id(item['representative'])

                    if 'district' in item:
                        # Extract state code from the 'district' value
                        state_code = item['district'][:2]
                        
                        # Replace 'district' value with the corresponding value from congressional_districts
                        item['district'] = f"{congressional_districts.get(state_code, state_code)}"

                res_list +=data

        res_list = sorted(res_list, key=lambda x: x['transactionDate'], reverse=True)
                
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")

    return res_list


def create_politician_db(data, stock_symbols, stock_raw_data, etf_symbols, etf_raw_data):
    grouped_data = defaultdict(list)
    # Group elements by id
    for item in data:
        #Bug: Data provider does offer not always ticker but in edge cases symbol (Suck my ass FMP!)
        if ('ticker' in item and item['ticker'] in stock_symbols) or ('symbol' in item and item['symbol'] in stock_symbols):
            for j in stock_raw_data:
                if (item.get('ticker') or (item.get('symbol'))) == j['symbol']:
                    item['ticker'] = j['symbol']
                    item['name'] = j['name']
                    item['assetType'] = 'stock'
                    break
        elif ('ticker' in item and item['ticker'] in etf_symbols) or ('symbol' in item and item['symbol'] in etf_symbols):
            for j in etf_raw_data:
                if (item.get('ticker') or (item.get('symbol'))) == j['symbol']:
                    item['ticker'] = j['symbol']
                    item['name'] = j['name']
                    item['assetType'] = 'etf'
                    break
       

        grouped_data[item['id']].append(item)

    # Convert defaultdict to list
    grouped_data_list = list(grouped_data.values())

    #keys_to_keep = {'dateRecieved', 'id', 'transactionDate', 'representative', 'assetType', 'type', 'disclosureDate', 'symbol', 'name', 'amount'}

    for item in tqdm(grouped_data_list):
        try:
            #item = [{key: entry[key] for key in entry if key in keys_to_keep} for entry in item]
            # Sort items by 'transactionDate'
            item = sorted(item, key=lambda x: x['transactionDate'], reverse=True)

            # Calculate top sectors
            sector_list = []
            industry_list = []
            for item2 in item:
                try:
                    # Try to get 'symbol' first; if it doesn't exist, use 'ticker'
                    symbol = item2.get('symbol') or item2.get('ticker')
                    if not symbol:
                        continue  # Skip if neither 'symbol' nor 'ticker' is present

                    ticker_data = stock_screener_data_dict.get(symbol, {})

                    # Extract specified columns data for each ticker
                    sector = ticker_data.get('sector', None)
                    industry = ticker_data.get('industry', None)
                except:
                    sector = None
                    industry = None

                if sector:
                    sector_list.append(sector)
                if industry:
                    industry_list.append(industry)


            # Get the top 3 most common sectors and industries
            sector_counts = Counter(sector_list)
            industry_counts = Counter(industry_list)
            main_sectors = [item2[0] for item2 in sector_counts.most_common(3)]
            main_industries = [item2[0] for item2 in industry_counts.most_common(3)]

 
            # Prepare the data to save in the file
            result = {
                'mainSectors': main_sectors,
                'mainIndustries': main_industries,
                'history': item
            }

            # Save to JSON file
            if result:
                folder_path = "json/congress-trading/politician-db"
                os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

                file_path = os.path.join(folder_path, f"{item[0]['id']}.json")
                with open(file_path, 'wb') as file:
                    file.write(orjson.dumps(result))

        except Exception as e:
            print(e)


def create_search_list():
    folder_path = 'json/congress-trading/politician-db/'
    # Initialize the list that will hold the search data
    search_politician_list = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            # Open and read the JSON file
            with open(file_path, 'rb') as file:
                data = orjson.loads(file.read())


                # Access the history, which is a list of transactions
                history = data.get('history', [])
                if not history:
                    continue  # Skip if there is no history
                
                # Get the first item in the history list
                first_item = history[0]

                # Filter out senators (assuming you only want to process non-senators)
                if 'Senator' in first_item['representative']:
                    continue

                # Create the politician search entry
                search_politician_list.append({
                    'representative': first_item['representative'],
                    'id': first_item['id'],
                    'totalTrades': len(history),
                    'district': first_item.get('district', ''),
                    'lastTrade': first_item['transactionDate'],
                })

    # Sort the list by the 'lastTrade' date in descending order
    search_politician_list = sorted(search_politician_list, key=lambda x: x['lastTrade'], reverse=True)

    # Write the search list to a JSON file
    with open('json/congress-trading/search_list.json', 'wb') as file:
        file.write(orjson.dumps(search_politician_list))

async def run():
    try:
        con = sqlite3.connect('stocks.db')
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT symbol, name, sector FROM stocks WHERE symbol NOT LIKE '%.%'")
        stock_raw_data = cursor.fetchall()
        stock_raw_data = [{
            'symbol': row[0],
            'name': row[1],
            'sector': row[2],
        } for row in stock_raw_data]

        stock_symbols = [item['symbol'] for item in stock_raw_data]

        con.close()



        etf_con = sqlite3.connect('etf.db')
        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol, name FROM etfs")
        etf_raw_data = etf_cursor.fetchall()
        etf_raw_data = [{
            'symbol': row[0],
            'name': row[1],
        } for row in etf_raw_data]
        etf_symbols = [item['symbol'] for item in etf_raw_data]
        etf_con.close()


        total_symbols = etf_symbols + stock_symbols
        chunk_size = 500
        politician_list = []

    except Exception as e:
        print(f"Failed to fetch symbols: {e}")
        return

    try:
        
        connector = aiohttp.TCPConnector(limit=100)  # Adjust the limit as needed
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in tqdm(range(0, len(total_symbols), chunk_size)):
                try:
                    symbols_chunk = total_symbols[i:i + chunk_size]
                    data = await get_congress_data(symbols_chunk,session)
                    politician_list +=data
                    print('sleeping')
                    await asyncio.sleep(30)
                except Exception as e:
                    print(e)
                    pass
        
        
        create_politician_db(politician_list, stock_symbols, stock_raw_data, etf_symbols, etf_raw_data)
        create_search_list()

    except Exception as e:
        print(f"Failed to run fetch and save data: {e}")

try:
    asyncio.run(run())
except Exception as e:
    print(e)
