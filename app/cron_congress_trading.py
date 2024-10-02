import ujson
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

load_dotenv()
api_key = os.getenv('FMP_API_KEY')


async def save_json_data(symbol, data):
    async with aiofiles.open(f"json/congress-trading/company/{symbol}.json", 'w') as file:
        await file.write(ujson.dumps(data))

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

def replace_representative(office):
    replacements = {
        'Carper, Thomas R. (Senator)': 'Tom Carper',
        'Thomas R. Carper': 'Tom Carper',
        'Tuberville, Tommy (Senator)': 'Tommy Tuberville',
        'Ricketts, Pete (Senator)': 'John Ricketts',
        'Pete Ricketts': 'John Ricketts',
        'Moran, Jerry (Senator)': 'Jerry Moran',
        'Fischer, Deb (Senator)': 'Deb Fischer',
        'Mullin, Markwayne (Senator)': 'Markwayne Mullin',
        'Whitehouse, Sheldon (Senator)': 'Sheldon Whitehouse',
        'Toomey, Pat (Senator)': 'Pat Toomey',
        'Sullivan, Dan (Senator)': 'Dan Sullivan',
        'Capito, Shelley Moore (Senator)': 'Shelley Moore Capito',
        'Roberts, Pat (Senator)': 'Pat Roberts',
        'King, Angus (Senator)': 'Angus King',
        'Hoeven, John (Senator)': 'John Hoeven',
        'Duckworth, Tammy (Senator)': 'Tammy Duckworth',
        'Perdue, David (Senator)': 'David Perdue',
        'Inhofe, James M. (Senator)': 'James M. Inhofe',
        'Murray, Patty (Senator)': 'Patty Murray',
        'Boozman, John (Senator)': 'John Boozman',
        'Loeffler, Kelly (Senator)': 'Kelly Loeffler',
        'Reed, John F. (Senator)': 'John F. Reed',
        'Collins, Susan M. (Senator)': 'Susan M. Collins',
        'Cassidy, Bill (Senator)': 'Bill Cassidy',
        'Wyden, Ron (Senator)': 'Ron Wyden',
        'Hickenlooper, John (Senator)': 'John Hickenlooper',
        'Booker, Cory (Senator)': 'Cory Booker',
        'Donald Beyer, (Senator).': 'Donald Sternoff Beyer',
        'Peters, Gary (Senator)': 'Gary Peters',
        'Donald Sternoff Beyer, (Senator).': 'Donald Sternoff Beyer',
        'Donald S. Beyer, Jr.': 'Donald Sternoff Beyer',
        'Donald Sternoff Honorable Beyer': 'Donald Sternoff Beyer',
        'K. Michael Conaway': 'Michael Conaway',
        'C. Scott Franklin': 'Scott Franklin',
        'Robert C. "Bobby" Scott': 'Bobby Scott',
        'Madison Cawthorn': 'David Madison Cawthorn',
        'Cruz, Ted (Senator)': 'Ted Cruz',
        'Smith, Tina (Senator)': 'Tina Smith',
        'Graham, Lindsey (Senator)': 'Lindsey Graham',
        'Hagerty, Bill (Senator)': 'Bill Hagerty',
        'Scott, Rick (Senator)': 'Rick Scott',
        'Warner, Mark (Senator)': 'Mark Warner',
        'McConnell, A. Mitchell Jr. (Senator)': 'Mitch McConnell',
        'Mitchell McConnell': 'Mitch McConnell',
        'Charles J. "Chuck" Fleischmann': 'Chuck Fleischmann',
        'Vance, J.D. (Senator)': 'James Vance',
        'Neal Patrick MD, Facs Dunn': 'Neal Dunn',
        'Neal Patrick MD, Facs Dunn (Senator)': 'Neal Dunn',
        'Neal Patrick Dunn, MD, FACS': 'Neal Dunn',
        'Neal P. Dunn': 'Neal Dunn',
        'Tillis, Thom (Senator)': 'Thom Tillis',
        'W. Gregory Steube': 'Greg Steube',
        'W. Grego Steube': 'Greg Steube',
        'W. Greg Steube': 'Greg Steube',
        'David David Madison Cawthorn': 'David Madison Cawthorn',
        'Blunt, Roy (Senator)': 'Roy Blunt',
        'Thune, John (Senator)': 'John Thune',
        'Rosen, Jacky (Senator)': 'Jacky Rosen',
        'Britt, Katie (Senator)': 'Katie Britt',
        'Britt, Katie': 'Katie Britt',
        'James Costa': 'Jim Costa',
        'Lummis, Cynthia (Senator)': 'Cynthia Lummis',
        'Coons, Chris (Senator)': 'Chris Coons',
        'Udall, Tom (Senator)': 'Tom Udall',
        'Kennedy, John (Senator)': 'John Kennedy',
        'Bennet, Michael (Senator)': 'Michael Bennet',
        'Casey, Robert P. Jr. (Senator)': 'Robert Casey',
        'Van Hollen, Chris (Senator)': 'Chris Van Hollen',
        'Manchin, Joe (Senator)': 'Joe Manchin',
        'Cornyn, John (Senator)': 'John Cornyn',
        'Enzy, Michael (Senator)': 'Michael Enzy',
        'Cardin, Benjamin (Senator)': 'Benjamin Cardin',
        'Kaine, Tim (Senator)': 'Tim Kaine',
        'Joseph P. Kennedy III': 'Joe Kennedy',
        'James E Hon Banks': 'Jim Banks',
        'Michael F. Q. San Nicolas': 'Michael San Nicolas',
        'Barbara J Honorable Comstock': 'Barbara Comstock',
        'Darin McKay LaHood': 'Darin LaHood',
        'Mr ': '',
        'Mr. ': '',
        'Dr ': '',
        'Dr. ': '',
        'Mrs ': '',
        'Mrs. ': '',
        '(Senator)': '',
    }

    for old, new in replacements.items():
        office = office.replace(old, new)
        office = ' '.join(office.split())
    return office

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
        url_senate = f"https://financialmodelingprep.com/api/v4/senate-trading?symbol={symbol}&apikey={api_key}"
        url_house = f"https://financialmodelingprep.com/api/v4/senate-disclosure?symbol={symbol}&apikey={api_key}"
        
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
                    if 'dateRecieved' in item:
                        item['disclosureDate'] = item['dateRecieved']

                res_list +=data

        res_list = sorted(res_list, key=lambda x: x['transactionDate'], reverse=True)
                
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")

    return res_list


def create_politician_db(data, stock_symbols, stock_raw_data, etf_symbols, etf_raw_data, crypto_symbols, crypto_raw_data):
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
        elif ('ticker' in item and item['ticker'] in crypto_symbols) or ('symbol' in item and item['symbol'] in crypto_symbols):
            for j in crypto_raw_data:
                if (item.get('ticker') or (item.get('symbol'))) == j['symbol']:
                    item['ticker'] = j['symbol']
                    item['name'] = j['name']
                    item['assetType'] = 'crypto'
                    break

        grouped_data[item['id']].append(item)

    # Convert defaultdict to list
    grouped_data_list = list(grouped_data.values())

    for item in tqdm(grouped_data_list):
        # Sort items by 'transactionDate'
        item = sorted(item, key=lambda x: x['transactionDate'], reverse=True)

        # Calculate top sectors
        sector_counts = Counter()
        for holding in item:
            symbol = holding['ticker']
            sector = next((entry['sector'] for entry in stock_raw_data if entry['symbol'] == symbol), None)
            if sector:
                sector_counts[sector] += 1

        # Calculate the total number of holdings
        total_holdings = sum(sector_counts.values())

        # Calculate the percentage for each sector and get the top 5
        top_5_sectors_percentage = [
            {sector: round((count / total_holdings) * 100, 2)}
            for sector, count in sector_counts.most_common(5)
        ]

        # Prepare the data to save in the file
        result = {
            'topSectors': top_5_sectors_percentage,
            'history': item
        }

        # Save to JSON file
        with open(f"json/congress-trading/politician-db/{item[0]['id']}.json", 'w') as file:
            ujson.dump(result, file)


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
            with open(file_path, 'r') as file:
                data = ujson.load(file)
                
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
    with open('json/congress-trading/search_list.json', 'w') as file:
        ujson.dump(search_politician_list, file)

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

        crypto_con = sqlite3.connect('crypto.db')
        crypto_cursor = crypto_con.cursor()
        crypto_cursor.execute("PRAGMA journal_mode = wal")
        crypto_cursor.execute("SELECT DISTINCT symbol, name FROM cryptos")
        crypto_raw_data = crypto_cursor.fetchall()
        crypto_raw_data = [{
            'symbol': row[0],
            'name': row[1],
        } for row in crypto_raw_data]
        crypto_symbols = [item['symbol'] for item in crypto_raw_data]
        crypto_con.close()

        total_symbols = crypto_symbols +etf_symbols + stock_symbols
        chunk_size = 500
        politician_list = []

    except Exception as e:
        print(f"Failed to fetch symbols: {e}")
        return

    try:
        
        connector = aiohttp.TCPConnector(limit=100)  # Adjust the limit as needed
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in tqdm(range(0, len(total_symbols), chunk_size)):
                symbols_chunk = total_symbols[i:i + chunk_size]
                data = await get_congress_data(symbols_chunk,session)
                politician_list +=data
                print('sleeping for 60 sec')
                await asyncio.sleep(60)  # Wait for 60 seconds between chunks
        
        
        create_politician_db(politician_list, stock_symbols, stock_raw_data, etf_symbols, etf_raw_data, crypto_symbols, crypto_raw_data)
        create_search_list()

    except Exception as e:
        print(f"Failed to run fetch and save data: {e}")

try:
    asyncio.run(run())
except Exception as e:
    print(e)

