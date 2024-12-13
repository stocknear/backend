from datetime import datetime, timedelta
import orjson
import time
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
import asyncio
import aiohttp
from tqdm import tqdm
from dotenv import load_dotenv
import os
import re

load_dotenv()
api_key = os.getenv('FMP_API_KEY')

query_template = """
    SELECT 
        profile
    FROM 
        stocks 
    WHERE
        symbol = ?
"""

MONTH_MAP = {
    '01': 'January', '02': 'February', '03': 'March', '04': 'April',
    '05': 'May', '06': 'June', '07': 'July', '08': 'August',
    '09': 'September', '10': 'October', '11': 'November', '12': 'December'
}

STATE_MAP = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}



async def save_json(symbol, data):
    with open(f"json/profile/{symbol}.json", 'w') as file:
        file.write(orjson.dumps(data).decode('utf-8'))

def custom_sort(entry):
    title_lower = entry['position'].lower()
    # Most priority: CEO or Chief Executive Officer
    ceo_keywords = ['ceo', 'chief executive officer']
    if any(keyword in title_lower for keyword in ceo_keywords):
        return (0, 0, entry['name'])
    
    # Second priority: Other Chief-level positions
    chief_keywords = [
        'chief financial officer', 
        'chief operating officer', 
        'chief technology officer', 
        'chief information officer',
        'chief marketing officer',
        'chief legal officer',
        'chief people officer'
    ]
    if any(keyword in title_lower for keyword in chief_keywords):
        return (0, 1, entry['name'])
    
    # Lowest priority: Other positions
    return (1, 0, entry['name'])

def sort_executives(executives):
    return sorted(executives, key=custom_sort)

async def fetch_sec_filings(session, symbol):
    url = f"https://financialmodelingprep.com/api/v3/sec_filings/{symbol}?limit=10&page=0&apikey={api_key}"
    async with session.get(url) as response:
        data = await response.json()
    
    def get_filing_title(filing_type):
        if "/A" in filing_type:
            prefix = "[Amend] "
            filing_type = filing_type.replace("/A", "")
        else:
            prefix = ""
        
        if filing_type == "8-K":
            return f"{prefix}Current Report"
        elif filing_type == "10-Q":
            return f"{prefix}Quarterly Report"
        elif filing_type == "10-K":
            return f"{prefix}Annual Report"
        elif filing_type == "13F-HR":
            return f"{prefix}Quarterly report filed by institutional managers, holdings"
        elif filing_type == "SC 13G":
            return f"{prefix}Statement of acquisition of beneficial ownership by individuals"
        elif filing_type == "S-3ASR":
            return f"{prefix}Automatic shelf registration statement of securities of well-known seasoned issuers"
        else:
            return f"{prefix}Filing"
    
    return [
        {
            'date': datetime.strptime(entry['fillingDate'], "%Y-%m-%d %H:%M:%S").strftime("%b %d, %Y"),
            'type': entry['type'],
            'title': get_filing_title(entry['type']),
            'link': entry['finalLink']
        } 
        for entry in data
    ]

async def fetch_executives(session, symbol):
    url = f"https://financialmodelingprep.com/api/v3/key-executives/{symbol}?apikey={api_key}"
    async with session.get(url) as response:
        data = await response.json()
    
    # Clean and process executives
    processed_executives = []
    for item in data:
        try:
            clean_name = item['name'].replace("Ms.","").replace("Mr.","").replace("Mrs.","").replace("Ms","").replace("Mr","").strip()
            processed_executives.append({'name': clean_name,'position': item['title']})
        except:
            pass
    
    # Sort executives to put CEO first
    sorted_executives = sort_executives(processed_executives)
    
    return sorted_executives


async def fetch_company_core_information(session, symbol):
    url = f"https://financialmodelingprep.com/api/v4/company-core-information?symbol={symbol}&apikey={api_key}"
    async with session.get(url) as response:
        data = await response.json()
    
    if not data:
        return {}

    company_info = data[0]

    # Convert fiscalYearEnd to "Month1-Month2" format
    fiscal_year_end = company_info.get('fiscalYearEnd')
    if fiscal_year_end:
        month_end = fiscal_year_end.split('-')[0]
        month_name_end = MONTH_MAP.get(month_end, '')
        
        # Find the start month by getting the next month after the end month
        month_end_num = int(month_end)
        month_start_num = (month_end_num % 12) + 1
        month_name_start = MONTH_MAP.get(f"{month_start_num:02}", '')

        company_info['fiscalYearRange'] = f"{month_name_start} - {month_name_end}"


    keys_to_remove = ['businessAddress', 'mailingAddress','sicDescription', 'registrantName', 'stateOfIncorporation', 'fiscalYearEnd']

    # Creating a new dictionary without the unwanted keys
    company_info = {key: value for key, value in company_info.items() if key not in keys_to_remove}

    return company_info

async def get_data(session, symbol, con):
    try:
        df = pd.read_sql_query(query_template, con, params=(symbol,))
        if df.empty:
            return

        data= df.to_dict(orient='records')[0]
        company_profile =  orjson.loads(data['profile'])[0]
        if company_profile['state'] in STATE_MAP:
            company_profile['state'] = STATE_MAP[company_profile['state']]



        company_profile['ceo'] = company_profile['ceo'].replace("Ms.","").replace("Mr.","").replace("Mrs.","").replace("Ms","").replace("Mr","").strip()

        keys_to_keep = ['currency', 'country', 'description', 'isin', 'cusip', 'sector','industry', 'ceo','website','fullTimeEmployees','address','city','state','ipoDate']

        # Creating a new dictionary without the unwanted keys
        company_profile = {key: value for key, value in company_profile.items() if key in keys_to_keep}

        # Fetch SEC filings
        filings = await fetch_sec_filings(session, symbol)
        
        # Fetch executives
        executives = await fetch_executives(session, symbol)
        
        # Fetch company core information
        core_info = await fetch_company_core_information(session, symbol)
        
        #print(filings)
        #print(executives)
        res = {**company_profile,**core_info, 'executives': executives, 'filings': filings}
        
        if len(res) > 0:
            await save_json(symbol, res)

    except Exception as e:
        print(f"Error processing {symbol}: {e}")

async def run():

    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    symbols = [row[0] for row in cursor.fetchall()]
    
   
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, symbol in enumerate(tqdm(symbols), 1):
            tasks.append(get_data(session, symbol, con))
            
            # Batch processing and rate limiting
            if i % 100 == 0:
                await asyncio.gather(*tasks)
                tasks = []
                print(f'Processed {i} symbols, sleeping...')
                await asyncio.sleep(60)
        
        # Process any remaining tasks
        if tasks:
            await asyncio.gather(*tasks)

    con.close()

def main():
    """
    Entry point for the script.
    """
    asyncio.run(run())

if __name__ == "__main__":
    main()