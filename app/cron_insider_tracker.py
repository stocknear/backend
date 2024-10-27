import ujson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
from itertools import groupby
from operator import itemgetter
from aiofiles import open as async_open
from tqdm import tqdm
from dotenv import load_dotenv
import os
import time

load_dotenv()
api_key = os.getenv('FMP_API_KEY')


async def save_json(data):
    async with async_open(f"json/tracker/insider/data.json", 'w') as file:
        await file.write(ujson.dumps(data))


def format_name(name):
    """
    Formats a name from "LASTNAME MIDDLE FIRSTNAME" format to "Firstname Middle Lastname"
    
    Args:
        name (str): Name in uppercase format (e.g., "SINGLETON J MATTHEW")
    
    Returns:
        str: Formatted name (e.g., "Matthew J Singleton")
    """
    # Split the name into parts
    parts = name.strip().split()
    
    # Handle empty string or single word
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0].capitalize()
    
    # The first part is the last name
    lastname = parts[0].capitalize()
    
    # The remaining parts are in reverse order
    other_parts = parts[1:]
    other_parts.reverse()
    
    # Capitalize each part
    other_parts = [part.capitalize() for part in other_parts]
    
    # Join all parts
    return " ".join(other_parts + [lastname])

def aggregate_transactions(transactions, min_value=100_000):

    # Sort transactions by the keys we want to group by
    sorted_transactions = sorted(
        transactions,
        key=lambda x: (x['reportingName'], x['symbol'], x['transactionType'])
    )
    
    # Group by reportingName, symbol, and transactionType
    result = []
    for key, group in groupby(
        sorted_transactions,
        key=lambda x: (x['reportingName'], x['symbol'], x['transactionType'])
    ):
        group_list = list(group)
        
        # Calculate average value
        avg_value = sum(t['value'] for t in group_list) / len(group_list)
        
        # Only include transactions with average value >= min_value
        if avg_value >= min_value:
            # Find latest filing date
            latest_date = max(
                datetime.strptime(t['filingDate'], '%Y-%m-%d %H:%M:%S')
                for t in group_list
            ).strftime('%Y-%m-%d %H:%M:%S')
            
            # Create aggregated transaction with formatted name
            result.append({
                'reportingName': format_name(key[0]),
                'symbol': key[1],
                'transactionType': key[2],
                'filingDate': latest_date,
                'avgValue': avg_value
            })
    
    # Sort the final result by filingDate
    return sorted(result, key=lambda x: x['filingDate'], reverse=True)


async def get_data(session, symbols):
    res_list = []
    for page in range(0, 20):  # Adjust the number of pages as needed
        url = f"https://financialmodelingprep.com/api/v4/insider-trading?page={page}&apikey={api_key}"
        async with session.get(url) as response:
            try:
                if response.status == 200:
                    data = await response.json()
                    
                    # Filter and adjust transactionType based on acquistionOrDisposition
                    filtered_data = [
                        {
                            "reportingName": item.get("reportingName"),
                            "symbol": item.get("symbol"),
                            "filingDate": item.get("filingDate"),
                            "value": round(item.get("securitiesTransacted") * item.get("price"),2),
                            "transactionType": "Buy" if item.get("acquistionOrDisposition") == "A" 
                                                else "Sell" if item.get("acquistionOrDisposition") == "D" 
                                                else None,  # None if neither "A" nor "D"
                        }
                        for item in data
                        if item.get("acquistionOrDisposition") in ["A", "D"] and item.get('price') > 0 and item.get("securitiesTransacted") > 0  # Filter out if not "A" or "D"
                    ]
                    
                    res_list += filtered_data
                else:
                    print(f"Failed to fetch data. Status code: {response.status}")
            except Exception as e:
                print(f"Error while fetching data: {e}")
                break

    res_list = aggregate_transactions(res_list)
    

    new_data = []
    for item in res_list:
        try:
            symbol = item['symbol']
            with open(f"json/quote/{symbol}.json") as file:
                stock_data = ujson.load(file)
                item['marketCap'] = stock_data['marketCap']
                item['price'] = round(stock_data['price'],2)
                item['changesPercentage'] = round(stock_data['changesPercentage'],2)
                new_data.append({**item})
        except:
            pass

    return new_data


async def run():
    # Connect to SQLite
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    
    # Fetch stock symbols
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    # Fetch data asynchronously using aiohttp
    async with aiohttp.ClientSession() as session:
        data = await get_data(session, stock_symbols)
        if len(data) > 0:
            print(f"Fetched {len(data)} records.")
            await save_json(data)


try:
    asyncio.run(run())
except Exception as e:
    print(f"Error: {e}")
