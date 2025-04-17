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
import pandas as pd

load_dotenv()
api_key = os.getenv('FMP_API_KEY')


async def save_json(data):
    async with async_open(f"json/tracker/insider/data.json", 'w') as file:
        await file.write(ujson.dumps(data))

def remove_outliers(group, tolerance=0.5):
    # Calculate the median price
    median_price = group['price'].median()
    
    # Define the lower and upper bounds within 50% of the median
    lower_bound = median_price * (1 - tolerance)
    upper_bound = median_price * (1 + tolerance)
    
    # Filter the group based on these bounds
    return group[(group['price'] >= lower_bound) & (group['price'] <= upper_bound)]


def format_name(name):
    # Split the name into parts
    parts = name.strip().split()

    # Handle empty string or single word
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0].capitalize()

    # Remove the first part if it's a single letter
    if len(parts[0]) == 1:
        parts = parts[1:]

    # Remove the last part if it's a single letter or ends with a dot
    if len(parts[-1]) == 1 or parts[-1].endswith("."):
        parts = parts[:-1]

    # Define abbreviations to be fully capitalized
    abbreviations = {"llc", "inc", "ltd", "corp", "co"}

    # Capitalize each part, handle abbreviations, and preserve numbers
    formatted_parts = [
        part.upper() if part.lower().strip(",.") in abbreviations else part.capitalize()
        for part in parts
    ]

    # Join the parts to form the final name
    return " ".join(formatted_parts)


def aggregate_transactions(transactions, min_value=1E6):

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
        
        # Calculate the total number of shares in the group
        total_shares = sum(t['shares'] for t in group_list)
        
        # Only include transactions with average value >= min_value
        if avg_value >= min_value:
            # Find latest filing date
            latest_date = max(
                #datetime.strptime(t['filingDate'], '%Y-%m-%d %H:%M:%S')
                datetime.strptime(t['filingDate'], '%Y-%m-%d')
                for t in group_list
            ).strftime('%Y-%m-%d')
            
            # Create aggregated transaction with formatted name and total shares
            result.append({
                'reportingName': format_name(key[0]),
                'symbol': key[1],
                'transactionType': key[2],
                'filingDate': latest_date,
                'avgValue': avg_value,
                'totalShares': total_shares  # Added total shares here
            })
    
    # Sort the final result by filingDate
    return sorted(result, key=lambda x: x['filingDate'], reverse=True)



async def get_data(session, symbols):
    res_list = []
    for page in range(0, 100):  # Adjust the number of pages as needed
        url = f"https://financialmodelingprep.com/stable/insider-trading/latest?page={page}&apikey={api_key}"
        async with session.get(url) as response:
            try:
                if response.status == 200:
                    data = await response.json()
                    
                    # Filter and adjust transactionType based on acquisitionOrDisposition
                    filtered_data = [
                        {
                            "reportingName": item.get("reportingName"),
                            "symbol": item.get("symbol"),
                            "filingDate": item.get("filingDate"),
                            "shares": item.get("securitiesTransacted"),
                            "value": round(item.get("securitiesTransacted",0) * item.get("price",0),2),
                            "price": item.get("price",0),
                            "transactionType": item.get("transactionType",None),
                            
                            #"transactionType": "Buy" if item.get("acquisitionOrDisposition") == "A" 
                            #                    else "Sell" if item.get("acquisitionOrDisposition") == "D" 
                            #                    else None,  # None if neither "A" nor "D"
                        }
                        for item in data
                        if item.get("acquisitionOrDisposition") in ["A", "D"] and item.get('price') > 0 and item.get("securitiesTransacted") > 0  # Filter out if not "A" or "D"
                    ]
                    res_list += filtered_data
                else:
                    print(f"Failed to fetch data. Status code: {response.status}")
            except Exception as e:
                print(f"Error while fetching data: {e}")
                break

    df = pd.DataFrame(res_list)

    filtered_df = df.groupby('symbol', group_keys=False).apply(remove_outliers)
    res_list = filtered_df.to_dict('records')

    res_list = aggregate_transactions(res_list)

    new_data = []
    for item in res_list:
        try:
            symbol = item['symbol']
            with open(f"json/quote/{symbol}.json") as file:
                stock_data = ujson.load(file)
                item['name'] = stock_data['name']
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
