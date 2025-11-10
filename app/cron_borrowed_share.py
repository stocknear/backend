import ujson
import asyncio
import aiohttp
import sqlite3
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

async def get_data(ticker):
    #https://iborrowdesk.com/api/ticker/LEO
    url = "https://iborrowdesk.com/api/ticker/" + ticker.upper()
    try:
        r = requests.get(url, headers=headers)
        data = r.json()['daily']
        # Desired keys to keep
        keys_to_keep = ["available", "date", "fee", "rebate"]

        # Filtering the dictionaries
        filtered_data = [{k: v for k, v in entry.items() if k in keys_to_keep} for entry in data]
        return filtered_data
    except Exception as e:
        print(e)
        return []

async def save_json(symbol, data):
    # Use async file writing to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    path = f"json/borrowed-share/companies/{symbol}.json"
    await loop.run_in_executor(None, ujson.dump, data, open(path, 'w'))

async def process_ticker(ticker):
    data = await get_data(ticker)
    if len(data)>0:
        await save_json(ticker, data)

async def run():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketcap >=1E9 AND symbol NOT LIKE '%.%'")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    con.close()
    etf_con.close()

    total_symbols = stocks_symbols #+ etf_symbols

    async with aiohttp.ClientSession() as session:
        tasks = []
        for ticker in total_symbols:
            tasks.append(process_ticker(ticker))
        
        # Run tasks concurrently in batches to avoid too many open connections
        batch_size = 10  # Adjust based on your system's capacity
        for i in tqdm(range(0, len(tasks), batch_size)):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch)
            await asyncio.sleep(30)

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(f"An error occurred: {e}")




#If url fails in the future this is the way to get the data from the source
'''
from ftplib import FTP
import pandas as pd
from datetime import datetime, timedelta

# FTP server credentials
ftp_host = "ftp2.interactivebrokers.com"
ftp_user = "shortstock"
ftp_password = ""  # Replace with actual password

# File to download
filename = "usa.txt"

# Connect to FTP server
try:
    ftp = FTP(ftp_host)
    ftp.login(user=ftp_user, passwd=ftp_password)
    print(f"Connected to {ftp_host}")
    
    # Download the file
    with open(filename, 'wb') as file:
        ftp.retrbinary(f"RETR {filename}", file.write)
    print(f"Downloaded {filename}")

    # Close FTP connection
    ftp.quit()

    # Process the downloaded file (assuming it's a pipe-separated CSV)
    df = pd.read_csv(filename, sep="|", skiprows=1)
    df = df[["#SYM", "FEERATE", "AVAILABLE"]]  # Adjust columns as per your actual file structure
    df.columns = ["ticker", "fee", "available"]
    df["available"] = df["available"].replace(">10000000", 10000000)
    
    # Append additional data to df if needed (e.g., full_ticker_df)
    # df = df.append(full_ticker_df)
    
    df = df.drop_duplicates(subset="ticker", keep="first")
    df["date_updated"] = datetime.utcnow() - timedelta(hours=5)
    df.fillna(0, inplace=True)
    print(df)
    # Save processed data to CSV
    processed_filename = "usa_processed.csv"
    df.to_csv(processed_filename, index=False)
    #print(f"Processed data saved to {processed_filename}")

except Exception as e:
    print(f"Error: {e}")
'''