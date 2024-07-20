import pandas as pd
import numpy as np
import glob
import requests
import os
import sqlite3
from zipfile import ZipFile
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define some configuration variables
OUTPUT_PATH = r"./json/swap"
COMPANIES_PATH = r"./json/swap/companies"
MAX_WORKERS = 4
CHUNK_SIZE = 1000  # Adjust this value based on your system's RAM
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Ensure the companies directory exists
os.makedirs(COMPANIES_PATH, exist_ok=True)

con = sqlite3.connect('stocks.db')

cursor = con.cursor()
cursor.execute("PRAGMA journal_mode = wal")
cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
stock_symbols = [row[0] for row in cursor.fetchall()]

con.close()



start = datetime.datetime.today() - datetime.timedelta(days=180)
end = datetime.datetime.today()
dates = [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]

# Generate filenames for each date
filenames = [
    f"SEC_CUMULATIVE_EQUITIES_{year}_{month}_{day}.zip"
    for year, month, day in [
        (date.strftime("%Y"), date.strftime("%m"), date.strftime("%d"))
        for date in dates
    ]
]


def download_and_process(filename):
    csv_output_filename = os.path.join(OUTPUT_PATH, filename.replace('.zip', '.csv'))
    if os.path.exists(csv_output_filename ):
        print(f"{csv_output_filename} already exists. Skipping download and processing.")
        return

    url = f"https://pddata.dtcc.com/ppd/api/report/cumulative/sec/{filename}"
    req = requests.get(url)
    if req.status_code != 200:
        print(f"Failed to download {url}")
        return
    
    with open(filename, "wb") as f:
        f.write(req.content)
    
    with ZipFile(filename, "r") as zip_ref:
        csv_filename = zip_ref.namelist()[0]
        zip_ref.extractall()
    
    output_filename = os.path.join(OUTPUT_PATH, f"{csv_filename}")
    
    # Process the CSV in chunks
    chunk_list = []
    for chunk in pd.read_csv(csv_filename, chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines="skip"):
        chunk_list.append(chunk)
    
    # Concatenate chunks and save
    pd.concat(chunk_list, ignore_index=True).to_csv(output_filename, index=False)
    
    # Delete original downloaded files

    os.remove(filename)
    os.remove(csv_filename)

tasks = []
for filename in filenames:
    tasks.append(executor.submit(download_and_process, filename))

for task in tqdm(as_completed(tasks), total=len(tasks)):
    pass

files = glob.glob(OUTPUT_PATH + "/" + "*")

def process_and_save_by_ticker():
    csv_files = glob.glob(os.path.join(OUTPUT_PATH, "*.csv"))
    
    # Initialize DataFrames for each stock symbol
    stock_data = {symbol: pd.DataFrame() for symbol in stock_symbols}
    
    for file in tqdm(csv_files, desc="Processing files"):
        if not os.path.isfile(file):  # Skip if not a file
            continue
        try:
            for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines="skip"):
                if chunk.empty:
                    continue
                if "Dissemination Identifier" not in chunk.columns:
                    chunk.rename(columns={
                        "Dissemintation ID": "Dissemination Identifier",
                        "Original Dissemintation ID": "Original Dissemination Identifier"
                    }, inplace=True)
                
                # Filter and append data for each stock symbol
                for symbol in stock_symbols:
                    if "Primary Asset Class" in chunk.columns or "Action Type" in chunk.columns:
                        symbol_data = chunk[chunk["Underlying Asset ID"].str.contains(f"{symbol}.", na=False)]
                    else:
                        symbol_data = chunk[chunk["Underlier ID-Leg 1"].str.contains(f"{symbol}.", na=False)]
                    
                    stock_data[symbol] = pd.concat([stock_data[symbol], symbol_data], ignore_index=True)
                
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
    
    # Save data for each stock symbol
    for symbol, data in stock_data.items():
        if not data.empty:
            # Treat "Original Dissemination Identifier" and "Dissemination Identifier" as long integers
            data["Original Dissemination Identifier"] = data["Original Dissemination Identifier"].astype("Int64")
            data["Dissemination Identifier"] = data["Dissemination Identifier"].astype("Int64")
            data = data.drop(columns=["Unnamed: 0"], errors="ignore")
            
            # Keep only specific columns
            columns_to_keep = ["Effective Date", "Notional amount-Leg 1", "Expiration Date", "Total notional quantity-Leg 1"]
            data = data[columns_to_keep]

            # Save to CSV
            output_file = os.path.join(COMPANIES_PATH, f"{symbol}.csv")
            data.to_csv(output_file, index=False)
            print(f"Saved data for {symbol} to {output_file}")


process_and_save_by_ticker()