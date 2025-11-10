import pandas as pd
import numpy as np
import glob
import requests
import os
import sqlite3
import ujson
from zipfile import ZipFile
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta
import shutil

# Define configuration variables
OUTPUT_PATH = "./json/swap"
COMPANIES_PATH = "./json/swap/companies"
MAX_WORKERS = 4
CHUNK_SIZE = 5000  # Adjust based on system RAM
DAYS_TO_PROCESS = 360

# Ensure directories exist
# Remove the directory
shutil.rmtree('json/swap/companies')
os.makedirs(COMPANIES_PATH, exist_ok=True)


def get_stock_symbols():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 1E9 AND symbol NOT LIKE '%.%'")
        total_symbols = [row[0] for row in cursor.fetchall()]
        return total_symbols

stock_symbols = get_stock_symbols()


# Function to clean and convert to numeric values
def clean_and_convert(series):
    return pd.to_numeric(series.replace({',': ''}, regex=True).str.extract(r'(\d+)', expand=False), errors='coerce').fillna(0).astype(int)


def generate_filenames():
    end = datetime.today()
    start = end - timedelta(days=DAYS_TO_PROCESS)
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return [f"SEC_CUMULATIVE_EQUITIES_{date.strftime('%Y_%m_%d')}.zip" for date in dates]

def download_and_process(filename):
    csv_output_filename = os.path.join(OUTPUT_PATH, filename.replace('.zip', '.csv'))
    if os.path.exists(csv_output_filename):
        print(f"{csv_output_filename} already exists. Skipping.")
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
    
    output_filename = os.path.join(OUTPUT_PATH, csv_filename)
    
    columns_to_keep = [
        "Underlying Asset ID", "Underlier ID-Leg 1",
        "Effective Date", "Notional amount-Leg 1",
        "Expiration Date", "Total notional quantity-Leg 1",
        "Dissemination Identifier", "Original Dissemination Identifier",
        "Dissemintation ID", "Original Dissemintation ID",
        "Primary Asset Class", "Action Type"
    ]
    
    chunk_list = []
    for chunk in pd.read_csv(csv_filename, chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines="skip", usecols=lambda x: x in columns_to_keep):
        # Rename columns if necessary
        if "Dissemination Identifier" not in chunk.columns:
            chunk.rename(columns={
                "Dissemintation ID": "Dissemination Identifier",
                "Original Dissemintation ID": "Original Dissemination Identifier"
            }, inplace=True)
        
        chunk_list.append(chunk)
    
    pd.concat(chunk_list, ignore_index=True).to_csv(output_filename, index=False)
    
    os.remove(filename)
    os.remove(csv_filename)
    
    print(f"Processed and saved {output_filename}")


def process_and_save_by_ticker():
    csv_files = glob.glob(os.path.join(OUTPUT_PATH, "*.csv"))
    
    # Sort CSV files by date (assuming filename format is "SEC_CUMULATIVE_EQUITIES_YYYY_MM_DD.csv")
    sorted_csv_files = sorted(csv_files, key=lambda x: datetime.strptime("_".join(os.path.splitext(os.path.basename(x))[0].split('_')[3:]), "%Y_%m_%d"), reverse=True)
    
    # Select only the N latest files
    latest_csv_files = sorted_csv_files[:100]

    # Create a set of stock symbols for faster lookup
    stock_symbols_set = set(stock_symbols)
    
    for file in tqdm(latest_csv_files, desc="Processing files"):
        if not os.path.isfile(file):  # Skip if not a file
            continue
        try:
            # Read the CSV file in chunks
            for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines="skip"):
                if chunk.empty:
                    continue
                
                # Rename columns if necessary
                if "Dissemination Identifier" not in chunk.columns:
                    chunk.rename(columns={
                        "Dissemintation ID": "Dissemination Identifier",
                        "Original Dissemintation ID": "Original Dissemination Identifier"
                    }, inplace=True)
                
                # Determine which column to use for filtering
                filter_column = "Underlying Asset ID" if "Primary Asset Class" in chunk.columns or "Action Type" in chunk.columns else "Underlier ID-Leg 1"
                
                # Extract the symbol from the filter column
                chunk['symbol'] = chunk[filter_column].str.split('.').str[0]
                
                # Filter the chunk to include only rows with symbols in our list
                filtered_chunk = chunk[chunk['symbol'].isin(stock_symbols_set)]
                
                # If the filtered chunk is not empty, process and save it
                if not filtered_chunk.empty:
                    columns_to_keep = ["symbol", "Effective Date", "Notional amount-Leg 1", "Expiration Date", "Total notional quantity-Leg 1"]
                    filtered_chunk = filtered_chunk[columns_to_keep]
                    
                    # Convert 'Notional amount-Leg 1' and 'Total notional quantity-Leg 1' to integers
                    filtered_chunk['Notional amount-Leg 1'] = clean_and_convert(filtered_chunk['Notional amount-Leg 1'])
                    filtered_chunk['Total notional quantity-Leg 1'] = clean_and_convert(filtered_chunk['Total notional quantity-Leg 1'])

                    # Group by symbol and append to respective files
                    for symbol, group in filtered_chunk.groupby('symbol'):
                        output_file = os.path.join(COMPANIES_PATH, f"{symbol}.json")
                        group = group.drop(columns=['symbol'])
                        
                        # Convert DataFrame to list of dictionaries
                        records = group.to_dict('records')
                        
                        if os.path.exists(output_file):
                            with open(output_file, 'r+') as f:
                                data = ujson.load(f)
                                data.extend(records)
                                f.seek(0)
                                ujson.dump(data, f)
                        else:
                            with open(output_file, 'w') as f:
                                ujson.dump(records, f)
        
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
    
    # Final processing of each symbol's file
    for symbol in tqdm(stock_symbols, desc="Final processing"):
        file_path = os.path.join(COMPANIES_PATH, f"{symbol}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = ujson.load(f)
                
                # Convert to DataFrame for processing
                df = pd.DataFrame(data)
                df["Original Dissemination Identifier"] = df["Original Dissemination Identifier"].astype("Int64")
                df["Dissemination Identifier"] = df["Dissemination Identifier"].astype("Int64")
                
                # Convert back to list of dictionaries and save
                processed_data = df.to_dict('records')
                with open(file_path, 'w') as f:
                    ujson.dump(processed_data, f)
                
                print(f"Processed and saved data for {symbol}")
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")


if __name__ == "__main__":
    filenames = generate_filenames()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(download_and_process, filenames), total=len(filenames)))
    process_and_save_by_ticker()