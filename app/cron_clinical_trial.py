from pytrials.client import ClinicalTrials
import ujson
import asyncio
import aiohttp
import sqlite3
from tqdm import tqdm
from datetime import datetime,timedelta
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import time


ct = ClinicalTrials()

async def get_data(company_name):
    try:
    	get_ct_data = ct.get_study_fields(
		    search_expr=f"{company_name}",
		    fields=["Study Results","Funder Type","Start Date", "Completion Date","Study Status","Study Title", 'Phases', 'Brief Summary', 'Age','Sex', 'Enrollment','Study Type','Sponsor','Study URL','NCT Number'],
		    max_studies=1000,
		    )
    	df = pd.DataFrame.from_records(get_ct_data[1:], columns=get_ct_data[0])
    	df['Completion Date'] = pd.to_datetime(df['Completion Date'],errors='coerce')
    	df_sorted = df.sort_values(by='Completion Date', ascending=False)
    	# Convert 'Completion Date' back to string format
    	df_sorted['Completion Date'] = df_sorted['Completion Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)
    	df_sorted['Phases'] = df_sorted['Phases'].replace('PHASE2|PHASE3', 'Phase 2/3')
    	df_sorted['Phases'] = df_sorted['Phases'].replace('PHASE1|PHASE2', 'Phase 1/2')
    	df_sorted['Phases'] = df_sorted['Phases'].replace('EARLY_PHASE1', 'Phase 1')

    	df_sorted['Study Status'] = df_sorted['Study Status'].replace('ACTIVE_NOT_RECRUITING', 'Active')
    	df_sorted['Study Status'] = df_sorted['Study Status'].replace('NOT_YET_RECRUITING', 'Active')
    	df_sorted['Study Status'] = df_sorted['Study Status'].replace('UNKNOWN', '-')

    	data = df_sorted.to_dict('records')
    	return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return []

async def save_json(symbol, data):
    # Use async file writing to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    path = f"json/clinical-trial/companies/{symbol}.json"
    await loop.run_in_executor(None, ujson.dump, data, open(path, 'w'))

async def process_ticker(symbol, name):
    data = await get_data(name)
    if len(data)>0:
        await save_json(symbol, data)

async def run():
    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol, name FROM stocks WHERE industry = 'Biotechnology' AND symbol NOT LIKE '%.%'")
    company_data = [{'symbol': row[0], 'name': row[1]} for row in cursor.fetchall()]
    con.close()
    #test mode
    #company_data = [{'symbol': 'DSGN', 'name': 'Design Therapeutics, Inc.'}]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for item in company_data:
            tasks.append(process_ticker(item['symbol'], item['name']))
        
        # Run tasks concurrently in batches to avoid too many open connections
        batch_size = 10  # Adjust based on your system's capacity
        for i in tqdm(range(0, len(tasks), batch_size)):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(f"An error occurred: {e}")