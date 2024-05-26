import aiohttp
import asyncio
import sqlite3
import certifi
import json
import pandas as pd
from tqdm import tqdm
import re
import pandas as pd
from datetime import datetime
import subprocess
import time
import warnings
from dotenv import load_dotenv
import os

# Filter out the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

conn = sqlite3.connect('stocks.db') 
cursor = conn.cursor()

# Execute the SQL query
cursor.execute("SELECT symbol FROM stocks")

# Fetch all the results into a list
symbol_list = [row[0] for row in cursor.fetchall()]
conn.close()


load_dotenv()
api_key = os.getenv('FMP_API_KEY')
quarter_date = '2024-3-31'


if os.path.exists("backup_db/institute.db"):
    os.remove('backup_db/institute.db')


def get_jsonparsed_data(data):
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}


class InstituteDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode = wal")
        self.conn.commit()
        self._create_table()

    def close_connection(self):
        self.cursor.close()
        self.conn.close()

    def _create_table(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS institutes (
            cik TEXT PRIMARY KEY,
            name TEXT
        )
        """)


    def get_column_type(self, value):
        column_type = ""

        if isinstance(value, str):
            column_type = "TEXT"
        elif isinstance(value, int):
            column_type = "INTEGER"
        elif isinstance(value, float):
            column_type = "REAL"
        else:
            # Handle other data types or customize based on your specific needs
            column_type = "TEXT"

        return column_type

    def remove_null(self, value):
        if isinstance(value, str) and value == None:
            value = 'n/a'
        elif isinstance(value, int) and value == None:
            value = 0
        elif isinstance(value, float) and value == None:
            value = 0
        else:
            # Handle other data types or customize based on your specific needs
            pass

        return value






    async def save_portfolio_data(self, session, cik):
        try:
            urls = [
                f"https://financialmodelingprep.com/api/v4/institutional-ownership/industry/portfolio-holdings-summary?cik={cik}&date={quarter_date}&page=0&apikey={api_key}",
                f"https://financialmodelingprep.com/api/v4/institutional-ownership/portfolio-holdings?cik={cik}&date={quarter_date}&page=0&apikey={api_key}",
                f"https://financialmodelingprep.com/api/v4/institutional-ownership/portfolio-holdings-summary?cik={cik}&date={quarter_date}&page=0&apikey={api_key}"
            ]

            portfolio_data = {}

            for url in urls:
                async with session.get(url) as response:
                    data = await response.text()
                    parsed_data = get_jsonparsed_data(data)

                    try:
                        if isinstance(parsed_data, list) and "industry/portfolio-holdings-summary" in url:
                            # Handle list response, save as JSON object
                            portfolio_data['industry'] = json.dumps(parsed_data)
                        if isinstance(parsed_data, list) and "https://financialmodelingprep.com/api/v4/institutional-ownership/portfolio-holdings?cik=" in url:
                            # Handle list response, save as JSON object

                            parsed_data = [item for item in parsed_data if 'symbol' in item and item['symbol'] is not None and item['symbol'] in symbol_list] #symbol must be included in the database
                            portfolio_data['holdings'] = json.dumps(parsed_data)

                            
                            number_of_stocks = len(parsed_data)
                            total_market_value = sum(item['marketValue'] for item in parsed_data)
                            avg_performance_percentage = sum(item['performancePercentage'] for item in parsed_data) / len(parsed_data)
                            
                            performance_percentages = [item.get("performancePercentage", 0) for item in parsed_data]
                            positive_performance_count = sum(1 for percentage in performance_percentages if percentage > 0)
                            win_rate = round(positive_performance_count / len(performance_percentages) * 100,2)
                            data_dict = {
                                'winRate': win_rate,
                                'numberOfStocks': number_of_stocks,
                                'marketValue': total_market_value,
                                'avgPerformancePercentage': avg_performance_percentage,
                            }

                            portfolio_data.update(data_dict)

                        elif isinstance(parsed_data, list) and "https://financialmodelingprep.com/api/v4/institutional-ownership/portfolio-holdings-summary" in url:
                            # Handle list response, save as JSON object
                            data_dict = {
                                #'numberOfStocks': parsed_data[0]['portfolioSize'],
                                #'marketValue': parsed_data[0]['marketValue'],
                                'averageHoldingPeriod': parsed_data[0]['averageHoldingPeriod'],
                                'turnover': parsed_data[0]['turnover'],
                                #'performancePercentage': parsed_data[0]['performancePercentage']
                            }
                            portfolio_data.update(data_dict)



                    except:
                        pass

            # Check if columns already exist in the table
            self.cursor.execute("PRAGMA table_info(institutes)")
            columns = {column[1]: column[2] for column in self.cursor.fetchall()}
            
            holdings_list = json.loads(portfolio_data['holdings'])

            symbols_to_check = {holding['symbol'] for holding in holdings_list[:3]}  # Extract the first two symbols
            symbols_not_in_list = not any(symbol in symbol_list for symbol in symbols_to_check)


            if symbols_not_in_list or 'industry' not in portfolio_data or len(json.loads(portfolio_data['industry'])) == 0:
                # If 'industry' is not a list, delete the row and return
                #print(f"Deleting row for cik {cik} because 'industry' is not a list.")
                self.cursor.execute("DELETE FROM institutes WHERE cik = ?", (cik,))
                self.conn.commit()
                return

            # Update column definitions with keys from portfolio_data
            column_definitions = {
                key: (self.get_column_type(portfolio_data.get(key, None)), self.remove_null(portfolio_data.get(key, None)))
                for key in portfolio_data
            }

            for column, (column_type, value) in column_definitions.items():
                if column not in columns and column_type:
                    self.cursor.execute(f"ALTER TABLE institutes ADD COLUMN {column} {column_type}")

                self.cursor.execute(f"UPDATE institutes SET {column} = ? WHERE cik = ?", (value, cik))

            self.conn.commit()

        except Exception as e:
            print(f"Failed to fetch portfolio data for cik {cik}: {str(e)}")



    async def save_insitute(self, institutes):

        institute_data = []

        for item in institutes:
            cik = item.get('cik', '')
            name = item.get('name', '')


            institute_data.append((cik, name))
        

        self.cursor.execute("BEGIN TRANSACTION")  # Begin a transaction

        for data in institute_data:
            cik, name = data
            self.cursor.execute("""
            INSERT OR IGNORE INTO institutes (cik, name)
            VALUES (?, ?)
            """, (cik, name))
            self.cursor.execute("""
            UPDATE institutes SET name = ?
            WHERE cik = ?
            """, (name, cik))

        self.cursor.execute("COMMIT")  # Commit the transaction
        self.conn.commit()

       

        # Save OHLC data for each ticker using aiohttp
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            i = 0
            for item in tqdm(institute_data):
                cik, name = item
                tasks.append(self.save_portfolio_data(session, cik))

                i += 1
                if i % 700 == 0:
                    await asyncio.gather(*tasks)
                    tasks = []
                    print('sleeping mode: ', i)
                    await asyncio.sleep(60)  # Pause for 60 seconds

            
            if tasks:
                await asyncio.gather(*tasks)

        
url = f"https://financialmodelingprep.com/api/v4/institutional-ownership/list?apikey={api_key}"


async def fetch_tickers():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.text()
            return get_jsonparsed_data(data)


db = InstituteDatabase('backup_db/institute.db')
loop = asyncio.get_event_loop()
all_tickers = loop.run_until_complete(fetch_tickers())
loop.run_until_complete(db.save_insitute(all_tickers))
db.close_connection()