import aiohttp
import asyncio
import sqlite3
import json
import ujson
import pandas as pd
import os
from tqdm import tqdm
import re
import pandas as pd
from datetime import datetime

import warnings

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')


# Filter out the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")


start_date = datetime(2015, 1, 1).strftime("%Y-%m-%d")
end_date = datetime.today().strftime("%Y-%m-%d")



if os.path.exists("backup_db/index.db"):
    os.remove('backup_db/index.db')


def get_jsonparsed_data(data):
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}


class IndexDatabase:
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
        CREATE TABLE IF NOT EXISTS indices (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            exchange TEXT,
            exchangeShortName TEXT,
            type TEXT
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

    def delete_data_if_condition(self, condition, symbol):
        # Get a list of all tables in the database
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in self.cursor.fetchall()]

        for table in tables:
            # Check if the table name is not 'indices' (the main table)
            if table != 'indices':
                # Construct a DELETE query to delete data from the table based on the condition
                delete_query = f"DELETE FROM {table} WHERE {condition}"

                # Execute the DELETE query with the symbol as a parameter
                self.cursor.execute(delete_query, (symbol,))
                self.conn.commit()


    async def save_index(self, indices):
        symbols = []
        names = []
        ticker_data = []

        for index in indices:
            exchange_short_name = index.get('exchangeShortName', '')
            ticker_type = 'INDEX'
            symbol = index.get('symbol', '')
            name = index.get('name', '')
            exchange = index.get('exchange', '')

            symbols.append(symbol)
            names.append(name)
            ticker_data.append((symbol, name, exchange, exchange_short_name, ticker_type))
        
        self.cursor.execute("BEGIN TRANSACTION")  # Begin a transaction

        for data in ticker_data:
            symbol, name, exchange, exchange_short_name, ticker_type = data
            self.cursor.execute("""
            INSERT OR IGNORE INTO indices (symbol, name, exchange, exchangeShortName, type)
            VALUES (?, ?, ?, ?, ?)
            """, (symbol, name, exchange, exchange_short_name, ticker_type))
            self.cursor.execute("""
            UPDATE indices SET name = ?, exchange = ?, exchangeShortName = ?, type = ?
            WHERE symbol = ?
            """, (name, exchange, exchange_short_name, ticker_type, symbol))

        self.cursor.execute("COMMIT")  # Commit the transaction
        self.conn.commit()

    

        # Save OHLC data for each ticker using aiohttp
        async with aiohttp.ClientSession() as session:
            tasks = []
            i = 0
            for index_data in tqdm(ticker_data):
                symbol, name, exchange, exchange_short_name, ticker_type = index_data
                symbol = symbol.replace("-", "")
                tasks.append(self.save_ohlc_data(session, symbol))

                i += 1
                if i % 150 == 0:
                    await asyncio.gather(*tasks)
                    tasks = []
                    print('sleeping mode: ', i)
                    await asyncio.sleep(60)  # Pause for 60 seconds

            
            if tasks:
                await asyncio.gather(*tasks)


    def _create_ticker_table(self, symbol):
        cleaned_symbol = symbol
        # Check if table exists
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{symbol}'")
        table_exists = self.cursor.fetchone() is not None

        if not table_exists:
            query = f"""
            CREATE TABLE '{cleaned_symbol}' (
                date TEXT,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume INT,
                change_percent FLOAT,
            );
            """
            self.cursor.execute(query)

    async def save_ohlc_data(self, session, symbol):
        try:
            #self._create_ticker_table(symbol)  # Create table for the symbol

            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=bar&from={start_date}&to={end_date}&apikey={api_key}"

            try:
                async with session.get(url) as response:
                    data = await response.text()

                ohlc_data = get_jsonparsed_data(data)
                if 'historical' in ohlc_data:
                    ohlc_values = [(item['date'], item['open'], item['high'], item['low'], item['close'], item['volume'], item['changePercent']) for item in ohlc_data['historical'][::-1]]

                    df = pd.DataFrame(ohlc_values, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'change_percent'])
                
                    # Perform bulk insert
                    df.to_sql(symbol, self.conn, if_exists='append', index=False)

            except Exception as e:
                print(f"Failed to fetch OHLC data for symbol {symbol}: {str(e)}")
        except Exception as e:
            print(f"Failed to create table for symbol {symbol}: {str(e)}")


url = f"https://financialmodelingprep.com/stable/index-list?apikey={api_key}"


async def fetch_tickers():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.text()
            return get_jsonparsed_data(data)


db = IndexDatabase('backup_db/index.db')
loop = asyncio.get_event_loop()
all_tickers = loop.run_until_complete(fetch_tickers())

all_tickers = [
    item for item in all_tickers
    if item['currency'] == 'USD' and '.' not in item['symbol'] and '-' not in item['symbol']
]

print(len(all_tickers))

try:
    loop.run_until_complete(db.save_index(all_tickers))
except Exception as e:
    print(f"Error saving index: {str(e)}")
finally:
    db.close_connection()