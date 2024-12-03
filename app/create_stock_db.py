import aiohttp
import asyncio
import sqlite3
import json
import ujson
import pandas as pd
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from ta.utils import *
from ta.volatility import *
from ta.momentum import *
from ta.trend import *
from ta.volume import *
import warnings

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')

# Filter out the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")


start_date = datetime(2015, 1, 1).strftime("%Y-%m-%d")
end_date = datetime.today().strftime("%Y-%m-%d")

quarter_date = '2024-06-30'


if os.path.exists("backup_db/stocks.db"):
    os.remove('backup_db/stocks.db')


def get_jsonparsed_data(data):
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}


class StockDatabase:
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
        CREATE TABLE IF NOT EXISTS stocks (
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


    async def save_fundamental_data(self, session, symbol):
        try:
            urls = [
                f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}",
                f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}",
                f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{symbol}?limit=400&apikey={api_key}",
                f"https://financialmodelingprep.com/api/v4/historical/employee_count?symbol={symbol}&apikey={api_key}",
                f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_split/{symbol}?apikey={api_key}",
                f"https://financialmodelingprep.com/api/v4/stock_peers?symbol={symbol}&apikey={api_key}",
                f"https://financialmodelingprep.com/api/v4/institutional-ownership/institutional-holders/symbol-ownership-percent?date={quarter_date}&symbol={symbol}&page=0&apikey={api_key}",
                f"https://financialmodelingprep.com/api/v4/historical/shares_float?symbol={symbol}&apikey={api_key}",
                f"https://financialmodelingprep.com/api/v4/revenue-product-segmentation?symbol={symbol}&structure=flat&period=annual&apikey={api_key}",
                f"https://financialmodelingprep.com/api/v4/revenue-geographic-segmentation?symbol={symbol}&structure=flat&apikey={api_key}",
                f"https://financialmodelingprep.com/api/v3/analyst-estimates/{symbol}?apikey={api_key}",
            ]

            fundamental_data = {}


            for url in urls:

                async with session.get(url) as response:
                    data = await response.text()
                    parsed_data = get_jsonparsed_data(data)

                    try:
                        if isinstance(parsed_data, list) and "profile" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['profile'] = ujson.dumps(parsed_data)
                            data_dict = {
                                        'beta': parsed_data[0]['beta'],
                                        'country': parsed_data[0]['country'],
                                        'sector': parsed_data[0]['sector'],
                                        'industry': parsed_data[0]['industry'],
                                        'discounted_cash_flow': round(parsed_data[0]['dcf'],2),
                                        }
                            fundamental_data.update(data_dict)

                        elif isinstance(parsed_data, list) and "quote" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['quote'] = ujson.dumps(parsed_data)
                            data_dict = {
                                        'price': parsed_data[0]['price'],
                                        'changesPercentage': round(parsed_data[0]['changesPercentage'],2),
                                        'marketCap': parsed_data[0]['marketCap'],
                                        'volume': parsed_data[0]['volume'],
                                        'avgVolume': parsed_data[0]['avgVolume'],
                                        'eps': parsed_data[0]['eps'],
                                        'pe': parsed_data[0]['pe'],
                                        }
                            fundamental_data.update(data_dict)

                        elif isinstance(parsed_data, list) and "sector-benchmark" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['esg_sector_benchmark'] = ujson.dumps(parsed_data)

                            fundamental_data.update(data_dict)
                       
                        elif "stock_dividend" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['stock_dividend'] = ujson.dumps(parsed_data)
                        elif "employee_count" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['history_employee_count'] = ujson.dumps(parsed_data)
                        elif "stock_split" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['stock_split'] = ujson.dumps(parsed_data['historical'])
                        elif "stock_peers" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['stock_peers'] = ujson.dumps([item for item in parsed_data[0]['peersList'] if item != ""])
                        elif "institutional-ownership/institutional-holders" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['shareholders'] = ujson.dumps(parsed_data)
                        elif "historical/shares_float" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['historicalShares'] = ujson.dumps(parsed_data)
                        elif "revenue-product-segmentation" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['revenue_product_segmentation'] = ujson.dumps(parsed_data)
                        elif "revenue-geographic-segmentation" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['revenue_geographic_segmentation'] = ujson.dumps(parsed_data)
                        elif "analyst-estimates" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['analyst_estimates'] = ujson.dumps(parsed_data)
                    except Exception as e:
                        print(e)
                        pass


            # Check if columns already exist in the table
            self.cursor.execute("PRAGMA table_info(stocks)")
            columns = {column[1]: column[2] for column in self.cursor.fetchall()}

            # Update column definitions with keys from fundamental_data
            column_definitions = {
                key: (self.get_column_type(fundamental_data.get(key, None)), self.remove_null(fundamental_data.get(key, None)))
                for key in fundamental_data
            }


            for column, (column_type, value) in column_definitions.items():
                if column not in columns and column_type:
                    self.cursor.execute(f"ALTER TABLE stocks ADD COLUMN {column} {column_type}")

                self.cursor.execute(f"UPDATE stocks SET {column} = ? WHERE symbol = ?", (value, symbol))

            self.conn.commit()

        except Exception as e:
            print(f"Failed to fetch fundamental data for symbol {symbol}: {str(e)}")


    async def save_stocks(self, stocks):
        symbols = []
        names = []
        ticker_data = []

        for stock in stocks:
            exchange_short_name = stock.get('exchangeShortName', '')
            ticker_type = stock.get('type', '')
            if exchange_short_name in ['XETRA','NYSE', 'NASDAQ','AMEX', 'PNK','EURONEXT'] and ticker_type in ['stock']:
                symbol = stock.get('symbol', '')
                if exchange_short_name == 'PNK' and symbol not in ['TSSI','DRSHF','NTDOY','OTGLF','TCEHY', 'KRKNF','BYDDY','XIACY','NSRGY','TLPFY','TLPFF']:
                    pass
                elif exchange_short_name == 'EURONEXT' and symbol not in ['ALEUP.PA','ALNEV.PA','ALGAU.PA','ALDRV.PA','ALHYG.PA','ALVMG.PA','TEP.PA']:
                    pass
                else:
                    name = stock.get('name', '')
                    exchange = stock.get('exchange', '')

                    #if name and '-' not in symbol:
                    if name:
                        symbols.append(symbol)
                        names.append(name)

                        ticker_data.append((symbol, name, exchange, exchange_short_name, ticker_type))
        

        self.cursor.execute("BEGIN TRANSACTION")  # Begin a transaction

        for data in ticker_data:
            symbol, name, exchange, exchange_short_name, ticker_type = data

            # Check if the symbol already exists
            self.cursor.execute("SELECT symbol FROM stocks WHERE symbol = ?", (symbol,))
            exists = self.cursor.fetchone()

            # If it doesn't exist, insert it
            if not exists:
                self.cursor.execute("""
                INSERT INTO stocks (symbol, name, exchange, exchangeShortName, type)
                VALUES (?, ?, ?, ?, ?)
                """, (symbol, name, exchange, exchange_short_name, ticker_type))

            # Update the existing row
            else:
                self.cursor.execute("""
                UPDATE stocks SET name = ?, exchange = ?, exchangeShortName = ?, type = ?
                WHERE symbol = ?
                """, (name, exchange, exchange_short_name, ticker_type, symbol))

        self.conn.commit()

        # Save OHLC data for each ticker using aiohttp
        async with aiohttp.ClientSession() as session:
            tasks = []
            i = 0
            for stock_data in tqdm(ticker_data):
                symbol, name, exchange, exchange_short_name, ticker_type = stock_data
                #symbol = symbol.replace("-", "")  # Remove "-" from symbol
                tasks.append(self.save_ohlc_data(session, symbol))
                tasks.append(self.save_fundamental_data(session, symbol))

                i += 1
                if i % 60 == 0:
                    await asyncio.gather(*tasks)
                    tasks = []
                    print('sleeping mode 30 seconds')
                    await asyncio.sleep(30)  # Pause for 60 seconds

            
            if tasks:
                await asyncio.gather(*tasks)


    def _create_ticker_table(self, symbol):
        cleaned_symbol = symbol  # Ensure this is a safe string to use as a table name
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS '{cleaned_symbol}' (
                date TEXT UNIQUE,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume INT,
                change_percent FLOAT
            );
        """)
        self.conn.commit()

    async def save_ohlc_data(self, session, symbol):
        try:
            self._create_ticker_table(symbol)  # Ensure the table exists

            # Fetch OHLC data from the API
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=bar&from={start_date}&apikey={api_key}"
            async with session.get(url) as response:
                data = await response.text()
            
            ohlc_data = get_jsonparsed_data(data)
            if 'historical' in ohlc_data:
                historical_data = ohlc_data['historical'][::-1]

                for entry in historical_data:
                    # Prepare the data for each entry
                    date = entry.get('date')
                    open_price = entry.get('open')
                    high = entry.get('high')
                    low = entry.get('low')
                    close = entry.get('close')
                    volume = entry.get('volume')
                    change_percent = entry.get('changePercent')

                    # Check if this date's data already exists
                    self.cursor.execute(f"SELECT date FROM '{symbol}' WHERE date = ?", (date,))
                    exists = self.cursor.fetchone()

                    # If it doesn't exist, insert the new data
                    if not exists:
                        self.cursor.execute(f"""
                            INSERT INTO '{symbol}' (date, open, high, low, close, volume, change_percent)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (date, open_price, high, low, close, volume, change_percent))
                
                # Commit all changes to the database
                self.conn.commit()

        except Exception as e:
            print(f"Failed to fetch or insert OHLC data for symbol {symbol}: {str(e)}")



url = f"https://financialmodelingprep.com/api/v3/available-traded/list?apikey={api_key}"


async def fetch_tickers():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.text()
            return get_jsonparsed_data(data)


db = StockDatabase('backup_db/stocks.db')
loop = asyncio.get_event_loop()
all_tickers = loop.run_until_complete(fetch_tickers())
#all_tickers = [item for item in all_tickers if item['symbol'] == 'ZJK']
'''
existing_names = set()

filtered_data = []
for item in all_tickers:
    if '.' not in item['symbol'] and item['name'] not in existing_names:
        filtered_data.append(item)
        existing_names.add(item['name'])

print(len(filtered_data))

for item in filtered_data:
    if 'RHM.DE' in item['symbol']:
        print(item)

time.sleep(1000)
'''

loop.run_until_complete(db.save_stocks(all_tickers))
db.close_connection()