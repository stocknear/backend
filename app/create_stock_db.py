import aiohttp
import asyncio
import sqlite3
import json
import ujson
import pandas as pd
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timezone
import warnings
from utils.helper import get_last_completed_quarter
import time
from typing import List, Dict, Set

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')

# Filter out the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")


start_date = datetime(2015, 1, 1).strftime("%Y-%m-%d")
end_date = datetime.today().strftime("%Y-%m-%d")
quarter, year = get_last_completed_quarter()


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
                f"https://financialmodelingprep.com/stable/dividends?symbol={symbol}&apikey={api_key}",
                f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_split/{symbol}?apikey={api_key}",
                f"https://financialmodelingprep.com/api/v4/stock_peers?symbol={symbol}&apikey={api_key}",
                f"https://financialmodelingprep.com/stable/institutional-ownership/extract-analytics/holder?symbol={symbol}&year={year}&quarter={quarter}&apikey={api_key}",
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
                        # ----- NEW: filter out stale quotes -----
                        if isinstance(parsed_data, list) and "quote" in url:
                            quote = parsed_data[0]
                            symbol = quote.get('symbol')
                            exchange = quote.get('exchange', None)

                            avg_volume = quote.get('avgVolume', 0)
                            quote_ts = quote.get("timestamp")
                            now_ts = datetime.now(timezone.utc).timestamp()

                            # If older than 5 days, delete symbol and stop processing
                            if exchange == 'OTC' and (avg_volume < 1000 or (quote_ts and (now_ts - quote_ts) > 10 * 24 * 3600)):
                                self.cursor.execute("DELETE FROM stocks WHERE symbol = ?", (symbol,))
                                self.cursor.execute(f"DROP TABLE IF EXISTS '{symbol}'")
                                self.conn.commit()
                                print(f"Deleting old outdated ticker {symbol}")
                                return

                        if isinstance(parsed_data, list) and "profile" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['profile'] = ujson.dumps(parsed_data)
                            data_dict = {
                                        'beta': parsed_data[0]['beta'],
                                        'country': parsed_data[0]['country'],
                                        'sector': parsed_data[0]['sector'],
                                        'industry': parsed_data[0]['industry'],
                                        }
                            fundamental_data.update(data_dict)

                        elif isinstance(parsed_data, list) and "quote" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['quote'] = ujson.dumps(parsed_data)
                            data_dict = {
                                        'price': parsed_data[0]['price'],
                                        'changesPercentage':parsed_data[0]['changesPercentage'],
                                        'marketCap': parsed_data[0]['marketCap'],
                                        'volume': parsed_data[0]['volume'],
                                        'avgVolume': parsed_data[0]['avgVolume'],
                                        'eps': parsed_data[0]['eps'],
                                        'pe': parsed_data[0]['pe'],
                                        }
                            fundamental_data.update(data_dict)
                       
                        elif "dividends" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['stock_dividend'] = ujson.dumps(parsed_data)
                        elif "stock_split" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['stock_split'] = ujson.dumps(parsed_data['historical'])
                        elif "stock_peers" in url:
                            # Handle list response, save as JSON object
                            fundamental_data['stock_peers'] = ujson.dumps([item for item in parsed_data[0]['peersList'] if item != ""])
                        elif "institutional-ownership" in url:
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
            try:
                symbol = stock.get('symbol',None)
                exchange_short_name = stock.get('exchangeShortName', '')
                ticker_type = stock.get('type', '')
                name = stock.get('name', '')
                exchange = stock.get('exchange', '')

                symbols.append(symbol)
                names.append(name)
                ticker_data.append((symbol, name, exchange, exchange_short_name, ticker_type))
            except Exception as e:
                print(e)
        

        self.cursor.execute("BEGIN TRANSACTION")  # Begin a transaction

        for data in ticker_data:
            try:
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
            except:
                pass

        self.conn.commit()

        # Save OHLC data for each ticker using aiohttp
        async with aiohttp.ClientSession() as session:
            tasks = []
            i = 0
            for stock_data in tqdm(ticker_data):
                try:
                    symbol, name, exchange, exchange_short_name, ticker_type = stock_data
                    #symbol = symbol.replace("-", "")  # Remove "-" from symbol
                    tasks.append(self.save_ohlc_data(session, symbol))
                    tasks.append(self.save_fundamental_data(session, symbol))

                    i += 1
                    if i % 60 == 0:
                        await asyncio.gather(*tasks)
                        tasks = []
                        print('sleeping')
                        await asyncio.sleep(60)  # Pause for 60 seconds
                except:
                    pass

            
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
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=bar&from={start_date}&to={end_date}&apikey={api_key}"
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







async def fetch_json(session: aiohttp.ClientSession, url: str) -> List[Dict]:
    """Generic async JSON fetcher with error handling."""
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientError as e:
        print(f"Error fetching {url}: {e}")
        return []

async def fetch_all_data(api_key: str) -> tuple[List[Dict], Set[str]]:
    """Fetch both all tickers and OTC tickers concurrently."""
    all_tickers_url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={api_key}"
    OTC_url = f"https://financialmodelingprep.com/stable/company-screener?exchange=OTC&marketCapMoreThan=10000000000&isETF=false&limit=5000&apikey={api_key}"
    
    async with aiohttp.ClientSession() as session:
        # Fetch both URLs concurrently
        all_tickers_task = fetch_json(session, all_tickers_url)
        OTC_task = fetch_json(session, OTC_url)
        
        all_tickers_data, OTC_data = await asyncio.gather(all_tickers_task, OTC_task)
    
    # Filter exchanges and symbols in one pass
    valid_exchanges = {'OTC', 'AMEX', 'NYSE', 'NASDAQ'}
    allowed_dash_symbols = {'BRK-A', 'BRK-B'}
    
    filtered_tickers = [
        item for item in all_tickers_data
        if (item.get('exchangeShortName') in valid_exchanges and
            ('-' not in item.get('symbol', '') or item.get('symbol') in allowed_dash_symbols))
    ]
    
    # Create OTC symbols set, excluding specific symbols
    excluded_OTC = {'VWAPY', 'VLKAF', 'VLKPF', 'DTEGF', 'RNMBF'}
    OTC_symbols = {item['symbol'] for item in OTC_data} - excluded_OTC
    
    return filtered_tickers, OTC_symbols

def filter_tickers(all_tickers: List[Dict], OTC_symbols: Set[str]) -> List[Dict]:
    """Filter tickers based on exchange and OTC inclusion rules."""
    filtered = []
    
    for ticker in all_tickers:
        try:
            exchange = ticker.get('exchangeShortName')
            symbol = ticker.get('symbol')
            asset_type = ticker.get('type',None)
            price = ticker.get('price',0)

            if asset_type == 'stock' and price > 0.5:
                if exchange == 'OTC':
                    # Only include OTC tickers that are in our filtered OTC list
                    if symbol in OTC_symbols:
                        filtered.append(ticker)
                else:
                    # Include all non-OTC tickers
                    filtered.append(ticker)
        except:
            pass
    
    return filtered

async def main():
    db = StockDatabase('backup_db/stocks.db')
    try:
        # Fetch all data concurrently
        all_tickers, OTC_symbols = await fetch_all_data(api_key)
        
        # Filter the tickers
        filtered_data = filter_tickers(all_tickers, OTC_symbols)
        
        # For testing - uncomment to limit results
        #test_symbols = {'BRK-A', 'BRK-B', 'AMD', 'NTDOY'}
        #filtered_data = [t for t in filtered_data if t.get('symbol') in test_symbols]
       
        await db.save_stocks(filtered_data)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        db.close_connection()

if __name__ == '__main__':
    asyncio.run(main())