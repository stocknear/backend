import orjson
import sqlite3
import asyncio
import aiohttp
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')


# Load stock screener data
with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}


query_etf_holding = f"SELECT holding from etfs WHERE symbol = ?"
quote_cache = {}

async def save_json(category, data, category_type='market-cap'):
    with open(f"json/{category_type}/list/{category}.json", 'wb') as file:
        file.write(orjson.dumps(data))

async def get_quote_data(symbol):
    """Get quote data for a symbol from JSON file"""
    if symbol in quote_cache:
        return quote_cache[symbol]
    else:
        try:
            with open(f"json/quote/{symbol}.json") as file:
                quote_data = orjson.loads(file.read())
                quote_cache[symbol] = quote_data  # Cache the loaded data
                return quote_data
        except:
            return None

async def process_category(cursor, category, condition, category_type='market-cap'):
    base_query = """
        SELECT DISTINCT s.symbol, s.name, s.exchangeShortName, s.marketCap, s.sector
        FROM stocks s 
        WHERE {}
    """
    
    full_query = base_query.format(condition)
    cursor.execute(full_query)
    raw_data = cursor.fetchall()
    
    res_list = []
    for row in raw_data:
        symbol = row[0]
        quote_data = await get_quote_data(symbol)
        if quote_data:
            item = {
                'symbol': symbol,
                'name': row[1],
                'price': round(quote_data.get('price'), 2) if quote_data.get('price') is not None else None,
                'changesPercentage': round(quote_data.get('changesPercentage'), 2) if quote_data.get('changesPercentage') is not None else None,
                'marketCap': quote_data.get('marketCap', None),
                'revenue': None,
            }
            
            # Add screener data if available
            if symbol in stock_screener_data_dict:
                item['revenue'] = stock_screener_data_dict[symbol].get('revenue')
            
            if item['marketCap'] > 0:
                res_list.append(item)
    
    # Sort by market cap and save
    sorted_result = sorted(res_list, key=lambda x: x['marketCap'] if x['marketCap'] else 0, reverse=True)
    # Add rank to each item
    for rank, item in enumerate(sorted_result, 1):
        item['rank'] = rank

    await save_json(category, sorted_result, category_type)
    print(f"Processed and saved {len(sorted_result)} stocks for {category}")
    return sorted_result


async def get_etf_holding(etf_symbols, etf_con):
    etf_symbols = ['AGG']

    for ticker in tqdm(etf_symbols):
        res = []
        df = pd.read_sql_query(query_etf_holding, etf_con, params=(ticker,))
        try:
            # Load holdings data from the SQL query result
            data = orjson.loads(df['holding'].iloc[0])
            last_update = data[0]['updated'][0:10]
            # Rename 'asset' to 'symbol' and keep other keys the same
            res = [
                {
                    'symbol': item.get('asset', None),
                    'name': item.get('name', None).capitalize() if item.get('name') else None,
                    'weightPercentage': item.get('weightPercentage', None),
                    'sharesNumber': item.get('marketValue', None) if not item.get('asset') and item.get('sharesNumber') == 0 else item.get('sharesNumber', None)
                }
                for item in data
                if item.get('marketValue', 0) >= 0  # Exclude items with a negative marketValue
            ]

            for item in res:
                try:
                    symbol = item['symbol']
                    
                    # Check if the symbol data is already in the cache
                    if symbol in quote_cache:
                        quote_data = quote_cache[symbol]
                    else:
                        # Load the quote data from file if not in cache
                        try:
                            with open(f"json/quote/{symbol}.json") as file:
                                quote_data = orjson.loads(file.read())
                                quote_cache[symbol] = quote_data  # Cache the loaded data
                                item['price'] = round(quote_data.get('price'), 2) if quote_data else None
                                item['changesPercentage'] = round(quote_data.get('changesPercentage'), 2) if quote_data else None
                                item['name'] = quote_data.get('name') if quote_data else None
                        except:
                            quote_data = None
                except:
                    pass

                # Assign price and changesPercentage if available, otherwise set to None
                item['weightPercentage'] = round(item.get('weightPercentage'), 2) if item['weightPercentage'] else None

        except Exception as e:
            last_update = None
            res = []
        # Save results to a file if there's data to write
        if res:
            for rank, item in enumerate(res, 1):
                item['rank'] = rank
            with open(f"json/etf/holding/{ticker}.json", 'wb') as file:
                final_res = {'lastUpdate': last_update, 'holdings': res}
                file.write(orjson.dumps(final_res))


async def get_etf_provider(etf_con):

    cursor = etf_con.cursor()
    cursor.execute("SELECT DISTINCT etfProvider FROM etfs")
    etf_provider = [row[0] for row in cursor.fetchall()]
    query = "SELECT symbol, name, expenseRatio, totalAssets, numberOfHoldings FROM etfs WHERE etfProvider = ?"
    
    for provider in etf_provider:
        try:
            cursor.execute(query, (provider,))
            raw_data = cursor.fetchall()
            # Extract only relevant data and sort it
            # Extract only relevant data and filter only integer totalAssets
            res = [
                {'symbol': row[0], 'name': row[1], 'expenseRatio': row[2], 'totalAssets': row[3], 'numberOfHoldings': row[4]}
                for row in raw_data if isinstance(row[3], float) or isinstance(row[3], int)
            ]
            for item in res:
                try:
                    symbol = item['symbol']
                    with open(f"json/quote/{symbol}.json") as file:
                        quote_data = orjson.loads(file.read())
                    # Assign price and changesPercentage if available, otherwise set to None
                    item['price'] = round(quote_data.get('price'), 2) if quote_data else None
                    item['changesPercentage'] = round(quote_data.get('changesPercentage'), 2) if quote_data else None
                    item['name'] = quote_data.get('name') if quote_data else None
                except:
                    pass

            sorted_res = sorted(res, key=lambda x: x['totalAssets'], reverse=True)


            # Save results to a file if there's data to write
            if sorted_res:
                with open(f"json/etf/provider/{provider}.json", 'wb') as file:
                    file.write(orjson.dumps(sorted_res))
        except Exception as e:
            print(e)
            pass
    cursor.close()


async def get_magnificent_seven():
  
    symbol_list = ['MSFT','AAPL','GOOGL','AMZN','NVDA','META','TSLA']
   

    res_list = []
    for symbol in symbol_list:
        try:
            revenue = stock_screener_data_dict[symbol].get('revenue',None)

            try:
                with open(f"json/quote/{symbol}.json") as file:
                        quote_data = orjson.loads(file.read())
            except:
                quote_data = None

            # Assign price and changesPercentage if available, otherwise set to None
            price = round(quote_data.get('price'), 2) if quote_data else None
            changesPercentage = round(quote_data.get('changesPercentage'), 2) if quote_data else None
            marketCap = quote_data.get('marketCap') if quote_data else None
            name = quote_data.get('name') if quote_data else None

            res_list.append({'symbol': symbol, 'name': name, 'price': price, \
                    'changesPercentage': changesPercentage, 'marketCap': marketCap, \
                    'revenue': revenue})

        except Exception as e:
            print(e)

    if res_list:
        res_list = sorted(res_list, key=lambda x: x['marketCap'], reverse=True)
        for rank, item in enumerate(res_list, start=1):
                item['rank'] = rank
                
        with open(f"json/magnificent-seven/data.json", 'wb') as file:
            file.write(orjson.dumps(res_list))


async def etf_bitcoin_list():
    try:
        with sqlite3.connect('etf.db') as etf_con:
            etf_cursor = etf_con.cursor()
            etf_cursor.execute("PRAGMA journal_mode = wal")
            etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
            etf_symbols = [row[0] for row in etf_cursor.fetchall()]

            res_list = []
            query_template = """
                SELECT 
                    symbol, name, expenseRatio, totalAssets
                FROM 
                    etfs
                WHERE
                    symbol = ?
            """
            
            for symbol in etf_symbols:
                try:
                    data = pd.read_sql_query(query_template, etf_con, params=(symbol,))
                    name = data['name'].iloc[0]
                    
                    if 'bitcoin' in name.lower():
                        expense_ratio = round(float(data['expenseRatio'].iloc[0]), 2)
                        total_assets = int(data['totalAssets'].iloc[0])
                        
                        try:
                            with open(f"json/quote/{symbol}.json", "rb") as file:
                                quote_data = orjson.loads(file.read())
                        except (FileNotFoundError, orjson.JSONDecodeError):
                            quote_data = None

                        price = round(quote_data.get('price'), 2) if quote_data else None
                        changesPercentage = round(quote_data.get('changesPercentage'), 2) if quote_data else None
                        if total_assets > 0:
                            res_list.append({
                                'symbol': symbol,
                                'name': name,
                                'expenseRatio': expense_ratio,
                                'totalAssets': total_assets,
                                'price': price,
                                'changesPercentage': changesPercentage
                            })
                except Exception as e:
                    print(f"Error processing symbol {symbol}: {e}")
            
            if res_list:
                res_list = sorted(res_list, key=lambda x: x['totalAssets'], reverse=True)
                for rank, item in enumerate(res_list, start=1):
                    item['rank'] = rank
                    
                with open("json/etf-bitcoin-list/data.json", 'wb') as file:
                    file.write(orjson.dumps(res_list))

    except Exception as e:
        print(f"Database error: {e}")

async def get_all_reits_list(cursor):
    base_query = """
        SELECT DISTINCT s.symbol, s.name, s.exchangeShortName, s.marketCap, s.sector
        FROM stocks s 
        WHERE {}
    """
    
    # Use the specific condition within the dictionary
    condition = "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND industry LIKE '%REIT%' AND symbol NOT LIKE '%-%'"
    full_query = base_query.format(condition)
    
    # Execute the query and fetch all rows
    cursor.execute(full_query)  # Assuming cursor is async
    raw_data = cursor.fetchall()
    
    res_list = []
    for row in raw_data:
        symbol = row[0]
        
        # Fetch quote data asynchronously
        try:
            quote_data = await get_quote_data(symbol)
        except Exception as e:
            print(f"Error fetching quote data for {symbol}: {e}")
            continue
        
        if quote_data:
            item = {
                'symbol': symbol,
                'name': row[1],
                'price': round(quote_data.get('price', 0), 2),
                'changesPercentage': round(quote_data.get('changesPercentage', 0), 2),
                'marketCap': quote_data.get('marketCap', 0),
            }
            
            # Get dividend yield if available
            item['dividendYield'] = stock_screener_data_dict.get(symbol, {}).get('dividendYield', None)
            
            # Append item if conditions are met
            if item['marketCap'] > 0 and item['dividendYield'] is not None:
                res_list.append(item)
    
    if res_list:
        res_list = sorted(res_list, key=lambda x: x['marketCap'] or 0, reverse=True)
        
        # Add rank to each item
        for rank, item in enumerate(res_list, 1):
            item['rank'] = rank

        with open("json/industry/list/reits.json", 'wb') as file:
            file.write(orjson.dumps(res_list))


async def get_index_list():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks")
        symbols = [row[0] for row in cursor.fetchall()]

    async with aiohttp.ClientSession() as session:

        for index_list in ['nasdaq','dowjones','sp500']:
            url = f"https://financialmodelingprep.com/api/v3/{index_list}_constituent?apikey={api_key}"
            async with session.get(url) as response:
                data = await response.json()
                data = [{k: v for k, v in stock.items() if stock['symbol'] in symbols} for stock in data]
                data = [entry for entry in data if entry]

                res_list = []
                for item in data:
                    try:
                        symbol = item['symbol']
                        quote_data = await get_quote_data(symbol)

                        if quote_data:
                            item = {
                                'symbol': symbol,
                                'name': quote_data.get('name',None),
                                'price': round(quote_data.get('price', 0), 2),
                                'changesPercentage': round(quote_data.get('changesPercentage', 0), 2),
                                'marketCap': quote_data.get('marketCap', 0),
                                'revenue': None,
                            }
                            item['revenue'] = stock_screener_data_dict[symbol].get('revenue')

                        if item['marketCap'] > 0:
                            res_list.append(item)
                    except Exception as e:
                        print(e)

            if res_list:
                res_list = sorted(res_list, key=lambda x: x['marketCap'] or 0, reverse=True)
                
                # Add rank to each item
                for rank, item in enumerate(res_list, 1):
                    item['rank'] = rank

                if index_list == 'nasdaq':
                    extension = '100'
                else:
                    extension = ''
                with open(f"json/stocks-list/list/{index_list+extension}.json", 'wb') as file:
                    file.write(orjson.dumps(res_list))

async def get_all_stock_tickers():
    try:
        '''
        with sqlite3.connect('etf.db') as etf_con:
            etf_cursor = etf_con.cursor()
            etf_cursor.execute("PRAGMA journal_mode = wal")
            etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
            etf_symbols = [row[0] for row in etf_cursor.fetchall()]
        '''
        with sqlite3.connect('stocks.db') as con:
            cursor = con.cursor()
            cursor.execute("PRAGMA journal_mode = wal")
            cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
            stock_symbols = [row[0] for row in cursor.fetchall()]

        res_list = []
        for symbol in stock_symbols:
            try:
                
                try:
                    with open(f"json/quote/{symbol}.json", "rb") as file:
                        quote_data = orjson.loads(file.read())
                except (FileNotFoundError, orjson.JSONDecodeError):
                    quote_data = None

                if quote_data:
                    item = {
                        'symbol': symbol,
                        'name': quote_data.get('name',None),
                        'price': round(quote_data.get('price'), 2) if quote_data.get('price') is not None else None,
                        'changesPercentage': round(quote_data.get('changesPercentage'), 2) if quote_data.get('changesPercentage') is not None else None,
                        'marketCap': quote_data.get('marketCap', None),
                        'revenue': None,
                    }
                    
                    # Add screener data if available
                    if symbol in stock_screener_data_dict:
                        item['revenue'] = stock_screener_data_dict[symbol].get('revenue')
                    
                    if item['marketCap'] > 0:
                        res_list.append(item)

            
            except Exception as e:
                print(f"Error processing symbol {symbol}: {e}")
            
        if res_list:
            res_list = sorted(res_list, key=lambda x: x['symbol'], reverse=False)
                
            with open("json/stocks-list/list/all-stock-tickers.json", 'wb') as file:
                file.write(orjson.dumps(res_list))

    except Exception as e:
        print(f"Database error: {e}")

async def run():
    await asyncio.gather(
        get_all_stock_tickers(),
        get_index_list(),
        etf_bitcoin_list(),
        get_magnificent_seven()
    )


    """Main function to run the analysis for all categories"""
    market_cap_conditions = {
        'mega-cap-stocks': "marketCap >= 200e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'large-cap-stocks': "marketCap < 200e9 AND marketCap >= 10e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'mid-cap-stocks': "marketCap < 10e9 AND marketCap >= 2e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'small-cap-stocks': "marketCap < 2e9 AND marketCap >= 300e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'micro-cap-stocks': "marketCap < 300e6 AND marketCap >= 50e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'nano-cap-stocks': "marketCap < 50e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')"
    }

    sector_conditions = {
        'financial': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Financials' OR sector = 'Financial Services')",
        'healthcare': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Healthcare')",
        'technology': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Technology')",
        'industrials': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Industrials')",
        'consumer-cyclical': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Consumer Cyclical')",
        'real-estate': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Real Estate')",
        'basic-materials': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Basic Materials')",
        'communication-services': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Communication Services')",
        'energy': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Energy')",
        'consumer-defensive': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Consumer Defensive')",
        'utilities': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Utilities')"
    }

    country_conditions = {
        'de': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'DE'",
        'ca': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'CA'",
        'cn': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'CN'",
        'in': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'IN'",
        'il': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'IL'",
        'gb': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'GB'",
        'jp': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'JP'",
    }

    exchange_conditions = {
        'nasdaq': "exchangeShortName = 'NASDAQ'",
        'nyse': "exchangeShortName = 'NYSE'",
        'amex': "exchangeShortName = 'AMEX'",
    }

    try:
        con = sqlite3.connect('stocks.db')
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")

        etf_con = sqlite3.connect('etf.db')
        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]

        await get_all_reits_list(cursor)

        for category, condition in exchange_conditions.items():
            await process_category(cursor, category, condition, 'stocks-list')
            #await asyncio.sleep(1)  # Small delay between categories

        for category, condition in country_conditions.items():
            await process_category(cursor, category, condition, 'stocks-list')
            #await asyncio.sleep(1)  # Small delay between categories

        for category, condition in market_cap_conditions.items():
            await process_category(cursor, category, condition, 'market-cap')
            #await asyncio.sleep(1)  # Small delay between categories

        # Process sector categories
        for category, condition in sector_conditions.items():
            await process_category(cursor, category, condition, 'sector')
            #await asyncio.sleep(1)  # Small delay between categories
        
        
        await get_etf_holding(etf_symbols, etf_con)
        await get_etf_provider(etf_con)


    except Exception as e:
        print(e)
        raise
    finally:
        con.close()
        etf_con.close()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())
    except Exception as e:
        print(e)
    finally:
        loop.close()