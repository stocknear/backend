import orjson
import sqlite3
import asyncio
import aiohttp
import pandas as pd
from tqdm import tqdm


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
    """
    Process stocks for a specific category (market cap or sector)
    
    Args:
        cursor: Database cursor
        category: Category name
        condition: SQL WHERE condition
        category_type: Either 'market-cap' or 'sector'
    """
    base_query = """
        SELECT DISTINCT s.symbol, s.name, s.exchangeShortName, s.marketCap, s.sector
        FROM stocks s 
        WHERE {}
    """
    
    full_query = base_query.format(condition)
    cursor.execute(full_query)
    raw_data = cursor.fetchall()
    
    result_list = []
    for row in raw_data:
        symbol = row[0]
        quote_data = await get_quote_data(symbol)
        if quote_data:
            item = {
                'symbol': symbol,
                'name': row[1],
                'price': round(quote_data.get('price'),2),
                'changesPercentage': round(quote_data.get('changesPercentage'),2),
                'marketCap': quote_data.get('marketCap'),
                'sector': row[4],  # Include sector information
                'revenue': None,
            }
            
            # Add screener data if available
            if symbol in stock_screener_data_dict:
                item['revenue'] = stock_screener_data_dict[symbol].get('revenue')
            
            if item['marketCap'] > 0:
                result_list.append(item)
    
    # Sort by market cap and save
    sorted_result = sorted(result_list, key=lambda x: x['marketCap'] if x['marketCap'] else 0, reverse=True)
    # Add rank to each item
    for rank, item in enumerate(sorted_result, 1):
        item['rank'] = rank

    await save_json(category, sorted_result, category_type)
    print(f"Processed and saved {len(sorted_result)} stocks for {category}")
    return sorted_result

def get_etf_holding(etf_symbols, etf_con):

    for ticker in tqdm(etf_symbols):
        res = []
        df = pd.read_sql_query(query_etf_holding, etf_con, params=(ticker,))

        try:
            # Load holdings data from the SQL query result
            data = orjson.loads(df['holding'].iloc[0])
            res = [{key: item[key] for key in ('asset', 'weightPercentage', 'sharesNumber')} for item in data]
            
            for item in res:
                asset = item['asset']
                
                # Check if the asset data is already in the cache
                if asset in quote_cache:
                    quote_data = quote_cache[asset]
                else:
                    # Load the quote data from file if not in cache
                    try:
                        with open(f"json/quote/{asset}.json") as file:
                            quote_data = orjson.loads(file.read())
                            quote_cache[asset] = quote_data  # Cache the loaded data
                    except:
                        quote_data = None

                # Assign price and changesPercentage if available, otherwise set to None
                item['price'] = round(quote_data.get('price'), 2) if quote_data else None
                item['changesPercentage'] = round(quote_data.get('changesPercentage'), 2) if quote_data else None
                item['name'] = quote_data.get('name') if quote_data else None

        except Exception as e:
            print(e)
            res = []

        # Save results to a file if there's data to write
        if res:
            with open(f"json/etf/holding/{ticker}.json", 'wb') as file:
                file.write(orjson.dumps(res))

def get_etf_provider(etf_symbols, etf_con):


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
                    if symbol in quote_cache:
                        quote_data = quote_cache[symbol]
                    else:
                        # Load the quote data from file if not in cache
                        try:
                            with open(f"json/quote/{symbol}.json") as file:
                                quote_data = orjson.loads(file.read())
                                quote_cache[symbol] = quote_data  # Cache the loaded data
                        except:
                            quote_data = None

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
        except:
            pass
    cursor.close()


async def run():
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

    try:
        con = sqlite3.connect('stocks.db')
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")

        etf_con = sqlite3.connect('etf.db')
        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]

        # Process market cap categories
        
        for category, condition in market_cap_conditions.items():
            await process_category(cursor, category, condition, 'market-cap')
            await asyncio.sleep(1)  # Small delay between categories

        # Process sector categories
        for category, condition in sector_conditions.items():
            await process_category(cursor, category, condition, 'sector')
            await asyncio.sleep(1)  # Small delay between categories
        

        get_etf_holding(etf_symbols, etf_con)
        get_etf_provider(etf_symbols, etf_con)


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