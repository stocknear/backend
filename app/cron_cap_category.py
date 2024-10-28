import orjson
import sqlite3
import asyncio
import aiohttp
from tqdm import tqdm


# Load stock screener data
with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}


async def save_json(cap_category, data):
    """Save data to JSON file for specific market cap category."""
    with open(f"json/market-cap/list/{cap_category}.json", 'wb') as file:  # Note: 'wb' for binary write
        file.write(orjson.dumps(data))

async def get_quote_data(symbol):
    """Get quote data for a symbol from JSON file"""
    try:
        with open(f"json/quote/{symbol}.json", 'r') as file:
            return orjson.loads(file.read())
    except FileNotFoundError:
        return None

async def process_market_cap_category(cursor, category, condition):
    """Process stocks for a specific market cap category"""
    base_query = """
        SELECT DISTINCT s.symbol, s.name, s.exchangeShortName, s.marketCap 
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
                'price': quote_data.get('price'),
                'changesPercentage': quote_data.get('changesPercentage'),
                'marketCap': quote_data.get('marketCap'),
                'revenue': None,
            }
            
            # Add screener data if available
            if symbol in stock_screener_data_dict:
                item['revenue'] = stock_screener_data_dict[symbol].get('revenue')
            
            result_list.append(item)
    
    # Sort by market cap and save
    sorted_result = sorted(result_list, key=lambda x: x['marketCap'] if x['marketCap'] else 0, reverse=True)
    # Add rank to each item
    for rank, item in enumerate(sorted_result, 1):
        item['rank'] = rank

    await save_json(category, sorted_result)
    print(f"Processed and saved {len(sorted_result)} stocks for {category}")
    return sorted_result

async def run():
    """Main function to run the analysis for all market cap categories"""
    conditions = {
        'mega-cap-stocks': "marketCap >= 200e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'large-cap-stocks': "marketCap < 200e9 AND marketCap >= 10e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'mid-cap-stocks': "marketCap < 10e9 AND marketCap >= 2e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'small-cap-stocks': "marketCap < 2e9 AND marketCap >= 300e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'micro-cap-stocks': "marketCap < 300e6 AND marketCap >= 50e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'nano-cap-stocks': "marketCap < 50e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')"
    }

    try:
        con = sqlite3.connect('stocks.db')
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")

        # Process each market cap category
        for category, condition in conditions.items():
            await process_market_cap_category(cursor, category, condition)
            await asyncio.sleep(1)  # Small delay between categories

    except Exception as e:
        print(e)
        raise
    finally:
        con.close()

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())
    except Exception as e:
        print(e)
    finally:
        loop.close()