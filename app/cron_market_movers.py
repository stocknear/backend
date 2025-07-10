from datetime import date, datetime, timedelta, time
import ujson
import orjson
import sqlite3
import pandas as pd
import asyncio
import aiohttp
import pytz
from utils.helper import check_market_hours

from GetStartEndDate import GetStartEndDate

#Update Market Movers Price, ChangesPercentage, Volume and MarketCap regularly
berlin_tz = pytz.timezone('Europe/Berlin')
ny_timezone = pytz.timezone("America/New_York")

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')


market_cap_threshold = 10E9
volume_threshold = 50_000
price_threshold = 10
today = datetime.now(ny_timezone).date()


def check_market_hours():

    holidays = ['2025-01-01', '2025-01-09','2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25']
    
    # Get the current date and time in ET (Eastern Time)
    et_timezone = pytz.timezone('America/New_York')
    current_time = datetime.now(et_timezone)
    current_date_str = current_time.strftime('%Y-%m-%d')
    current_hour = current_time.hour
    current_minute = current_time.minute
    current_day = current_time.weekday()  # Monday is 0, Sunday is 6

    # Check if the current date is a holiday or weekend
    is_weekend = current_day >= 5  # Saturday (5) or Sunday (6)
    is_holiday = current_date_str in holidays

    # Determine the market status
    if is_weekend or is_holiday:
        return 0 #Closed
    elif current_hour < 9 or (current_hour == 9 and current_minute < 30):
        return 1 # Pre-Market
    elif 9 <= current_hour < 16 or (current_hour == 16 and current_minute == 0):
        return 0 #"Market hours."
    elif 16 <= current_hour < 24:
        return 2 #"After-market hours."
    else:
        return 0 #"Market is closed."

market_status = check_market_hours()

async def get_quote_of_stocks(ticker_list):
    res_list = []
    for symbol in ticker_list:
        try:
            with open(f"json/quote/{symbol}.json") as file:
                data = orjson.loads(file.read())
                res_list.append(data)
        except:
            pass

    return res_list

def deep_copy(data):
    if isinstance(data, dict):
        return {key: deep_copy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [deep_copy(item) for item in data]
    else:
        return data  # Base case for non-nested elements (e.g., int, float, str)



async def get_gainer_loser_active_stocks(symbols):
    res_list = []

    for symbol in symbols:
        try:
            # Load the main quote JSON file
            with open(f"json/quote/{symbol}.json", "r") as file:
                data = orjson.loads(file.read())
                market_cap = int(data.get('marketCap', 0))
                name = data.get('name', None)
                volume = data.get('volume', 0)
                changes_percentage = data.get("changesPercentage", None)
                price = data.get("price", None)
                exchange = data.get('exchange',None)
                dt = datetime.fromtimestamp(data['timestamp'], ny_timezone).date()
                # Ensure the stock meets criteria
                if (today - dt).days <= 5 and market_cap >= market_cap_threshold and price >= price_threshold and exchange in ['AMEX','NASDAQ','NYSE']:
                    if price and changes_percentage and changes_percentage < 100:
                        res_list.append({
                            "symbol": symbol,
                            "name": name,
                            "price": price,
                            "volume": volume,
                            "changesPercentage": changes_percentage,
                            "marketCap": market_cap
                        })
        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
            continue

    # Load past market movers data
    query_market_movers = """
        SELECT 
            gainer, loser, most_active
        FROM 
            market_movers
    """
    past_gainer = pd.read_sql_query(query_market_movers, con)
    gainer_json = eval(past_gainer['gainer'].iloc[0])
    loser_json = eval(past_gainer['loser'].iloc[0])
    active_json = eval(past_gainer['most_active'].iloc[0])

    # Initialize final data structure
    final_data = {
        'gainers': gainer_json.copy(),
        'losers': loser_json.copy(),
        'active': active_json.copy()
    }

    # Process current data
    current_data = {
        'gainers': sorted([x for x in res_list if x['changesPercentage'] > 0], 
                         key=lambda x: x['changesPercentage'], 
                         reverse=True),
        'losers': sorted([x for x in res_list if x['changesPercentage'] < 0], 
                        key=lambda x: x['changesPercentage']),
        'active': sorted([x for x in res_list if x['volume'] > 0], 
                        key=lambda x: x['volume'], 
                        reverse=True)
    }

    # Update latest quotes for current data
    unique_symbols = {stock["symbol"] for category in current_data.values() for stock in category}
    latest_quote = await get_quote_of_stocks(list(unique_symbols))

    # Update market cap and volume with latest data
    for category in current_data.keys():
        for stock in current_data[category]:
            symbol = stock["symbol"]
            quote_stock = next((item for item in latest_quote if item["symbol"] == symbol), None)
            if quote_stock:
                stock['marketCap'] = quote_stock.get('marketCap', stock['marketCap'])
                stock['volume'] = quote_stock.get('volume', stock['volume'])

    # Add fresh rankings to current data
    for category, stocks in current_data.items():
        # Sort again after updates
        if category == 'gainers':
            stocks.sort(key=lambda x: x['changesPercentage'], reverse=True)
        elif category == 'losers':
            stocks.sort(key=lambda x: x['changesPercentage'])
        elif category == 'active':
            stocks.sort(key=lambda x: x['volume'], reverse=True)
        
        # Apply sequential rankings
        for i, stock in enumerate(stocks, 1):
            stock['rank'] = i

        # Update the 1D data in final_data
        final_data[category]['1D'] = stocks


    categories = ['gainers', 'losers','active']
    #super weird bug that is only fixed with deep_copy for the ranking
    for category in categories:
        for period in ['1D', '1W', '1M', '1Y', '3Y', '5Y']:
            for rank, item in enumerate(final_data[category][period], start=1):
                # Create a deep copy of the item to avoid overwriting shared references
                final_data[category][period][rank - 1] = deep_copy(item)
                final_data[category][period][rank - 1]['rank'] = rank


    return final_data




async def get_pre_after_market_movers(symbols):
    res_list = []

    # Loop through the symbols and load the corresponding JSON files
    for symbol in symbols:
        try:
            # Load the main quote JSON file
            with open(f"json/quote/{symbol}.json", "r") as file:
                data = orjson.loads(file.read())
                market_cap = int(data.get('marketCap', 0))
                name = data.get('name',None)
                exchange = data.get('exchange',None)

            if market_cap >= market_cap_threshold:
                with open(f"json/pre-post-quote/{symbol}.json", "r") as file:
                    pre_post_data = orjson.loads(file.read())
                    price = pre_post_data.get("price", None)
                    changes_percentage = pre_post_data.get("changesPercentage", None)
                    with open(f"json/one-day-price/{symbol}.json", 'rb') as file:
                        one_day_price = orjson.loads(file.read())
                        # Filter out entries where 'close' is None
                        filtered_prices = [price for price in one_day_price if price['close'] is not None]

                    if price and price >= price_threshold and exchange in ['AMEX','NASDAQ','NYSE'] and changes_percentage and len(filtered_prices) > 100: #300
                        res_list.append({
                            "symbol": symbol,
                            "name": name,
                            "price": price,
                            "changesPercentage": changes_percentage
                        })
        except:
            pass


    gainers = sorted([x for x in res_list if x['changesPercentage'] > 0], key=lambda x: x['changesPercentage'], reverse=True)
    losers = sorted([x for x in res_list if x['changesPercentage'] < 0], key=lambda x: x['changesPercentage'])


    for index, item in enumerate(gainers, start=1):
        item['rank'] = index  # Add rank field
    for index, item in enumerate(losers, start=1):
        item['rank'] = index  # Add rank field

    data = {'gainers': gainers, 'losers': losers}

    unique_symbols = set()

    # Iterate through categories and symbols
    for category in data.keys():
        # Add rank and process symbols
        for index, stock_data in enumerate(data[category], start=1):
            stock_data['rank'] = index  # Add rank field
            symbol = stock_data["symbol"]
            unique_symbols.add(symbol)

    # Convert the set to a list if needed
    unique_symbols_list = list(unique_symbols)

    # Get the latest quote of all unique symbols and map it back to the original data list to update all values
    latest_quote = await get_quote_of_stocks(unique_symbols_list)

    # Updating values in the data list based on matching symbols from the quote list
    for category in data.keys():
        for stock_data in data[category]:
            symbol = stock_data["symbol"]
            quote_stock = next((item for item in latest_quote if item["symbol"] == symbol), None)
            if quote_stock:
                stock_data['marketCap'] = quote_stock['marketCap']
                stock_data['volume'] = quote_stock['volume']

    return data


try:
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    symbols = [row[0] for row in cursor.fetchall()]
    #Filter out tickers
    #symbols = [symbol for symbol in symbols if symbol != "STEC"]

    
    data = asyncio.run(get_gainer_loser_active_stocks(symbols))
    for category in data.keys():
        with open(f"json/market-movers/markethours/{category}.json", 'w') as file:
            file.write(orjson.dumps(data[category]).decode("utf-8"))
    
    
    data = asyncio.run(get_pre_after_market_movers(symbols))
    if market_status == 1:
        for category in data.keys():
            with open(f"json/market-movers/premarket/{category}.json", 'w') as file:
                file.write(orjson.dumps(data[category]).decode("utf-8"))
    elif market_status == 2:
        for category in data.keys():
            with open(f"json/market-movers/afterhours/{category}.json", 'w') as file:
                file.write(orjson.dumps(data[category]).decode("utf-8"))
    

    con.close()
except Exception as e:
    print(e)