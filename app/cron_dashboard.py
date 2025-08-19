import aiohttp
import aiofiles
import ujson
import orjson
import sqlite3
import pandas as pd
import asyncio
import pytz
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta, date, timezone
import sqlite3
import email.utils  # for parsing RFC 2822 dates like "Fri, 30 May 2025 16:01:09 -0400"

headers = {"accept": "application/json"}

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

def add_time_ago(news_items):
    now_utc = datetime.now(timezone.utc)
    
    for item in news_items:
        created_dt = email.utils.parsedate_to_datetime(item['created']).astimezone(timezone.utc)
        diff = now_utc - created_dt
        minutes = int(diff.total_seconds() / 60)

        if minutes < 1:
            item['timeAgo'] = "1m"
        elif minutes < 60:
            item['timeAgo'] = f"{minutes}m"
        elif minutes < 1440:
            hours = minutes // 60
            item['timeAgo'] = f"{hours}h"
        else:
            days = minutes // 1440
            item['timeAgo'] = f"{days}D"
    
    return news_items


load_dotenv()
API_KEY = os.getenv('BENZINGA_API_KEY')


async def save_json(data):
    with open(f"json/dashboard/data.json", 'w') as file:
        ujson.dump(data, file)


def parse_time(time_str):
    try:
        # Try parsing as full datetime
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            # Try parsing as time only
            time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
            # Combine with today's date
            return datetime.combine(date.today(), time_obj)
        except ValueError:
            # If all else fails, return a default datetime
            return datetime.min

def remove_duplicates(elements):
    seen = set()
    unique_elements = []
    
    for element in elements:
        if element['symbol'] not in seen:
            seen.add(element['symbol'])
            unique_elements.append(element)
    
    return unique_elements

def weekday():
    today = datetime.today()
    if today.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        yesterday = today - timedelta(2)
    else:
        yesterday = today - timedelta(1)

    return yesterday.strftime('%Y-%m-%d')


today = datetime.today().strftime('%Y-%m-%d')
tomorrow = (datetime.today() + timedelta(1))
yesterday = weekday()

if tomorrow.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
    tomorrow = tomorrow + timedelta(days=(7 - tomorrow.weekday()))

tomorrow = tomorrow.strftime('%Y-%m-%d')

async def get_upcoming_earnings(session, end_date, filter_today=True):
    url = "https://api.benzinga.com/api/v2.1/calendar/earnings"
    importance_list = ["1", "2", "3", "4", "5"]
    res_list = []
    today = date.today().strftime('%Y-%m-%d')

    for importance in importance_list:
        querystring = {
            "token": API_KEY,
            "parameters[importance]": importance,
            "parameters[date_from]": today,
            "parameters[date_to]": end_date,
            "parameters[date_sort]": "date"
        }
        try:
            async with session.get(url, params=querystring, headers=headers) as response:
                res = ujson.loads(await response.text())['earnings']
                
                # Apply the time filter if filter_today is True
                if filter_today:
                    res = [
                        e for e in res if
                        datetime.strptime(e['date'], "%Y-%m-%d").date() != date.today() or
                        datetime.strptime(e['time'], "%H:%M:%S").time() >= datetime.strptime("16:00:00", "%H:%M:%S").time()
                    ]
                
                for item in res:
                    try:
                        symbol = item['ticker']
                        name = item['name']
                        time = item['time']
                        is_today = item['date'] == today
                        eps_prior = float(item['eps_prior']) if item['eps_prior'] != '' else None
                        eps_est = float(item['eps_est']) if item['eps_est'] != '' else None
                        revenue_est = float(item['revenue_est']) if item['revenue_est'] != '' else None
                        revenue_prior = float(item['revenue_prior']) if item['revenue_prior'] != '' else None

                        if symbol in stock_symbols and revenue_est is not None and revenue_prior is not None and eps_prior is not None and eps_est is not None:
                            with open(f"json/quote/{symbol}.json","r") as file:
                                quote_data = orjson.loads(file.read())

                            market_cap = quote_data.get('marketCap',0)

                            res_list.append({
                                'symbol': symbol,
                                'name': name,
                                'time': time,
                                'isToday': is_today,
                                'marketCap': market_cap,
                                'epsPrior': eps_prior,
                                'epsEst': eps_est,
                                'revenuePrior': revenue_prior,
                                'revenueEst': revenue_est
                            })
                    except Exception as e:
                        print('Upcoming Earnings:', e)
                        pass
        except Exception as e:
            print(e)
            pass

    try:
        res_list = remove_duplicates(res_list)
        res_list.sort(key=lambda x: x['marketCap'], reverse=True)
        return res_list[:10]
    except Exception as e:
        print(e)
        return []


async def get_recent_earnings(session):
    url = "https://api.benzinga.com/api/v2.1/calendar/earnings"
    res_list = []
    importance_list = ["1","2","3","4","5"]
    
    for importance in importance_list:
        querystring = {
            "token": API_KEY,
            "parameters[importance]": importance, 
            "parameters[date_from]": yesterday,
            "parameters[date_to]": today,
            "parameters[date_sort]": "date"
        }
        try:
            async with session.get(url, params=querystring, headers=headers) as response:
                res = ujson.loads(await response.text())['earnings']
                for item in res:
                    try:
                        symbol = item['ticker']
                        name = item['name']
                        time = item['time']
                        updated = int(item['updated'])  # Convert to integer for proper comparison
                        
                        # Convert numeric fields, handling empty strings
                        eps_prior = float(item['eps_prior']) if item['eps_prior'] != '' else None
                        eps_surprise = float(item['eps_surprise']) if item['eps_surprise'] != '' else None
                        eps = float(item['eps']) if item['eps'] != '' else 0
                        revenue_prior = float(item['revenue_prior']) if item['revenue_prior'] != '' else None
                        revenue_surprise = float(item['revenue_surprise']) if item['revenue_surprise'] != '' else None
                        revenue = float(item['revenue']) if item['revenue'] != '' else None
                        
                        if (symbol in stock_symbols and 
                            revenue is not None and 
                            revenue_prior is not None and 
                            eps_prior is not None and 
                            eps is not None and 
                            revenue_surprise is not None and 
                            eps_surprise is not None):
                            
                            with open(f"json/quote/{symbol}.json","r") as file:
                                quote_data = orjson.loads(file.read())
                            market_cap = quote_data.get('marketCap',0)
                            
                            res_list.append({
                                'symbol': symbol,
                                'date': item['date'],
                                'name': name,
                                'time': time,
                                'marketCap': market_cap,
                                'epsPrior': eps_prior,
                                'epsSurprise': eps_surprise,
                                'eps': eps,
                                'revenuePrior': revenue_prior,
                                'revenueSurprise': revenue_surprise,
                                'revenue': revenue,
                                'updated': updated
                            })
                    except Exception as e:
                        print('Recent Earnings:', e)
                        pass
        except Exception as e:
            print('API Request Error:', e)
            pass
    
    # Remove duplicates
    res_list = remove_duplicates(res_list)
    
    # Sort first by the most recent 'updated' timestamp, then by market cap
    res_list.sort(key=lambda x: (-x['updated'], -x['marketCap']))
    
    # Remove market cap before returning and limit to top 10
    res_list = [{k: v for k, v in d.items() if k not in ['marketCap', 'updated']} for d in res_list]
    
    return res_list[:10]


async def get_analyst_report():
    try:
        # Connect to the database and retrieve symbols
        with sqlite3.connect('stocks.db') as con:
            cursor = con.cursor()
            cursor.execute("PRAGMA journal_mode = wal")
            cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%' AND marketCap > 10E9")
            symbols = {row[0] for row in cursor.fetchall()}  # Use a set for fast lookups

        # Define the directory path
        directory = Path("json/analyst/insight")
        
        # Track the latest data and symbol based on the "date" key in the JSON file
        latest_data = None
        latest_symbol = None
        latest_date = datetime.min  # Initialize to the earliest possible date
        
        # Loop through all .json files in the directory
        for file_path in directory.glob("*.json"):
            symbol = file_path.stem  # Get the filename without extension
            if symbol in symbols:
                # Read each JSON file asynchronously
                async with aiofiles.open(file_path, mode='r') as file:
                    data = ujson.loads(await file.read())
                    
                    # Parse the "date" field and compare it to the latest_date
                    data_date = datetime.strptime(data.get('date', ''), '%b %d, %Y')
                    if data_date > latest_date:
                        latest_date = data_date
                        latest_data = data
                        latest_symbol = symbol

        # If the latest report and symbol are found, add additional data from the summary file
        if latest_symbol and latest_data:
            summary_path = Path(f"json/analyst/summary/all_analyst/{latest_symbol}.json")
            if summary_path.is_file():  # Ensure the summary file exists
                async with aiofiles.open(summary_path, mode='r') as file:
                    summary_data = ujson.loads(await file.read())
                    # Merge the summary data into the latest data dictionary
                    latest_data.update({
                        'symbol': latest_symbol,
                        'numOfAnalyst': summary_data.get('numOfAnalyst'),
                        'consensusRating': summary_data.get('consensusRating'),
                        'medianPriceTarget': summary_data.get('medianPriceTarget'),
                        'avgPriceTarget': summary_data.get('avgPriceTarget'),
                        'lowPriceTarget': summary_data.get('lowPriceTarget'),
                        'highPriceTarget': summary_data.get('highPriceTarget')
                    })

            # Load the current price from the quote file
            quote_path = Path(f"json/quote/{latest_symbol}.json")
            if quote_path.is_file():
                async with aiofiles.open(quote_path, mode='r') as file:
                    quote_data = ujson.loads(await file.read())
                    price = quote_data.get('price')

                    if price:
                        # Calculate the percentage change for each target relative to the current price
                        def calculate_percentage_change(target):
                            return round(((target - price) / price) * 100, 2) if target is not None else None

                        latest_data.update({
                            'medianPriceChange': calculate_percentage_change(latest_data.get('medianPriceTarget')),
                            'avgPriceChange': calculate_percentage_change(latest_data.get('avgPriceTarget')),
                            'lowPriceChange': calculate_percentage_change(latest_data.get('lowPriceTarget')),
                            'highPriceChange': calculate_percentage_change(latest_data.get('highPriceTarget')),
                        })

                #print(f"The latest report for symbol {latest_symbol}:", latest_data)

        # Return the latest data found
        return latest_data if latest_data else {}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

async def get_latest_wiim():
    url = "https://api.benzinga.com/api/v2/news"
    querystring = {
        "token": API_KEY,
        "dateFrom": yesterday,
        "dateTo": today,
        "sort": "updated:desc",
        "pageSize": 1000,
        "channels": "WIIM"
    }
    max_retries = 3
    retry_delay = 2  # seconds
    res_list = []

    for attempt in range(max_retries):
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=querystring, headers=headers) as response:
                if response.status != 200:
                    await asyncio.sleep(retry_delay)
                    continue
                
                data = ujson.loads(await response.text())
                data = add_time_ago(data)
                for item in data:
                    try:
                        item['ticker'] = item['stocks'][0].get('name', None).replace('/', '-')  #important for BRK-A & BRK-B

                        with open(f"json/quote/{item['ticker']}.json", "r") as file:
                            quote_data = ujson.load(file)
                            item['marketCap'] = quote_data.get('marketCap', None)
                        
                        if item['ticker'] in stock_symbols:
                            item['assetType'] = 'stocks'
                        else:
                            item['assetType'] = 'etf'

                        res_list.append({
                            'date': item['created'],
                            'text': item['title'],
                            'marketCap': item['marketCap'],
                            'ticker': item['ticker'],
                            'assetType': item['assetType'],
                            "timeAgo": item['timeAgo']
                        })
                    except:
                        pass
                
                if res_list:
                    break  # Exit retry loop if data is fetched successfully
        
        if res_list:
            break
        else:
            await asyncio.sleep(retry_delay)

    res_list = sorted(
        res_list,
        key=lambda item: datetime.strptime(item['date'], '%a, %d %b %Y %H:%M:%S %z'),
        reverse=True
    )
    for item in res_list:
        dt = datetime.strptime(item['date'], '%a, %d %b %Y %H:%M:%S %z')
        item['date'] = dt.strftime('%Y-%m-%d')

    return res_list[:20]


def get_dark_pool():
    with open(f"json/dark-pool/feed/data.json","rb") as file:
        data = orjson.loads(file.read())

    n = 5
    items = data[:]
    top = []
    count = min(n, len(items))
    keys_to_keep = {"ticker", "price", "size", "premium", "assetType", "sizeAvgVolRatio"}

    for _ in range(count):
        try:
            # Find index of item with highest premium
            max_idx = 0
            for i in range(1, len(items)):
                if items[i].get('premium', 0) > items[max_idx].get('premium', 0):
                    max_idx = i

            # Pop the max item and filter its keys
            item = items.pop(max_idx)
            filtered = {k: item[k] for k in keys_to_keep if k in item}
            top.append(filtered)
        except:
            pass

    return top

def get_options_flow():
    with open(f"json/options-flow/feed/data.json","rb") as file:
        data = orjson.loads(file.read())
    n = 10
    items = data[:]
    top = []
    count = min(n, len(items))
    keys_to_keep = {"ticker", "strike_price","cost_basis", "underlying_type","sentiment", "put_call"}

    for _ in range(count):
        try:
            # Find index of item with highest cost_basis
            max_idx = 0
            for i in range(1, len(items)):
                if items[i].get('cost_basis', 0) > items[max_idx].get('cost_basis', 0):
                    max_idx = i

            # Pop the max item and filter its keys
            item = items.pop(max_idx)
            filtered = {k: item[k] for k in keys_to_keep if k in item}
            top.append(filtered)
        except:
            pass
        
    return top

def get_economic_calendar():
    today_str = date.today().isoformat()

    # Load data from JSON
    with open("json/economic-calendar/data.json", "rb") as f:
        data = orjson.loads(f.read())

    # Filter US events for today
    us_events = [
        event for event in data
        if event.get("countryCode", "").lower() == "us"
        and event.get("date") == today_str
    ]
    
    filtered = []
    # Check thresholds in descending order
    for threshold in [1,2,3]:
        try:
            # Filter events by current threshold or higher
            filtered += [
                event for event in us_events
                if event.get("importance", 0) >= threshold
            ]
            if filtered:
                sorted_events = sorted(filtered, key=lambda ev: ev.get("time", ""))
                top_5_events = sorted_events[:5]
                # Transform to include only necessary keys
                return [
                    {key: ev.get(key) for key in ["time", "prior", "consensus", "event"]}
                    for ev in top_5_events
                ]
        except:
            pass
    
    # Return empty list if no events found
    return []


async def run():
    async with aiohttp.ClientSession() as session:

        economic_calendar_list = get_economic_calendar()

        options_flow_list = get_options_flow()
        dark_pool_list = get_dark_pool()
            
        recent_earnings = await get_recent_earnings(session)

        upcoming_earnings = await get_upcoming_earnings(session, today, filter_today=False)

       
        if len(upcoming_earnings) < 5:
            upcoming_earnings = await get_upcoming_earnings(session, today, filter_today=True)

        if len(upcoming_earnings) < 5:
            upcoming_earnings = await get_upcoming_earnings(session, tomorrow, filter_today=True)

        recent_wiim = await get_latest_wiim()

        
        upcoming_earnings = [
            item for item in upcoming_earnings 
            if item['symbol'] not in [earning['symbol'] for earning in recent_earnings]
        ]
        
        analyst_report = await get_analyst_report()
        
        '''
        try:
            with open("json/stocks-list/list/highest-open-interest-change.json", 'r') as file:
                highest_open_interest_change = ujson.load(file)[:3]
            
            with open("json/stocks-list/list/highest-option-iv-rank.json", 'r') as file:
                highest_iv_rank = ujson.load(file)[:3]

            with open("json/stocks-list/list/highest-option-premium.json", 'r') as file:
                highest_premium = ujson.load(file)[:3]
                optionsData = {
                    'premium': highest_premium,
                    'ivRank': highest_iv_rank,
                    'openInterest': highest_open_interest_change
                }
        except Exception as e:
            print(e)
            optionsData = {}
        '''
    

        market_status = check_market_hours()
        gainers_list = []
        losers_list = []

        if market_status == 0:
            try:
                with open("json/market-movers/markethours/gainers.json", 'r') as file:
                    gainers = ujson.load(file)
                with open("json/market-movers/markethours/losers.json", 'r') as file:
                    losers = ujson.load(file)
                gainers_list = gainers['1D'][:10]
                losers_list = losers['1D'][:10]
            except:
                market_movers = {}
        elif market_status == 1:
            try:
                with open("json/market-movers/premarket/gainers.json", 'r') as file:
                    data = ujson.load(file)
                    gainers = [
                        {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                         'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                        for item in data[:10]
                    ]

                with open("json/market-movers/premarket/losers.json", 'r') as file:
                    data = ujson.load(file)
                    losers = [
                        {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                         'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                        for item in data[:10]
                    ]
        
                gainers_list = gainers
                losers_list = losers

            except Exception as e:
                print(e)
                market_movers = {}
        elif market_status == 2:
            try:
                with open("json/market-movers/afterhours/gainers.json", 'r') as file:
                    data = ujson.load(file)
                    gainers = [
                        {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                         'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                        for item in data[:10]
                    ]

                with open("json/market-movers/afterhours/losers.json", 'r') as file:
                    data = ujson.load(file)
                    losers = [
                        {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                         'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                        for item in data[:10]
                    ]
    
                gainers_list = gainers
                losers_list = losers
            except:
                gainers_list = []
                losers_list = []

        data = {
            'gainers': gainers_list,
            'losers': losers_list,
            'marketStatus': market_status,
            'recentEarnings': recent_earnings,
            'upcomingEarnings': upcoming_earnings,
            'wiim': recent_wiim,
            'darkpool': dark_pool_list,
            'optionsFlow': options_flow_list,
            'analystReport': analyst_report,
            #"economicCalendar": economic_calendar_list,
        }

        if len(data) > 0:
            await save_json(data)

try:
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()


    asyncio.run(run())

except Exception as e:
    print(e)