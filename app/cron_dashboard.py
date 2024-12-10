import aiohttp
import aiofiles
import ujson
import sqlite3
import pandas as pd
import asyncio
import pytz
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta, date
import sqlite3


headers = {"accept": "application/json"}

def check_market_hours():

    holidays = [
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
        "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
        "2024-11-28", "2024-12-25"
    ]
    
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


load_dotenv()
benzinga_api_key = os.getenv('BENZINGA_API_KEY')
fmp_api_key = os.getenv('FMP_API_KEY')

query_template = """
    SELECT 
        marketCap
    FROM 
        stocks 
    WHERE
        symbol = ?
"""


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
            "token": benzinga_api_key,
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
                        eps_prior = float(item['eps_prior']) if item['eps_prior'] != '' else 0
                        eps_est = float(item['eps_est']) if item['eps_est'] != '' else 0
                        revenue_est = float(item['revenue_est']) if item['revenue_est'] != '' else 0
                        revenue_prior = float(item['revenue_prior']) if item['revenue_prior'] != '' else 0

                        if symbol in stock_symbols and revenue_est is not None and revenue_prior is not None and eps_prior is not None and eps_est is not None:
                            df = pd.read_sql_query(query_template, con, params=(symbol,))
                            market_cap = float(df['marketCap'].iloc[0]) if df['marketCap'].iloc[0] != '' else 0
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
            "token": benzinga_api_key,
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
                            
                            df = pd.read_sql_query(query_template, con, params=(symbol,))
                            market_cap = float(df['marketCap'].iloc[0]) if df['marketCap'].iloc[0] != '' else 0
                            
                            res_list.append({
                                'symbol': symbol,
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

'''
async def get_recent_dividends(session):
	url = "https://api.benzinga.com/api/v2.1/calendar/dividends"
	importance_list = ["1","2","3","4","5"]
	res_list = []
	for importance in importance_list:
		querystring = {"token": benzinga_api_key,"parameters[importance]":importance,"parameters[date_from]":yesterday,"parameters[date_to]":today}
		try:
			async with session.get(url, params=querystring, headers=headers) as response:
				res = ujson.loads(await response.text())['dividends']
				for item in res:
					try:
						symbol = item['ticker']
						name = item['name']
						dividend = float(item['dividend']) if item['dividend'] != '' else 0
						dividend_prior = float(item['dividend_prior']) if item['dividend_prior'] != '' else 0
						dividend_yield = round(float(item['dividend_yield'])*100,2) if item['dividend_yield'] != '' else 0
						ex_dividend_date = item['ex_dividend_date'] if item['ex_dividend_date'] != '' else 0
						payable_date = item['payable_date'] if item['payable_date'] != '' else 0
						record_date = item['record_date'] if item['record_date'] != '' else 0
						if symbol in stock_symbols and dividend != 0 and payable_date != 0 and dividend_prior != 0 and ex_dividend_date != 0 and record_date != 0 and dividend_yield != 0:
							df = pd.read_sql_query(query_template, con, params=(symbol,))
							market_cap = float(df['marketCap'].iloc[0]) if df['marketCap'].iloc[0] != '' else 0
							res_list.append({
								'symbol': symbol,
								'name': name,
								'dividend': dividend,
								'marketCap': market_cap,
								'dividendPrior':dividend_prior,
								'dividendYield': dividend_yield,
								'exDividendDate': ex_dividend_date,
								'payableDate': payable_date,
								'recordDate': record_date,
								'updated': item['updated']
								})
					except Exception as e:
						print('Recent Dividends:', e)
						pass
		except Exception as e:
			print(e)
			pass

	res_list = remove_duplicates(res_list)
	res_list.sort(key=lambda x: x['marketCap'], reverse=True)
	res_list = [{k: v for k, v in d.items() if k != 'marketCap'} for d in res_list]
	return res_list[0:5]
'''

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
            summary_path = Path(f"json/analyst/summary/{latest_symbol}.json")
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

async def run():
    async with aiohttp.ClientSession() as session:
        recent_earnings = await get_recent_earnings(session)

        upcoming_earnings = await get_upcoming_earnings(session, today, filter_today=False)

        upcoming_earnings = [
            item for item in upcoming_earnings 
            if item['symbol'] not in [earning['symbol'] for earning in recent_earnings]
        ]

        if len(upcoming_earnings) < 5:
            upcoming_earnings = await get_upcoming_earnings(session, today, filter_today=True)

        if len(upcoming_earnings) < 5:
            upcoming_earnings = await get_upcoming_earnings(session, tomorrow, filter_today=True)

        recent_analyst_report = await get_analyst_report()

        upcoming_earnings = [
            item for item in upcoming_earnings 
            if item['symbol'] not in [earning['symbol'] for earning in recent_earnings]
        ]


        
        try:
            with open("json/options-flow/feed/data.json", 'r') as file:
                options_flow = ujson.load(file)
                
                # Filter the options_flow to include only items with ticker in total_symbol
                options_flow = [item for item in options_flow if item['ticker'] in stock_symbols]
                
                highest_volume = sorted(options_flow, key=lambda x: int(x['volume']), reverse=True)
                highest_volume = [
                    {key: item[key] for key in ['cost_basis', 'ticker', 'underlying_type', 'date_expiration', 'put_call', 'volume', 'strike_price']}
                    for item in highest_volume[0:4]
                ]

                highest_premium = sorted(options_flow, key=lambda x: int(x['cost_basis']), reverse=True)
                highest_premium = [
                    {key: item[key] for key in ['cost_basis', 'ticker', 'underlying_type', 'date_expiration', 'put_call', 'volume', 'strike_price']}
                    for item in highest_premium[0:4]
                ]

                highest_open_interest = sorted(options_flow, key=lambda x: int(x['open_interest']), reverse=True)
                highest_open_interest = [
                    {key: item[key] for key in ['cost_basis', 'ticker', 'underlying_type', 'date_expiration', 'put_call', 'open_interest', 'strike_price']}
                    for item in highest_open_interest[0:4]
                ]

                options_flow = {
                    'premium': highest_premium,
                    'volume': highest_volume,
                    'openInterest': highest_open_interest
                }
        except Exception as e:
            print(e)
            options_flow = {}

        market_status = check_market_hours()
        if market_status == 0:
            try:
                with open("json/market-movers/markethours/gainers.json", 'r') as file:
                    gainers = ujson.load(file)
                with open("json/market-movers/markethours/losers.json", 'r') as file:
                    losers = ujson.load(file)
                market_movers = {'gainers': gainers['1D'][:5], 'losers': losers['1D'][:5]}
            except:
                market_movers = {}
        elif market_status == 1:
            try:
                with open("json/market-movers/premarket/gainers.json", 'r') as file:
                    data = ujson.load(file)
                    gainers = [
                        {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                         'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                        for item in data[:5]
                    ]

                with open("json/market-movers/premarket/losers.json", 'r') as file:
                    data = ujson.load(file)
                    losers = [
                        {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                         'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                        for item in data[:5]
                    ]
        
                market_movers = {'gainers': gainers, 'losers': losers}
            except:
                market_movers = {}
        elif market_status == 2:
            try:
                with open("json/market-movers/afterhours/gainers.json", 'r') as file:
                    data = ujson.load(file)
                    gainers = [
                        {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                         'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                        for item in data[:5]
                    ]

                with open("json/market-movers/afterhours/losers.json", 'r') as file:
                    data = ujson.load(file)
                    losers = [
                        {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                         'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                        for item in data[:5]
                    ]
    
                market_movers = {'gainers': gainers, 'losers': losers}
            except:
                market_movers = {}

        data = {
            'marketMovers': market_movers,
            'marketStatus': market_status,
            'optionsFlow': options_flow,
            'recentEarnings': recent_earnings,
            'upcomingEarnings': upcoming_earnings,
            'analystReport': recent_analyst_report,
        }

        if len(data) > 0:
            await save_json(data)

try:

	con = sqlite3.connect('stocks.db')
	etf_con = sqlite3.connect('etf.db')

	cursor = con.cursor()
	cursor.execute("PRAGMA journal_mode = wal")
	cursor.execute("SELECT DISTINCT symbol FROM stocks")
	stock_symbols = [row[0] for row in cursor.fetchall()]

	etf_cursor = etf_con.cursor()
	etf_cursor.execute("PRAGMA journal_mode = wal")
	etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
	etf_symbols = [row[0] for row in etf_cursor.fetchall()]

	total_symbols = stock_symbols+etf_symbols
	asyncio.run(run())
	con.close()
	etf_con.close()

except Exception as e:
    print(e)