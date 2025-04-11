import requests
import time
import orjson
import aiohttp
import aiofiles
from datetime import datetime, timedelta, timezone, date
import pytz
import os
from dotenv import load_dotenv
import asyncio
import sqlite3
import hashlib

load_dotenv()

ny_tz = pytz.timezone("America/New_York")

today = datetime.utcnow().date()
now = datetime.now(timezone.utc)
now_ny = datetime.now(ny_tz)
N_minutes_ago = now - timedelta(minutes=30)

DARK_POOL_WEBHOOK_URL = os.getenv("DISCORD_DARK_POOL_WEBHOOK")
OPTIONS_FLOW_WEBHOOK_URL = os.getenv("DISCORD_OPTIONS_FLOW_WEBHOOK")
RECENT_EARNINGS_WEBHOOK_URL = os.getenv("DISCORD_RECENT_EARNINGS_WEBHOOK")
BENZINGA_API_KEY = os.getenv('BENZINGA_API_KEY')

headers = {"accept": "application/json"}

def save_json(data):
    directory = "json/discord"
    os.makedirs(directory, exist_ok=True)
    with open(directory+"/dark_pool.json", 'wb') as file:
        file.write(orjson.dumps(data))
    
def format_number(num, decimal=False):
    """Abbreviate large numbers with B/M suffix and format appropriately"""
    # Handle scientific notation formats like 5E6
    if isinstance(num, str) and ('e' in num.lower() or 'E' in num.lower()):
        try:
            num = float(num)
        except ValueError:
            return num  # Return as is if conversion fails
    
    # Convert strings to numbers if needed
    if isinstance(num, str):
        try:
            num = float(num)
            if num.is_integer():
                num = int(num)
        except ValueError:
            return num  # Return as is if conversion fails
            
    # Format based on size
    if num >= 1_000_000_000:  # Billions
        formatted = num / 1_000_000_000
        # Only show decimal places if needed
        return f"{formatted:.2f}B".rstrip('0').rstrip('.') + 'B'
    elif num >= 1_000_000:  # Millions
        formatted = num / 1_000_000
        # Only show decimal places if needed
        return f"{formatted:.2f}".rstrip('0').rstrip('.') + 'M'
    elif decimal and isinstance(num, float) and not num.is_integer():
        return f"{num:,.2f}"
    else:
        return f"{num:,.0f}"  # Format smaller numbers with commas

def abbreviate_number(n):
    """
    Abbreviate a number to a readable format.
    E.g., 1500 -> '1.5K', 2300000 -> '2.3M'
    """
    if n is None:
        return "N/A"
    abs_n = abs(n)
    if abs_n < 1000:
        return str(n)
    elif abs_n < 1_000_000:
        return f"{n/1000:.1f}K"
    elif abs_n < 1_000_000_000:
        return f"{n/1_000_000:.1f}M"
    else:
        return f"{n/1_000_000_000:.1f}B"

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


async def get_recent_earnings(session):
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = weekday()

    url = "https://api.benzinga.com/api/v2.1/calendar/earnings"
    res_list = []
    importance_list = ["1","2","3","4","5"]
    
    for importance in importance_list:
        querystring = {
            "token": BENZINGA_API_KEY,
            "parameters[importance]": importance, 
            "parameters[date_from]": yesterday,
            "parameters[date_to]": today,
            "parameters[date_sort]": "date"
        }
        try:
            async with session.get(url, params=querystring, headers=headers) as response:
                res = orjson.loads(await response.text())['earnings']
                for item in res:
                    try:
                        symbol = item['ticker']
                        name = item['name']
                        time = item['time']
                        date = item['date']
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
                            price = quote_data.get('price',0)
                            changes_percentage = quote_data.get('changesPercentage',0)
                            
                            res_list.append({
                                'symbol': symbol,
                                'name': name,
                                'time': time,
                                'date': date,
                                'marketCap': market_cap,
                                'epsPrior': eps_prior,
                                'epsSurprise': eps_surprise,
                                'eps': eps,
                                'revenuePrior': revenue_prior,
                                'revenueSurprise': revenue_surprise,
                                'revenue': revenue,
                                'price': price,
                                'changesPercentage': changes_percentage,
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
    res_list = [{k: v for k, v in d.items() if k not in ['updated']} for d in res_list]
    
    return res_list[:10]



def dark_pool_flow():

    try:
        with open(f"json/discord/dark_pool.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    with open(f"json/dark-pool/feed/data.json", "r") as file:
        res_list = orjson.loads(file.read())
        res_list = [item for item in res_list if float(item['premium']) >= 100E6 and datetime.fromisoformat(item['date']).date() == today]


    if res_list:
        filtered_recent = [
            item for item in res_list
            if (dt := datetime.fromisoformat(item['date'])) >= N_minutes_ago
        ]
        # If there are any recent orders, find the one with the highest premium
        if filtered_recent:
            best_order = max(filtered_recent, key=lambda x: x['premium'])
            result = {k: best_order[k] for k in ['date', 'trackingID', 'price', 'size', 'premium', 'ticker']}
        else:
            result = None  # Or handle however you like (e.g., empty dict)


        if seen_list:
            seen_ids = {item['trackingID'] for item in seen_list}
        else:
            seen_ids = {}

        if result != None and result['trackingID'] not in seen_ids:
            symbol = result['ticker']
            quantity = format_number(result['size'])
            price = result['price']
            amount = format_number(result['premium'])

            
            message_timestamp = int((datetime.now() - timedelta(minutes=0)).timestamp())

            embed = {
                "color": 0xC475FD, 
                "thumbnail": {"url": "https://stocknear.com/pwa-64x64.png"},
                "title": "Dark Pool Order",
                "fields": [
                    {"name": "Symbol", "value": symbol, "inline": True},
                    {"name": "", "value": "", "inline": True},
                    {"name": "Quantity", "value": str(quantity), "inline": True},
                    {"name": "Price", "value": str(price), "inline": True},
                    {"name": "", "value": "\u200B                                                                       \u200B", "inline": True},
                    {"name": "Amount", "value": "$"+amount, "inline": True},
                    {"name": "", "value": "", "inline": False},
                    {"name": f"Data by Stocknear - <t:{message_timestamp}:R> - Delayed by 15 min.", "value": "", "inline": False}
                ],
                "footer": {"text": ""}
            }

            payload = {
                "content": "",
                "embeds": [embed]
            }

            response = requests.post(DARK_POOL_WEBHOOK_URL, json=payload)

            if response.status_code in (200, 204):
                seen_list.append({'date': result['date'], 'trackingID': result['trackingID']})
                with open("json/discord/dark_pool.json","wb") as file:
                    file.write(orjson.dumps(seen_list))
                print("Embed sent successfully!")

            else:
                print(f"Failed to send embed. Status code: {response.status_code}")
                print("Response content:", response.text)
        
        else:
            print("Dark pool already sent!")


def options_flow():
    now_ny = datetime.now(ny_tz)
    N_minutes_ago = now_ny - timedelta(minutes=5)
    today = now_ny.date()

    # Load seen entries
    try:
        with open("json/discord/options_flow.json", "rb") as file:
            seen_list = orjson.loads(file.read())
            today_iso = today.isoformat()
            seen_list = [item for item in seen_list if item['date'] == today_iso]
    except FileNotFoundError:
        seen_list = []
    except Exception as e:
        print(f"Error loading seen list: {e}")
        seen_list = []

    # Load current data
    try:
        with open("json/options-flow/feed/data.json", "rb") as file:
            res_list = orjson.loads(file.read())
    except Exception as e:
        print(f"Error loading data.json: {e}")
        res_list = []

    # Process and filter entries
    filtered = []
    for item in res_list:
        try:
            # Validate required fields
            if float(item.get('cost_basis', 0)) < 1E5:
                continue
                
            item_date = datetime.fromisoformat(item['date']).date()
            if item_date != today:
                continue
            
            item_time = datetime.strptime(item['time'], "%H:%M:%S").time()
            item_dt = ny_tz.localize(datetime.combine(item_date, item_time))
            if item_dt < N_minutes_ago:
                filtered.append(item)
        except Exception as e:
            print(f"Error processing item {item.get('id')}: {e}")
            continue

    if not filtered:
        print("No recent valid entries found")
        return

    # Find best order
    best_order = max(filtered, key=lambda x: float(x['cost_basis']))
    result = {
        k: best_order[k] for k in [
            'date', 'sentiment', 'option_activity_type', 'ticker', 'id',
            'strike_price', 'date_expiration', 'size', 'cost_basis',
            'execution_estimate', 'volume', 'open_interest', 'put_call'
        ]
    }

    # Check duplicates
    seen_ids = {item['id'] for item in seen_list}
    if result['id'] in seen_ids:
        print("Options Flow data already sent!")
        return

    # Prepare message
    try:
        symbol = result['ticker']
        size = format_number(result['size'])
        premium = format_number(result['cost_basis'])
        strike = result['strike_price']
        side = result['execution_estimate']
        volume = format_number(result['volume'])
        open_interest = format_number(result['open_interest'])
        put_call = result['put_call'].replace('Calls', 'Call').replace('Puts', 'Put')
        option_activity_type = result['option_activity_type']
        sentiment = result['sentiment']

        date_expiration = datetime.strptime(result['date_expiration'], "%Y-%m-%d").strftime("%d/%m/%Y")
        message_timestamp = int(now_ny.timestamp())

        color = 0x39FF14 if sentiment == 'Bullish' else 0xFF0000 if sentiment == 'Bearish' else 0xFFA500

        embed = {
            "color": color,
            "thumbnail": {"url": "https://stocknear.com/pwa-64x64.png"},
            "description": f"{put_call} {option_activity_type} ({sentiment})",
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Strike", "value": str(strike), "inline": True},
                {"name": "Expiration", "value": date_expiration, "inline": True},
                {"name": "Call/Put", "value": put_call, "inline": True},
                {"name": "Side", "value": str(side), "inline": True},
                {"name": "Size", "value": str(size), "inline": True},
                {"name": "Premium", "value": f"${premium}", "inline": True},
                {"name": "Volume", "value": str(volume), "inline": True},
                {"name": "OI", "value": str(open_interest), "inline": True},
                {"name": f"Data by Stocknear - <t:{message_timestamp}:R> - Delayed by 5 min", 
                 "value": "", "inline": False}
            ],
            "footer": {"text": ""}
        }

        payload = {"content": "", "embeds": [embed]}

        # Send to Discord
        response = requests.post(OPTIONS_FLOW_WEBHOOK_URL, json=payload)
        response.raise_for_status()

        # Update seen list
        seen_list.append({'date': result['date'], 'id': result['id']})
        with open("json/discord/options_flow.json", "wb") as file:
            file.write(orjson.dumps(seen_list))
        print("Embed sent successfully!")

    except Exception as e:
        print(f"Error sending message: {e}")

async def recent_earnings_message():

    try:
        with open(f"json/discord/recent_earnings.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    async with aiohttp.ClientSession() as session:
        res_list = await get_recent_earnings(session)
        for item in res_list:
            
            unique_str = f"{item['date']}-{item['symbol']}-{item['revenue']}-{item['eps']}"
            item['id'] = hashlib.md5(unique_str.encode()).hexdigest()

    
    if res_list:

        if seen_list:
            seen_ids = {item['id'] for item in seen_list}
        else:
            seen_ids = {}

        for item in res_list:
            try:
                if item != None and item['id'] not in seen_ids and item['marketCap'] > 100E9:
                    symbol = item['symbol']
                    price = item['price']
                    changes_percentage = round(item['changesPercentage'],2)
                    revenue = abbreviate_number(item['revenue'])
                    revenue_surprise = abbreviate_number(item.get("revenueSurprise", 0))
                    eps = item['eps']
                    eps_surprise = item.get("epsSurprise", 0)
                    revenue_surprise_text = "exceeds" if item["revenueSurprise"] > 0 else "misses"
                    eps_surprise_text = "exceeds" if eps_surprise > 0 else "misses"

                    market_cap = abbreviate_number(item['marketCap'])

                    revenue_yoy_change = (item["revenue"] / item["revenuePrior"] - 1) * 100
                    revenue_yoy_direction = "decline" if (item["revenue"] / item["revenuePrior"] - 1) < 0 else "growth"

                    eps_yoy_change = (item["eps"] / item["epsPrior"] - 1) * 100
                    eps_yoy_direction = "decline" if (item["eps"] / item["epsPrior"] - 1) < 0 else "growth"
                    
                    message_timestamp = int((datetime.now() - timedelta(minutes=0)).timestamp())

                    embed = {
                        "color": 0xC475FD,
                        "thumbnail": {"url": "https://stocknear.com/pwa-64x64.png"},
                        "title": "Earnings Surprise",
                        "fields": [
                            {"name": "Symbol", "value": symbol, "inline": True},
                            {"name": "", "value": "", "inline": True},
                            {"name": "Market Cap", "value": market_cap, "inline": True},
                            {"name": "Price", "value": str(price), "inline": True},
                            {"name": "", "value": "", "inline": True},
                            {"name": "% Change", "value": str(changes_percentage)+"%", "inline": True},
                            {"name": "", "value": "", "inline": False},
                            {"name": f"Revenue of {revenue} {revenue_surprise_text} estimates by {revenue_surprise}, with {revenue_yoy_change:.2f}% YoY {revenue_yoy_direction}.", "value": "", "inline": False},
                            {"name": f"EPS of {eps} {eps_surprise_text} estimates by {eps_surprise}, with {eps_yoy_change:.2f}% YoY {eps_yoy_direction}.", "value": "", "inline": False},
                            {"name": f"Data by Stocknear - <t:{message_timestamp}:R>", "value": "", "inline": False},
                        ],
                        "footer": {"text": ""}
                    }

                    payload = {
                        "content": "",
                        "embeds": [embed]
                    }

                    response = requests.post(RECENT_EARNINGS_WEBHOOK_URL, json=payload)

                    if response.status_code in (200, 204):
                        seen_list.append({'date': item['date'], 'id': item['id'], 'symbol': symbol})
                        print("Embed sent successfully!")

                    else:
                        print(f"Failed to send embed. Status code: {response.status_code}")
                        print("Response content:", response.text)
                
                else:
                    print("Earnings already sent!")


            except Exception as e:
                print(e)
        try:
            with open("json/discord/recent_earnings.json","wb") as file:
                file.write(orjson.dumps(seen_list))
        except:
            pass


async def main():
    options_flow()
    dark_pool_flow()
    await recent_earnings_message()

try:
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    asyncio.run(main())
 
except Exception as e:
    print(e)