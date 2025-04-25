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
import glob

load_dotenv()

ny_tz = pytz.timezone("America/New_York")

today = datetime.utcnow().date()
now = datetime.now(timezone.utc)
now_ny = datetime.now(ny_tz)
N_minutes_ago = now - timedelta(minutes=30)

DARK_POOL_WEBHOOK_URL = os.getenv("DISCORD_DARK_POOL_WEBHOOK")
OPTIONS_FLOW_WEBHOOK_URL = os.getenv("DISCORD_OPTIONS_FLOW_WEBHOOK")
RECENT_EARNINGS_WEBHOOK_URL = os.getenv("DISCORD_RECENT_EARNINGS_WEBHOOK")
EXECUTIVE_ORDER_WEBHOOK_URL = os.getenv("DISCORD_EXECUTIVE_ORDER_WEBHOOK")
ANALYST_REPORT_WEBHOOK_URL = os.getenv("DISCORD_ANALYST_REPORT_WEBHOOK")
WIIM_WEBHOOK_URL = os.getenv('DISCORD_WIIM_WEBHOOK')
CONGRESS_TRADING_WEBHOOK_URL = os.getenv("DISCORD_CONGRESS_TRADING_WEBHOOK")
TRUTH_SOCIAL_WEBHOOK_URL = os.getenv("DISCORD_TRUTH_SOCIAL_WEBHOOK")


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

def extract_first_value(s):
    """
    Extract the first value from a string like "$1K-$15K" or "$1M-$5M"
    and return it as an integer.
    """
    s = s.upper().replace('$', '')
    first_part = s.split('-')[0]

    if 'K' in first_part:
        return int(float(first_part.replace('K', '')) * 1_000)
    elif 'M' in first_part:
        return int(float(first_part.replace('M', '')) * 1_000_000)
    else:
        # If no K or M, just try converting directly
        return int(first_part)

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
    N_minutes_ago = now_ny - timedelta(minutes=10)
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


def recent_earnings():

    try:
        with open(f"json/discord/recent_earnings.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    with open(f"json/dashboard/data.json", 'rb') as file:
        data = orjson.loads(file.read())['recentEarnings']
        data = [item for item in data if datetime.fromisoformat(item['date']).date() == today]

    res_list = []
    for item in data:
        try:
            with open(f"json/quote/{item['symbol']}.json","r") as file:
                quote_data = orjson.loads(file.read())

                item['price'] = round(quote_data.get('price',0),2)
                item['changesPercentage'] = round(quote_data.get('changesPercentage',0),2)
                item['marketCap'] = quote_data.get('marketCap',0)
                item['eps'] = round(quote_data.get('eps',0),2)

            unique_str = f"{item['date']}-{item['symbol']}-{item.get('revenueSurprise',0)}-{item.get('epsSurprise',0)}"
            item['id'] = hashlib.md5(unique_str.encode()).hexdigest()
            
            if item['marketCap'] > 1E9:
                res_list.append(item)
        except:
            pass
    
    if res_list:

        if seen_list:
            seen_ids = {item['id'] for item in seen_list}
        else:
            seen_ids = {}

        for item in res_list:
            try:
                if item != None and item['id'] not in seen_ids and item['marketCap']:
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
        except Exception as e:
            print(e)

def executive_order():

    try:
        with open(f"json/discord/executive_order.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    res_list = []
    json_files = glob.glob("json/executive-orders/*.json")
    for filepath in json_files:
        try:
            file_id = os.path.basename(filepath)
            
            with open(filepath, "r") as file:
                # Load the JSON data from the file using orjson
                data = orjson.loads(file.read())
                
                if datetime.fromisoformat(data['date']).date() == today:
                    data['id'] = file_id.replace(".json","")

                res_list.append(data)
        except:
            pass

    if res_list:
    
        if seen_list:
            seen_ids = {item['id'] for item in seen_list}
        else:
            seen_ids = {}

        for item in res_list:
            try:
                if item['id'] not in seen_ids:
                    title = item['title']
                    description = item['description']
                    date = item['date']
                    date = datetime.strptime(date, "%Y-%m-%d")
                    date = date.strftime("%B %d, %Y")
                    source = f"[Source]({item['link']})"

                    message_timestamp = int((datetime.now() - timedelta(minutes=0)).timestamp())
                    

                    embed = {
                        "color": 0xFF0000, 
                        "thumbnail": {"url": "https://stocknear.com/pwa-64x64.png"},
                        "title": title,
                        "description": f"Signed on {date}",
                        "fields": [
                            {"name": "", "value": description, "inline": False},
                            {"name": "", "value": source, "inline": False},
                            {"name": f"Data by Stocknear - <t:{message_timestamp}:R>", "value": "", "inline": False}
                        ],
                        "footer": {"text": ""}
                    }

                    payload = {
                        "content": "",
                        "embeds": [embed]
                    }

                    response = requests.post(EXECUTIVE_ORDER_WEBHOOK_URL, json=payload)

                    if response.status_code in (200, 204):
                        seen_list.append({'date': item['date'], 'id': item['id']})
                        with open("json/discord/executive_order.json","wb") as file:
                            file.write(orjson.dumps(seen_list))
                        print("Embed sent successfully!")

                    else:
                        print(f"Failed to send embed. Status code: {response.status_code}")
                        print("Response content:", response.text)
        
                else:
                    print("Executive Order already sent!")

            except:
                pass


def analyst_report():

    try:
        with open(f"json/discord/analyst_report.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    with open(f"json/dashboard/data.json", 'rb') as file:
        data = orjson.loads(file.read())['analystReport']
        date_obj = datetime.strptime(data['date'], '%b %d, %Y')
        data['date'] = date_obj.strftime('%Y-%m-%d')

    if datetime.fromisoformat(data['date']).date() == today:
    
        if seen_list:
            seen_ids = {item['id'] for item in seen_list}
        else:
            seen_ids = {}

        if data['id'] not in seen_ids:
            symbol = data['symbol']
            insight = data['insight']
            message_timestamp = int((datetime.now() - timedelta(minutes=0)).timestamp())

            with open(f"json/quote/{symbol}.json","r") as file:
                quote_data = orjson.loads(file.read())

            market_cap = abbreviate_number(quote_data.get('marketCap',0))
            eps = round(quote_data.get('eps',0),2)
            changes_percentage = round(quote_data.get('changesPercentage',0),2)
            summary = (
                f"According to {data['numOfAnalyst']} analyst ratings, the average rating for {symbol} stock is '{data['consensusRating']}'. "
                f"The 12-month stock price forecast is {data['highPriceTarget']}, which is an "
                f"{'increase' if data['highPriceChange'] > 0 else 'decrease'} of {abs(data['highPriceChange'])}% from the latest price."
            )

            embed = {
                "color": 0xFFA500, 
                "thumbnail": {"url": "https://stocknear.com/pwa-64x64.png"},
                "title": "",
                "fields": [
                    {"name": "Symbol", "value": symbol, "inline": True},
                    {"name": "", "value": "", "inline": True},
                    {"name": "Market Cap", "value": str(market_cap), "inline": True},
                    {"name": "EPS", "value": str(eps), "inline": True},
                    {"name": "", "value": "", "inline": True},
                    {"name": "% Change", "value": str(changes_percentage)+"%", "inline": True},
                    {"name": "Analyst Insight Report", "value": insight, "inline": False},
                    {"name": "", "value": summary, "inline": False},
                    {"name": f"Data by Stocknear - <t:{message_timestamp}:R>", "value": "", "inline": False}
                ],
                "footer": {"text": ""}
            }
            payload = {
                "content": "",
                "embeds": [embed]
            }

            response = requests.post(ANALYST_REPORT_WEBHOOK_URL, json=payload)

            if response.status_code in (200, 204):
                seen_list.append({'date': data['date'], 'id': data['id']})
                with open("json/discord/analyst_report.json","wb") as file:
                    file.write(orjson.dumps(seen_list))
                print("Embed sent successfully!")

            else:
                print(f"Failed to send embed. Status code: {response.status_code}")
                print("Response content:", response.text)
        
        else:
            print("Analyst Report already sent!")


def wiim():
    try:
        with open(f"json/discord/wiim.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    with open(f"json/dashboard/data.json", 'rb') as file:
        data = orjson.loads(file.read())['wiim']
        data = [item for item in data if datetime.fromisoformat(item['date']).date() == today]

    res_list = []
    for item in data:
        try:
            with open(f"json/quote/{item['ticker']}.json","r") as file:
                quote_data = orjson.loads(file.read())

                item['price'] = round(quote_data.get('price',0),2)
                item['changesPercentage'] = round(quote_data.get('changesPercentage',0),2)
                item['marketCap'] = quote_data.get('marketCap',0)
                item['eps'] = round(quote_data.get('eps',0),2)

            unique_str = f"{item['date']}-{item['ticker']}-{item.get('text','')}"
            item['id'] = hashlib.md5(unique_str.encode()).hexdigest()
            
            if item['marketCap'] > 50E9:
                res_list.append(item)
        except:
            pass
    
    if res_list:

        if seen_list:
            seen_ids = {item['id'] for item in seen_list}
        else:
            seen_ids = {}

        for item in res_list:
            try:
                if item != None and item['id'] not in seen_ids and item['marketCap']:
                    symbol = item['ticker']
                    price = item['price']
                    changes_percentage = round(item['changesPercentage'],2)
                    eps = item['eps']
                    market_cap = abbreviate_number(item['marketCap'])
                    description = item['text']
                    message_timestamp = int((datetime.now() - timedelta(minutes=0)).timestamp())

                    color = 0x39FF14 if 'higher' in description else 0xFF0000 if 'lower' in description else 0xFFFFFF

                    embed = {
                        "color": color,
                        "thumbnail": {"url": "https://stocknear.com/pwa-64x64.png"},
                        "title": "Why Priced Moved",
                        "fields": [
                            {"name": "Symbol", "value": symbol, "inline": True},
                            {"name": "", "value": "", "inline": True},
                            {"name": "Market Cap", "value": market_cap, "inline": True},
                            {"name": "Price", "value": str(price), "inline": True},
                            {"name": "", "value": "", "inline": True},
                            {"name": "% Change", "value": str(changes_percentage)+"%", "inline": True},
                            {"name": "", "value": "", "inline": False},
                            {"name": f"{description}", "value": "", "inline": False},
                            {"name": f"Data by Stocknear - <t:{message_timestamp}:R>", "value": "", "inline": False},
                        ],
                        "footer": {"text": ""}
                    }

                    payload = {
                        "content": "",
                        "embeds": [embed]
                    }

                    response = requests.post(WIIM_WEBHOOK_URL, json=payload)

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
            with open("json/discord/wiim.json","wb") as file:
                file.write(orjson.dumps(seen_list))
        except Exception as e:
            print(e)


def congress_trading():
    try:
        with open(f"json/discord/congress_trading.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    with open(f"json/congress-trading/rss-feed/data.json", 'rb') as file:
        data = orjson.loads(file.read())
        data = [item for item in data if datetime.fromisoformat(item['disclosureDate']).date() == today]

    res_list = []
    for item in data:
        try:
            with open(f"json/quote/{item['ticker']}.json","r") as file:
                quote_data = orjson.loads(file.read())

                item['name'] = quote_data.get('name','n/a')

            item['amountInt'] = extract_first_value(item['amount'])
            unique_str = f"{item['disclosureDate']}-{item['transactionDate']}-{item['ticker']}-{item['amount']}-{item['representative']}"
            item['id'] = hashlib.md5(unique_str.encode()).hexdigest()
            
            if item['amountInt'] >= 15_000:
                res_list.append(item)

        except Exception as e:
            print(e)
    
    if res_list:

        if seen_list:
            seen_ids = {item['id'] for item in seen_list}
        else:
            seen_ids = {}

        for item in res_list:
            try:
                if item != None and item['id'] not in seen_ids:
                    symbol = item['ticker']
                    name = item['name']
                    representative = item['representative']
                    amount = item['amount']
                    transaction_type = item['type']
                    transaction_date = datetime.strptime(item['transactionDate'], "%Y-%m-%d").strftime("%d/%m/%Y")
                    disclosure_date = datetime.strptime(item['disclosureDate'], "%Y-%m-%d").strftime("%d/%m/%Y")

                    message_timestamp = int((datetime.now() - timedelta(minutes=0)).timestamp())

                    color = 0x39FF14 if transaction_type == 'Bought' else 0xFF0000

                    embed = {
                        "color": color,
                        "thumbnail": {"url": "https://stocknear.com/pwa-64x64.png"},
                        "title": "Congress Trading",
                        "fields": [
                            {"name": "Politician", "value": representative, "inline": True},
                            {"name": "", "value": "", "inline": True},
                            {"name": "Amount", "value": amount, "inline": True},
                            {"name": "Symbol", "value": symbol, "inline": True},
                            {"name": "", "value": "", "inline": True},
                            {"name": "Side", "value": transaction_type, "inline": True},
                            {"name": "Trade Date", "value": transaction_date, "inline": True},
                            {"name": "", "value": "", "inline": True},
                            {"name": "Filing Date", "value": disclosure_date, "inline": True},
                            {"name": "", "value": "", "inline": False},
                            {"name": f"Data by Stocknear - <t:{message_timestamp}:R>", "value": "", "inline": False},
                        ],
                        "footer": {"text": ""}
                    }

                    payload = {
                        "content": "",
                        "embeds": [embed]
                    }

                    response = requests.post(CONGRESS_TRADING_WEBHOOK_URL, json=payload)

                    if response.status_code in (200, 204):
                        seen_list.append({'date': item['disclosureDate'], 'id': item['id'], 'symbol': symbol})
                        print("Embed sent successfully!")

                    else:
                        print(f"Failed to send embed. Status code: {response.status_code}")
                        print("Response content:", response.text)
                
                else:
                    print("Congress Data already sent!")


            except Exception as e:
                print(e)
        try:
            with open("json/discord/congress_trading.json","wb") as file:
                file.write(orjson.dumps(seen_list))
        except Exception as e:
            print(e)



def truth_social():

    try:
        with open(f"json/discord/truth_social.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    with open(f"json/tracker/potus/data.json", 'rb') as file:
        data = orjson.loads(file.read())['posts']

    for item in data:
        dt = datetime.strptime(item["date"], "%B %d, %Y, %I:%M %p")
        item['time'] = item['date']
        item["date"] = dt.strftime("%Y-%m-%d")

    data = [item for item in data if datetime.fromisoformat(item['date']).date() == today]

    res_list = []
    for item in data:
        try:
            unique_str = f"{item['date']}-{item['content']}"
            item['id'] = hashlib.md5(unique_str.encode()).hexdigest()
            
            if item['externalLink'] == "" and len(item['content']) > 0:
                res_list.append(item)
        except Exception as e:
            print(e)

    if res_list:
    
        if seen_list:
            seen_ids = {item['id'] for item in seen_list}
        else:
            seen_ids = {}

        for item in res_list:
            try:
                if item['id'] not in seen_ids:
                    time = item['time']
                    description = item['content']
                    source = f"[Source]({item['source']})"
                    message_timestamp = int((datetime.now() - timedelta(minutes=0)).timestamp())
                    

                    embed = {
                        "color": 0x5448EE, 
                        "thumbnail": {"url": "https://stocknear.com/pwa-64x64.png"},
                        "title": "Donald J. Trump",
                        "description": time,
                        "fields": [
                            {"name": "", "value": description, "inline": False},
                            {"name": "", "value": source, "inline": False},
                            {"name": f"Data by Stocknear - <t:{message_timestamp}:R>", "value": "", "inline": False}
                        ],
                        "footer": {"text": ""}
                    }

                    payload = {
                        "content": "",
                        "embeds": [embed]
                    }

                    response = requests.post(TRUTH_SOCIAL_WEBHOOK_URL, json=payload)

                    if response.status_code in (200, 204):
                        seen_list.append({'date': item['date'], 'id': item['id']})
                        with open("json/discord/truth_social.json","wb") as file:
                            file.write(orjson.dumps(seen_list))
                        print("Embed sent successfully!")

                    else:
                        print(f"Failed to send embed. Status code: {response.status_code}")
                        print("Response content:", response.text)
        
                else:
                    print("Truth Social Post already sent!")

            except:
                pass


if __name__ == "__main__":
    options_flow()
    dark_pool_flow()
    recent_earnings()
    analyst_report()
    wiim()
    congress_trading()
    executive_order()
    truth_social()