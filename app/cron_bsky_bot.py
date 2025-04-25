from atproto import Client
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone, date
import requests
import hashlib
import orjson
import pytz

load_dotenv()

API_KEY = os.getenv('BENZINGA_API_KEY')
BSKY_USERNAME = os.getenv('BSYK_USERNAME')
BSKY_SECRET = os.getenv('BSKY_SECRET')

client = Client()
client.login(BSKY_USERNAME, BSKY_SECRET)

ny_tz = pytz.timezone("America/New_York")

today = datetime.utcnow().date()
now = datetime.now(timezone.utc)
now_ny = datetime.now(ny_tz)
N_minutes_ago = now - timedelta(minutes=30)


def save_json(data, file):
    directory = "json/bsky"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{file}.json", 'wb') as f:
        f.write(orjson.dumps(data))


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


def wiim():
    try:
        with open("json/bsky/wiim.json", "r") as f:
            seen_list = orjson.loads(f.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except Exception:
        seen_list = []

    with open("json/dashboard/data.json", 'rb') as f:
        data = orjson.loads(f.read())['wiim']
        data = [item for item in data if datetime.fromisoformat(item['date']).date() == today]

    res_list = []
    for item in data:
        try:
            unique_str = f"{item['date']}-{item['ticker']}-{item.get('text', '')}"
            item['id'] = hashlib.md5(unique_str.encode()).hexdigest()
            res_list.append(item)
        except Exception:
            pass

    if res_list:
        if seen_list:
            seen_ids = {item['id'] for item in seen_list}
        else:
            seen_ids = set()

        for item in res_list:
            try:
                if item is not None and item['id'] not in seen_ids:
                    symbol = item['ticker']
                    description = item.get('text', '')
                    message = f"${symbol} {description}"
                    post = client.send_post(message)

                    seen_list.append({'date': item['date'], 'id': item['id'], 'symbol': symbol})
                else:
                    print("WIIM already posted!")
            except Exception as e:
                print(e)

        try:
            save_json(seen_list, "wiim")
        except Exception as e:
            print(e)


def recent_earnings():

    try:
        with open("json/bsky/recent_earnings.json", "r") as file:
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
                item['marketCap'] = quote_data.get('marketCap',0)
                item['eps'] = quote_data.get('eps',0)

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
                    revenue = abbreviate_number(item['revenue'])
                    revenue_surprise = abbreviate_number(item.get("revenueSurprise", 0))
                    eps = item['eps']
                    eps_surprise = item.get("epsSurprise", 0)
                    revenue_surprise_text = "exceeds" if item["revenueSurprise"] > 0 else "misses"
                    eps_surprise_text = "exceeds" if eps_surprise > 0 else "misses"

                    revenue_yoy_change = (item["revenue"] / item["revenuePrior"] - 1) * 100
                    revenue_yoy_direction = "decline" if (item["revenue"] / item["revenuePrior"] - 1) < 0 else "growth"

                    eps_yoy_change = (item["eps"] / item["epsPrior"] - 1) * 100
                    eps_yoy_direction = "decline" if (item["eps"] / item["epsPrior"] - 1) < 0 else "growth"
                    
                    message_timestamp = int((datetime.now() - timedelta(minutes=0)).timestamp())


                    message = f"${symbol} Earnings Release 📝\n"
                    message +=f"· Revenue of {revenue} {revenue_surprise_text} estimates by {revenue_surprise}, with {revenue_yoy_change:.2f}% YoY {revenue_yoy_direction}. \n"
                    message +=f"· EPS of {eps} {eps_surprise_text} estimates by {eps_surprise}, with {eps_yoy_change:.2f}% YoY {eps_yoy_direction}."
                    message +=f"\n \n"
                    message+=f"Follow up here stocknear.com/stocks/{symbol}"
                    
                    post = client.send_post(message)

                    seen_list.append({'date': item['date'], 'id': item['id'], 'symbol': symbol})
                else:
                    print("Recent Earnings already posted!")
            except Exception as e:
                print(e)

            except Exception as e:
                print(e)
        try:
            with open("json/bsky/recent_earnings.json","wb") as file:
                file.write(orjson.dumps(seen_list))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    wiim()
    recent_earnings()
