from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta, timezone, date
import hashlib
import orjson
import pytz

load_dotenv()

LONG_LIVED_USER_TOKEN = os.getenv('FACEBOOK_ACCESS_TOKEN')
PAGE_ID = os.getenv('FACEBOOK_PAGE_ID')
GRAPH_BASE = "https://graph.facebook.com/v23.0"

ny_tz = pytz.timezone("America/New_York")

today = datetime.utcnow().date()
now = datetime.now(timezone.utc)
now_ny = datetime.now(ny_tz)



# Ensure folder exists
folder_path = os.path.join("json", "facebook")
os.makedirs(folder_path, exist_ok=True)

def get_page_token(long_lived_user_token, page_id):
    url = f"{GRAPH_BASE}/me/accounts"
    params = {"access_token": long_lived_user_token}
    resp = requests.get(url, params=params).json()
    for page in resp.get("data", []):
        if page["id"] == page_id:
            return page["access_token"]
    raise Exception("Page token not found!")

PAGE_TOKEN = get_page_token(LONG_LIVED_USER_TOKEN, PAGE_ID)

def save_json(data, file):
    directory = "json/facebook"
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


def send_post(message):
    """Send a post to Facebook page"""
    try:
        url = f"https://graph.facebook.com/{PAGE_ID}/feed"
        
        payload = {
            "message": message,
            "access_token": PAGE_TOKEN
        }
        
        response = requests.post(url, data=payload)
        result = response.json()
        
        if 'id' in result:
            post_id = result['id']
            print(f"Post sent successfully! ID: {post_id}")
            print(f"Post content: {message[:50]}...")
            return post_id
        else:
            print(f"Error in response: {result}")
            raise Exception(f"Facebook API error: {result}")
            
    except requests.exceptions.RequestException as e:
        print(f"Network error sending post: {e}")
        raise
    except Exception as e:
        print(f"Error sending post: {e}")
        raise


def wiim():
    try:
        with open("json/facebook/wiim.json", "r") as f:
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
                    asset_type = item['assetType']
                    description = item.get('text', '')
                    message = f"${symbol} {description}"
                    message +=f"\n \n"
                    message+=f"Follow up here stocknear.com/{asset_type}/{symbol}"
                    send_post(message)

                    seen_list.append({'date': item['date'], 'id': item['id'], 'symbol': symbol})
                else:
                    print("WIIM already posted!")
            except Exception as e:
                print(e)

        try:
            save_json(seen_list, "wiim")
        except Exception as e:
            print(e)


def dark_pool_flow():
    N_minutes_ago = now_ny - timedelta(minutes=30)
    try:
        with open("json/facebook/dark_pool.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    with open("json/dark-pool/feed/data.json", "r") as file:
        res_list = orjson.loads(file.read())
        res_list = [item for item in res_list if float(item['premium']) >= 100E6 and datetime.fromisoformat(item['date']).date() == today]
    if res_list:
        filtered_recent = [
            item for item in res_list
            if (dt := datetime.fromisoformat(item['date'])) >= N_minutes_ago
        ]

        # If there are any recent orders, find the one with the highest premium
        if filtered_recent:
            best_order = max(filtered_recent, key=lambda x: float(x['premium']))
            result = {k: best_order[k] for k in ['date', 'trackingID', 'price', 'size', 'premium', 'ticker']}
        else:
            result = None

        if seen_list:
            seen_ids = {item['trackingID'] for item in seen_list}
        else:
            seen_ids = {}

        if result != None and result['trackingID'] not in seen_ids:
            symbol = result['ticker']
            quantity = format_number(result['size'])
            price = result['price']
            amount = format_number(result['premium'])

            message = f"Unusual Dark Pool Order Alert for ${symbol}:\n\n"
            message += f"• Quantity: {quantity}\n"
            message += f"• Price: ${price}\n"
            message += f"• Amount: ${amount}\n\n"
            message += f"Follow up: stocknear.com/dark-pool-flow"
            
            try:
                send_post(message)
                seen_list.append({'date': result['date'], 'trackingID': result['trackingID']})
                save_json(seen_list, "dark_pool")
                print("Dark pool post sent successfully!")
            except Exception as e:
                print(f"Error sending dark pool post: {e}")
        else:
            print("Dark pool already posted!")


def options_flow():
    now_ny = datetime.now(ny_tz)
    N_minutes_ago_options = now_ny - timedelta(minutes=10)
    today_ny = now_ny.date()

    try:
        with open("json/facebook/options_flow.json", "rb") as file:
            seen_list = orjson.loads(file.read())
            today_iso = today_ny.isoformat()
            seen_list = [item for item in seen_list if item['date'] == today_iso]
    except FileNotFoundError:
        seen_list = []
    except Exception as e:
        print(f"Error loading seen list: {e}")
        seen_list = []

    try:
        with open("json/options-flow/feed/data.json", "rb") as file:
            res_list = orjson.loads(file.read())
    except Exception as e:
        print(f"Error loading data.json: {e}")
        res_list = []

    filtered = []
    for item in res_list:
        try:
            if float(item.get('cost_basis', 0)) < 1E5:
                continue
                
            item_date = datetime.fromisoformat(item['date']).date()
            if item_date != today_ny:
                continue
            
            item_time = datetime.strptime(item['time'], "%H:%M:%S").time()
            item_dt = ny_tz.localize(datetime.combine(item_date, item_time))
            if item_dt < N_minutes_ago_options:
                filtered.append(item)
        except Exception as e:
            print(f"Error processing item {item.get('id')}: {e}")
            continue

    if not filtered:
        print("No recent valid entries found")
        return

    best_order = max(filtered, key=lambda x: float(x['cost_basis']))
    result = {
        k: best_order[k] for k in [
            'date', 'sentiment', 'option_activity_type', 'ticker', 'id',
            'strike_price', 'date_expiration', 'size', 'cost_basis',
            'execution_estimate', 'volume', 'open_interest', 'put_call'
        ]
    }

    seen_ids = {item['id'] for item in seen_list}
    if result['id'] in seen_ids:
        print("Options Flow data already sent!")
        return

    try:
        symbol = result['ticker']
        size = format_number(result['size'])
        premium = format_number(result['cost_basis'])
        strike = result['strike_price']
        put_call = result['put_call'].replace('Calls', 'Call').replace('Puts', 'Put')
        sentiment = result['sentiment']
        option_type = result['option_activity_type']
        
        date_expiration = datetime.strptime(result['date_expiration'], "%Y-%m-%d").strftime("%m/%d/%Y")
        
        
        message = f"Unusual Options Flow Alert:\n\n"
        message += f"${symbol} {put_call} {option_type} ({sentiment})\n"
        message += f"• Strike: ${strike}\n"
        message += f"• Expiration: {date_expiration}\n"
        message += f"• Size: {size}\n"
        message += f"• Premium: ${premium}\n\n"
        message += f"Follow up: stocknear.com/options-flow"
        
        send_post(message)
        
        seen_list.append({'date': result['date'], 'id': result['id']})
        save_json(seen_list, "options_flow")
        print("Options flow post sent successfully!")

    except Exception as e:
        print(f"Error sending options post: {e}")


def recent_earnings():

    try:
        with open("json/facebook/recent_earnings.json", "r") as file:
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


                    message = f"${symbol} Earnings Release:\n"
                    message +=f"· Revenue of {revenue} {revenue_surprise_text} estimates by {revenue_surprise}, with {revenue_yoy_change:.2f}% YoY {revenue_yoy_direction}. \n"
                    message +=f"· EPS of {eps} {eps_surprise_text} estimates by {eps_surprise}, with {eps_yoy_change:.2f}% YoY {eps_yoy_direction}."
                    message +=f"\n \n"
                    message+=f"Follow up here stocknear.com/stocks/{symbol}"
                    
                    send_post(message)

                    seen_list.append({'date': item['date'], 'id': item['id'], 'symbol': symbol})
                else:
                    print("Recent Earnings already posted!")
            except Exception as e:
                print(e)

        try:
            with open("json/facebook/recent_earnings.json","wb") as file:
                print(seen_list)
                file.write(orjson.dumps(seen_list))
        except Exception as e:
            print(e)




def analyst_report():
    try:
        with open("json/facebook/analyst_report.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    with open("json/dashboard/data.json", 'rb') as file:
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
        
            with open(f"json/quote/{symbol}.json","r") as file:
                quote_data = orjson.loads(file.read())

            market_cap = abbreviate_number(quote_data.get('marketCap',0))
            
            summary = (
                f"According to {data['numOfAnalyst']} analysts, ${symbol} has a '{data['consensusRating']}' rating. "
                f"12-month target: ${data['highPriceTarget']} "
                f"({'↑' if data['highPriceChange'] > 0 else '↓'}{abs(data['highPriceChange'])}%)"
            )

            message = f"Analyst Report for ${symbol}:\n\n"
            message += f"{insight}\n\n"
            message += f"{summary}"

            message +=f"\n \n"
            message+=f"Follow up here stocknear.com/stocks/{symbol}"
            
            try:
                send_post(message)
                seen_list.append({'date': data['date'], 'id': data['id']})
                save_json(seen_list, "analyst_report")
                print("Analyst report post sent successfully!")
            except Exception as e:
                print(f"Error sending analyst report post: {e}")
        else:
            print("Analyst Report already sent!")


def congress_trading():
    try:
        with open("json/facebook/congress_trading.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    with open("json/congress-trading/rss-feed/data.json", 'rb') as file:
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
                    representative = item['representative']
                    amount = item['amount']
                    transaction_type = item['type']
                    raw_date = item['transactionDate']  # e.g. "2025-08-21"
                    trade_date = datetime.strptime(raw_date, "%Y-%m-%d").date()

                    # check if it's in the future
                    if trade_date > date.today():
                        trade_date = date.today()
                    transaction_date = trade_date.strftime("%b %d, %Y")

                    
                    message = f"Congress Trading Alert:\n\n"
                    message += f"{representative} {transaction_type.lower()} ${symbol}\n"
                    message += f"• Amount: {amount}\n"
                    message += f"• Trade Date: {transaction_date}\n\n"
                    #message += f"Track more trades: stocknear.com/congress-trading"
                    
                    try:
                        send_post(message)
                        seen_list.append({'date': item['disclosureDate'], 'id': item['id'], 'symbol': symbol})
                        print("Congress trading post sent successfully!")
                    except Exception as e:
                        print(f"Error sending congress trading post: {e}")
                else:
                    print("Congress Data already sent!")

            except Exception as e:
                print(e)
                
        try:
            save_json(seen_list, "congress_trading")
        except Exception as e:
            print(e)


if __name__ == '__main__':
    wiim()
    recent_earnings()
    dark_pool_flow()
    options_flow()
    analyst_report()
    #congress_trading()