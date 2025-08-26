import praw
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone, date
import hashlib
import orjson
import pytz
from typing import Optional, Dict, Any

load_dotenv()

# Reddit credentials
reddit_client = praw.Reddit(
    client_id=os.getenv('REDDIT_BOT_API_KEY'),
    client_secret=os.getenv('REDDIT_BOT_API_SECRET'),
    username=os.getenv('REDDIT_USERNAME'),
    password=os.getenv('REDDIT_PASSWORD'),
    user_agent=os.getenv('REDDIT_USER_AGENT', 'script:stocknear_bot:v2.0 (by /u/stocknear)')
)

# Target subreddit
subreddit = reddit_client.subreddit("stocknear")

# Timezone and date setup
ny_tz = pytz.timezone("America/New_York")
today = datetime.utcnow().date()
now = datetime.now(timezone.utc)
now_ny = datetime.now(ny_tz)

def save_json(data, file):
    """Save data to JSON file"""
    directory = "json/reddit"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{file}.json", 'wb') as f:
        f.write(orjson.dumps(data))

def format_number(num, decimal=False):
    """Abbreviate large numbers with B/M suffix and format appropriately"""
    if isinstance(num, str) and ('e' in num.lower() or 'E' in num.lower()):
        try:
            num = float(num)
        except ValueError:
            return num
    
    if isinstance(num, str):
        try:
            num = float(num)
            if num.is_integer():
                num = int(num)
        except ValueError:
            return num
            
    if num >= 1_000_000_000:
        formatted = num / 1_000_000_000
        return f"{formatted:.2f}B".rstrip('0').rstrip('.') + 'B'
    elif num >= 1_000_000:
        formatted = num / 1_000_000
        return f"{formatted:.2f}".rstrip('0').rstrip('.') + 'M'
    elif decimal and isinstance(num, float) and not num.is_integer():
        return f"{num:,.2f}"
    else:
        return f"{num:,.0f}"

def abbreviate_number(n):
    """Abbreviate a number to a readable format"""
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
    """Extract the first value from a string like "$1K-$15K" """
    s = s.upper().replace('$', '')
    first_part = s.split('-')[0]

    if 'K' in first_part:
        return int(float(first_part.replace('K', '')) * 1_000)
    elif 'M' in first_part:
        return int(float(first_part.replace('M', '')) * 1_000_000)
    else:
        return int(first_part)

def create_reddit_post(title: str, content: str = None, url: str = None, flair_id: str = None) -> bool:
    """
    Create a Reddit post with error handling
    
    Args:
        title: Post title
        content: Text content (for text posts)
        url: URL (for link posts) 
        flair_id: Flair ID for the post
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if url:
            post = subreddit.submit(
                title=title,
                url=url,
                flair_id=flair_id
            )
        else:
            post = subreddit.submit(
                title=title,
                selftext=content or "",
                flair_id=flair_id
            )
        
        print(f"✅ Reddit post created successfully: {post.url}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating Reddit post: {e}")
        print(f"Title: {title[:100]}...")
        return False

def wiim():
    try:
        with open("json/reddit/wiim.json", "r") as f:
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
        seen_ids = {item['id'] for item in seen_list} if seen_list else set()

        for item in res_list:
            try:
                if item is not None and item['id'] not in seen_ids:
                    symbol = item['ticker']
                    asset_type = item['assetType']
                    description = item.get('text', '')
                    
                    title = f"Market Insight: ${symbol}"
                    content = f"**Symbol:** ${symbol}\n\n"
                    content += f"**Insight:** {description}\n\n"
                    content += f"[View detailed analysis on Stocknear](https://stocknear.com/{asset_type}/{symbol})"
                    
                    flair_id = ""
                    
                    if create_reddit_post(title, content=content, flair_id=flair_id):
                        seen_list.append({'date': item['date'], 'id': item['id'], 'symbol': symbol})
                else:
                    print("WIIM already posted!")
            except Exception as e:
                print(f"Error posting WIIM: {e}")

        try:
            save_json(seen_list, "wiim")
        except Exception as e:
            print(f"Error saving WIIM data: {e}")

def dark_pool_flow():
    """Post unusual dark pool activity alerts"""
    N_minutes_ago = now_ny - timedelta(minutes=30)
    
    try:
        with open("json/reddit/dark_pool.json", "r") as file:
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

        if filtered_recent:
            best_order = max(filtered_recent, key=lambda x: float(x['premium']))
            result = {k: best_order[k] for k in ['date', 'trackingID', 'price', 'size', 'premium', 'ticker']}
        else:
            result = None

        seen_ids = {item['trackingID'] for item in seen_list} if seen_list else set()

        if result and result['trackingID'] not in seen_ids:
            symbol = result['ticker']
            quantity = format_number(result['size'])
            price = result['price']
            amount = format_number(result['premium'])

            title = f"🚨 Unusual Dark Pool Alert: ${symbol} - ${amount} Trade"
            content = f"**Large dark pool order detected for ${symbol}**\n\n"
            content += f"**Trade Details:**\n\n"
            content += f"• **Symbol:** ${symbol}\n\n"
            content += f"• **Quantity:** {quantity} shares\n\n"
            content += f"• **Price:** ${price}\n\n"
            content += f"• **Total Amount:** ${amount}\n\n"
            content += f"💡 **What this means:**\n\n"
            content += f"Dark pool trades are large institutional orders executed away from public markets. "
            content += f"This significant ${amount} order suggests major institutional interest in ${symbol}.\n\n"
            content += f"[Track more dark pool activity on Stocknear](https://stocknear.com/dark-pool-flow)\n\n"
            
            flair_id = ""
            
            try:
                if create_reddit_post(title, content=content, flair_id=flair_id):
                    seen_list.append({'date': result['date'], 'trackingID': result['trackingID']})
                    save_json(seen_list, "dark_pool")
                    print("Dark pool Reddit post created successfully!")
            except Exception as e:
                print(f"Error posting dark pool update: {e}")
        else:
            print("Dark pool already posted!")

def options_flow():
    """Post unusual options flow alerts"""
    now_ny = datetime.now(ny_tz)
    N_minutes_ago_options = now_ny - timedelta(minutes=10)
    today_ny = now_ny.date()

    try:
        with open("json/reddit/options_flow.json", "rb") as file:
            seen_list = orjson.loads(file.read())
            today_iso = today_ny.isoformat()
            seen_list = [item for item in seen_list if item['date'] == today_iso]
    except:
        seen_list = []

    try:
        with open("json/options-flow/feed/data.json", "rb") as file:
            res_list = orjson.loads(file.read())
    except Exception as e:
        print(f"Error loading options data: {e}")
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
            continue

    if not filtered:
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
        print("Options Flow data already posted!")
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
        
        title = f"Unusual Options Flow: ${symbol} {put_call} - ${premium}"
        content = f"**Large options order detected for ${symbol}**\n\n"
        content += f"**Options Details:**\n\n"
        content += f"• **Symbol:** ${symbol}\n\n"
        content += f"• **Type:** {put_call} {option_type}\n\n"
        content += f"• **Strike Price:** ${strike}\n\n"
        content += f"• **Expiration:** {date_expiration}\n\n"
        content += f"• **Size:** {size} contracts\n\n"
        content += f"• **Premium:** ${premium}\n\n"
        content += f"• **Sentiment:** {sentiment}\n\n"
        content += f"**Analysis:**\n"
        content += f"This {sentiment.lower()} {put_call.lower()} order suggests institutional traders are "
        
        if sentiment.lower() == "bullish":
            content += f"positioning for upward price movement in ${symbol}."
        elif sentiment.lower() == "bearish":
            content += f"positioning for downward price movement in ${symbol}."
        else:
            content += f"taking a neutral stance on ${symbol}."
            
        content += f"\n\n[Track more options flow on Stocknear](https://stocknear.com/options-flow)\n\n"
        
        flair_id = ""
        
        if create_reddit_post(title, content=content, flair_id=flair_id):
            seen_list.append({'date': result['date'], 'id': result['id']})
            save_json(seen_list, "options_flow")
            print("Options flow Reddit post created successfully!")

    except Exception as e:
        print(f"Error posting options flow update: {e}")

def recent_earnings():
    """Post recent earnings releases"""
    try:
        with open("json/reddit/recent_earnings.json", "r") as file:
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
                item['name'] = quote_data.get('name')
                item['eps'] = quote_data.get('eps',0)

            unique_str = f"{item['date']}-{item['symbol']}-{item.get('revenueSurprise',0)}-{item.get('epsSurprise',0)}"
            item['id'] = hashlib.md5(unique_str.encode()).hexdigest()
            
            if item['marketCap'] > 1E9:
                res_list.append(item)
        except:
            pass
    
    if res_list:
        seen_ids = {item['id'] for item in seen_list} if seen_list else set()

        for item in res_list:
            try:
                if item and item['id'] not in seen_ids and item['marketCap']:
                    symbol = item['symbol']
                    name = item['name']
                    revenue = abbreviate_number(item['revenue'])
                    revenue_surprise = abbreviate_number(item.get("revenueSurprise", 0))
                    eps = item['eps']
                    eps_surprise = item.get("epsSurprise", 0)
                    revenue_surprise_text = "beat" if item["revenueSurprise"] > 0 else "missed"
                    eps_surprise_text = "beat" if eps_surprise > 0 else "missed"

                    revenue_yoy_change = (item["revenue"] / item["revenuePrior"] - 1) * 100
                    revenue_yoy_direction = "decline" if revenue_yoy_change < 0 else "growth"

                    eps_yoy_change = (item["eps"] / item["epsPrior"] - 1) * 100
                    eps_yoy_direction = "decline" if eps_yoy_change < 0 else "growth"
                    
                    # Emojis based on performance
                    revenue_emoji = "📈" if item["revenueSurprise"] > 0 else "📉"
                    eps_emoji = "📈" if eps_surprise > 0 else "📉"
                    
                    title = f"{name} ({symbol}) Earnings Released"
                    content = f"**{name} (${symbol}) has released their quarterly earnings**\n\n"
                    content += f"**Financial Performance:**\n\n"
                    content += f"{revenue_emoji} **Revenue:** ${revenue}\n\n"
                    content += f"• {revenue_surprise_text.title()} estimates by ${revenue_surprise}\n\n"
                    content += f"• {revenue_yoy_change:.2f}% YoY {revenue_yoy_direction}\n\n\n\n"
                    content += f"{eps_emoji} **Earnings Per Share:** ${eps}\n\n"
                    content += f"• {eps_surprise_text.title()} estimates by ${eps_surprise}\n\n"
                    content += f"• {eps_yoy_change:.2f}% YoY {eps_yoy_direction}\n\n"
                    content += f"**Market Cap:** {format_number(item['marketCap'])}\n\n"
                    content += f"**What this means:**\n"
                    
                    if item["revenueSurprise"] > 0 and eps_surprise > 0:
                        content += f"${symbol} delivered a strong quarter, beating both revenue and EPS expectations. "
                    elif item["revenueSurprise"] > 0 or eps_surprise > 0:
                        content += f"${symbol} had mixed results, with some metrics beating expectations. "
                    else:
                        content += f"${symbol} faced challenges this quarter, missing analyst expectations. "
                    
                    content += f"This could impact the stock's near-term performance.\n\n"
                    content += f"[View detailed analysis on Stocknear](https://stocknear.com/stocks/{symbol})\n\n"
                    
                    flair_id = "b9f76638-772e-11ef-96c1-0afbf26bd890"  # Earnings flair
                    
                    if create_reddit_post(title, content=content, flair_id=flair_id):
                        seen_list.append({'date': item['date'], 'id': item['id'], 'symbol': symbol})
                else:
                    print("Recent Earnings already posted!")
            except Exception as e:
                print(f"Error posting earnings: {e}")

        try:
            save_json(seen_list, "recent_earnings")
        except Exception as e:
            print(f"Error saving earnings data: {e}")

def analyst_report():
    """Post analyst reports and ratings"""
    try:
        with open("json/reddit/analyst_report.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    with open("json/dashboard/data.json", 'rb') as file:
        data = orjson.loads(file.read())['analystReport']
        date_obj = datetime.strptime(data['date'], '%b %d, %Y')
        data['date'] = date_obj.strftime('%Y-%m-%d')

    if datetime.fromisoformat(data['date']).date() == today:
        seen_ids = {item['id'] for item in seen_list} if seen_list else set()

        if data['id'] not in seen_ids:
            symbol = data['symbol']
            insight = data['insight']
        
            with open(f"json/quote/{symbol}.json","r") as file:
                quote_data = orjson.loads(file.read())
                company_name = quote_data.get('name', symbol)

            market_cap = abbreviate_number(quote_data.get('marketCap',0))
            
            # Rating emoji
            rating = data['consensusRating'].lower()
                
            title = f"Analyst Report: {company_name} ({symbol}) - {data['consensusRating']}"
            content = f"**Professional analyst consensus for {company_name} (${symbol})**\n\n"
            content += f"**Analyst Consensus:**\n"
            content += f"• **Rating:** {data['consensusRating']}\n\n"
            content += f"• **Analysts Coverage:** {data['numOfAnalyst']} analysts\n\n"
            content += f"• **Price Target:** ${data['highPriceTarget']}\n\n"
            content += f"• **Potential Upside:** {'↗️' if data['highPriceChange'] > 0 else '↘️'} {abs(data['highPriceChange']):.1f}%\n\n"
            content += f"**Market Cap:** {market_cap}\n\n"
            content += f"**Key Insights:**\n{insight}\n\n"
            content += f"[View detailed analysis on Stocknear](https://stocknear.com/stocks/{symbol}/forecast)\n\n"
            content += f"[Compare analyst ratings](https://stocknear.com/analysts)"
            
            flair_id = ""
            
            try:
                if create_reddit_post(title, content=content, flair_id=flair_id):
                    seen_list.append({'date': data['date'], 'id': data['id']})
                    save_json(seen_list, "analyst_report")
                    print("Analyst report Reddit post created successfully!")
            except Exception as e:
                print(f"Error posting analyst report: {e}")
        else:
            print("Analyst Report already posted!")

def congress_trading():
    """Post congressional trading activity"""
    try:
        with open("json/reddit/congress_trading.json", "r") as file:
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
                item['name'] = quote_data.get('name','N/A')

            item['amountInt'] = extract_first_value(item['amount'])
            unique_str = f"{item['disclosureDate']}-{item['transactionDate']}-{item['ticker']}-{item['amount']}-{item['representative']}"
            item['id'] = hashlib.md5(unique_str.encode()).hexdigest()
            
            if item['amountInt'] >= 15_000:
                res_list.append(item)

        except Exception as e:
            print(f"Error processing congress item: {e}")
    
    if res_list:
        seen_ids = {item['id'] for item in seen_list} if seen_list else set()

        for item in res_list:
            try:
                if item and item['id'] not in seen_ids:
                    symbol = item['ticker']
                    representative = item['representative']
                    amount = item['amount']
                    transaction_type = item['type']
                    company_name = item['name']
                    transaction_date = datetime.strptime(item['transactionDate'], "%Y-%m-%d").strftime("%m/%d/%Y")
                                        
                    title = f"Congressional Trading: {representative} {transaction_type} {company_name} ({symbol})"
                    content = f"**Congressional trading activity disclosed**\n\n"
                    content += f"**Representative:** {representative}\n\n"
                    content += f"**Company:** {company_name} (${symbol})\n\n"
                    content += f"{transaction_emoji} **Transaction:** {transaction_type}\n\n"
                    content += f"**Amount:** {amount}\n\n"
                    content += f"**Trade Date:** {transaction_date}\n\n"
                    content += f"**About Congressional Trading:**\n\n"
                    content += f"Members of Congress are required to disclose their stock trades within 45 days. "
                    content += f"These disclosures provide insights into how elected officials are positioning their portfolios.\n\n"
                    content += f"**Important:** This information is for educational purposes only and should not be considered investment advice.\n\n"
                    content += f"[Track more congressional trades on Stocknear](https://stocknear.com/politicians)\n\n"
                    
                    flair_id = ""
                    
                    try:
                        if create_reddit_post(title, content=content, flair_id=flair_id):
                            seen_list.append({'date': item['disclosureDate'], 'id': item['id'], 'symbol': symbol})
                            print("Congress trading Reddit post created successfully!")
                    except Exception as e:
                        print(f"Error posting congress trading update: {e}")
                else:
                    print("Congress Data already posted!")

            except Exception as e:
                print(f"Error processing congress trading: {e}")
                
        try:
            save_json(seen_list, "congress_trading")
        except Exception as e:
            print(f"Error saving congress trading data: {e}")



def main():
    #options_flow()
    #dark_pool_flow()
    analyst_report()
    #congress_trading()
    recent_earnings()
    #wiim()

if __name__ == '__main__':
    main()