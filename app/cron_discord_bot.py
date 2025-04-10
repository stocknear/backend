import requests
import time
import orjson
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv


load_dotenv()
today = datetime.utcnow().date()
now = datetime.now(timezone.utc)
N_minutes_ago = now - timedelta(minutes=30)

WEBHOOK_URL = os.getenv("DISCORD_DARK_POOL_WEBHOOK")

def save_json(data):
    directory = "json/discord"
    try:
        os.makedirs(directory, exist_ok=True)
        with open(directory+"/dark_pool.json", 'wb') as file:
            file.write(orjson.dumps(data))
    except Exception as e:
        print(f"An error occurred while saving data: {e}")


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

def dark_pool_flow():
    
    try:
        with open(f"json/discord/dark_pool.json", "r") as file:
            seen_list = orjson.loads(file.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except:
        seen_list = []

    with open(f"json/dark-pool/feed/data.json", "r") as file:
        res_list = orjson.loads(file.read())
        res_list = [item for item in res_list if float(item['premium']) >= 30E6 and datetime.fromisoformat(item['date']).date() == today]


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
                "color": 0xC475FD,  # Green color from original embed
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

            response = requests.post(WEBHOOK_URL, json=payload)

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

if __name__ == "__main__":
    dark_pool_flow()