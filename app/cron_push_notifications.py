import pytz
from datetime import datetime, timedelta
from urllib.request import urlopen
import json
from pocketbase import PocketBase  # Client also works the same
import asyncio
import pytz
import requests
import hashlib
import orjson
import sqlite3
from tqdm import tqdm
import json

from dotenv import load_dotenv
import os

load_dotenv()
stocknear_api_key = os.getenv('STOCKNEAR_API_KEY')

pb_admin_email = os.getenv('POCKETBASE_ADMIN_EMAIL')
pb_password = os.getenv('POCKETBASE_PASSWORD')


berlin_tz = pytz.timezone('Europe/Berlin')
pb = PocketBase('http://127.0.0.1:8090')
admin_data = pb.collection('_superusers').auth_with_password(pb_admin_email, pb_password)


# Define the URL and the API key
origin = "https://stocknear.com" #"http://localhost:5173"
url = f"{origin}/api/sendPushSubscription"
headers = {"Content-Type": "application/json"}

today = datetime.today().strftime('%Y-%m-%d')


with sqlite3.connect('stocks.db') as con:
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

with sqlite3.connect('etf.db') as etf_con:
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

with sqlite3.connect('index.db') as index_con:
    index_cursor = index_con.cursor()
    index_cursor.execute("PRAGMA journal_mode = wal")
    index_cursor.execute("SELECT DISTINCT symbol FROM indices")
    index_symbols = [row[0] for row in index_cursor.fetchall()]

def generate_unique_id(date, text):
    # Concatenate the title and date to form a string
    unique_str = f"{date}-{text}"
    
    # Hash the concatenated string to ensure uniqueness
    unique_id = hashlib.md5(unique_str.encode()).hexdigest()
    
    return unique_id

def format_number(num, decimal=False):
    """Abbreviate large numbers with B/M suffix"""
    if decimal:
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        return f"{num:,.0f}"
    else:
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:,.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:,.2f}M"
        return f"{num:,.0f}"  # Format smaller numbers with commas

async def push_notification(title, text, user_id, link=None):
    data = {
        "title": title,
        "body": text,
        "url": f"{origin}/notifications" if link == None else f"{origin}/{link}",
        "userId": user_id,
        "key": stocknear_api_key,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))


async def push_wiim(user_id):
    """
    Pushes the latest WIIM news based on users' watchlists.

    Steps:
    1. Retrieve the watchlist for each user.
    2. Iterate through the tickers in the watchlist.
    3. Load the corresponding WIIM file and verify if its creation date matches today's date.
    4. If the date matches, check if a notification has already been sent using pushHash.
    5. If no notification has been sent, create and send one.
    """
    try:
        result = pb.collection("watchlist").get_full_list(query_params={"filter": f"user='{user_id}'"})
        all_tickers = set()
        for item in result:
            all_tickers.update(item.ticker)
        all_tickers = list(all_tickers)

        if all_tickers:
            for symbol in all_tickers:
                try:
                    with open(f"json/wiim/company/{symbol}.json","r") as file:
                        data = orjson.loads(file.read())[0]
                        date_string = datetime.strptime(data['date'], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
                    if date_string == today:
                        unique_id = generate_unique_id(date_string, data['text'])
                        #check if push notification already exist
                        all_notification = pb.collection("notifications").get_full_list(query_params={"filter": f"user='{user_id}'"})
                        exist = any(notify_item.push_hash == unique_id for notify_item in all_notification)
                        

                        if exist == False:
                            #check if user is subscribed to pushSubscription to receive push notifications
                            
                            if symbol in stocks_symbols:
                                asset_type = 'stock'
                            elif symbol in etf_symbols:
                                asset_type = 'etf'
                            else:
                                asset_type = 'index'

                            newNotification = {
                                'user': user_id,
                                'notifyType': 'wiim',
                                'sent': True,
                                'pushHash': unique_id,
                                'liveResults': {'symbol': symbol, 'assetType': asset_type},
                            }

                            notify_item = pb.collection('notifications').create(newNotification)

                            #if is_pro == True:

                            check_subscription = pb.collection("pushSubscription").get_full_list(query_params={"filter": f"user='{user_id}'"})
                            user_subscribed = False
                            for item in check_subscription:
                                if item.user == user_id:
                                    user_subscribed = True
                                    break
                            if user_subscribed:
                                if asset_type == 'stock':
                                    link = f"stocks/{symbol}"
                                elif asset_type == 'etf':
                                    link = f"etf/{symbol}"
                                elif asset_type == 'index':
                                    link = f"index/{symbol}"
                                else:
                                    link = None
                                print(link)
                                await push_notification(f'Why Priced Moved for {symbol}', data['text'], user_id, link=link)
                except Exception as e:
                    print(e)
    except Exception as e:
        print(e)

async def push_earnings_release(user_id):
    """
    Pushes the latest earnings releases based on users' watchlists.

    Steps:
    1. Retrieve the watchlist for each user.
    2. Iterate through the tickers in the watchlist.
    3. Load the corresponding earnings files and verify if its creation date matches today's date.
    4. If the date matches, check if a notification has already been sent using pushHash.
    5. If no notification has been sent, create and send one.
    """
    try:
        result = pb.collection("watchlist").get_full_list(query_params={"filter": f"user='{user_id}'"})
        all_tickers = set()
        for item in result:
            all_tickers.update(item.ticker)
        all_tickers = list(all_tickers)

        if all_tickers:
            for symbol in all_tickers:
                try:
                    with open(f"json/earnings/surprise/{symbol}.json","r") as file:
                        data = orjson.loads(file.read())
                        date_string = data['date']

                    if data['revenue'] != None and data['eps'] != None:
                        if date_string == today:
                            sorted_data = json.dumps(data, sort_keys=True)
                            unique_id = hashlib.md5(sorted_data.encode()).hexdigest()
                            
                            #check if push notification already exist
                            all_notification = pb.collection("notifications").get_full_list(query_params={"filter": f"user='{user_id}'"})
                            exist = any(notify_item.push_hash == unique_id for notify_item in all_notification)
                            

                            if exist == False:
                                #check if user is subscribed to pushSubscription to receive push notifications
                                
                                if symbol in stocks_symbols:
                                    asset_type = 'stock'
                                elif symbol in etf_symbols:
                                    asset_type = 'etf'
                                else:
                                    asset_type = 'index'

                                newNotification = {
                                    'user': user_id,
                                    'notifyType': 'earningsSurprise',
                                    'sent': True,
                                    'pushHash': unique_id,
                                    'liveResults': {'symbol': symbol, 'assetType': asset_type},
                                }

                                notify_item = pb.collection('notifications').create(newNotification)

                                #if is_pro == True:

                                check_subscription = pb.collection("pushSubscription").get_full_list(query_params={"filter": f"user='{user_id}'"})
                                user_subscribed = False
                                for item in check_subscription:
                                    if item.user == user_id:
                                        user_subscribed = True
                                        break
                                if user_subscribed:
                                    title = f'Earnings release for {symbol}'
                                    text = f"Revenue of {format_number(data['revenue'])} and EPS of {data['eps']}"
                                    await push_notification(title, text, user_id)
                except Exception as e:
                    print(e)
    except Exception as e:
        print(e)

async def push_top_analyst(user_id):
    """
    Pushes the latest earnings releases based on users' watchlists.

    Steps:
    1. Retrieve the watchlist for each user.
    2. Iterate through the tickers in the watchlist.
    3. Load the corresponding top analyst ratings files and verify if its creation date matches today's date.
    4. If the date matches, check if a notification has already been sent using pushHash.
    5. If no notification has been sent, create and send one.
    """
    try:
        result = pb.collection("watchlist").get_full_list(query_params={"filter": f"user='{user_id}'"})
        all_tickers = set()
        for item in result:
            all_tickers.update(item.ticker)
        all_tickers = list(all_tickers)
        if all_tickers:
            for symbol in all_tickers:
                try:
                    with open(f"json/analyst/history/{symbol}.json","r") as file:
                        data = orjson.loads(file.read())
                        data = [item for item in data if item['analystScore'] >=4 and item['date'] == today and item['rating_current'] != None and item['adjusted_pt_current'] != None]

                    for item in data:
                        try:
                            sorted_data = json.dumps(item, sort_keys=True)
                            unique_id = hashlib.md5(sorted_data.encode()).hexdigest()
                            
                            #check if push notification already exist
                            all_notification = pb.collection("notifications").get_full_list(query_params={"filter": f"user='{user_id}'"})
                            exist = any(notify_item.push_hash == unique_id for notify_item in all_notification)
                            

                            if exist == False:
                                #check if user is subscribed to pushSubscription to receive push notifications
                                asset_type = 'stock'
                             
                                newNotification = {
                                    'user': user_id,
                                    'notifyType': 'topAnalyst',
                                    'sent': True,
                                    'pushHash': unique_id,
                                    'liveResults': {'symbol': symbol, 'assetType': asset_type, 'analyst': item['analyst'], 'rating_current': item['rating_current'], 'adjusted_pt_current': item['adjusted_pt_current']},
                                }

                                notify_item = pb.collection('notifications').create(newNotification)

                                #if is_pro == True:

                                check_subscription = pb.collection("pushSubscription").get_full_list(query_params={"filter": f"user='{user_id}'"})
                                user_subscribed = False
                                for sub in check_subscription:
                                    if sub.user == user_id:
                                        user_subscribed = True
                                        break
                                if user_subscribed:
                                    title = f'New Top Analyst Rating for {symbol}'
                                    text = f"The rating company {item['analyst']} has issued a new rating of „{item['rating_current']}“ with an updated price target of ${item['adjusted_pt_current']}."
                                    await push_notification(title, text, user_id)
                        except Exception as e:
                            print(e)
                except Exception as e:
                    print(e)
    except Exception as e:
        print(e)


async def run():
    all_users = pb.collection("users").get_full_list()
    for item in tqdm(all_users):
        try:
            user_id = item.id

            #is_pro = True if item.tier == 'Pro' else False
            result = pb.collection('notificationChannels').get_list(query_params={"filter": f"user='{user_id}'"})
            channels = result.items
            for channel in channels:
                if channel.wiim == True:     
                    await push_wiim(user_id=user_id)

                if channel.earnings_surprise == True:
                    await push_earnings_release(user_id=user_id)

                if channel.top_analyst == True:
                    await push_top_analyst(user_id=user_id)
        
        except Exception as e:
            print(e)
       

try:
    asyncio.run(run())
except Exception as e:
    print(e)