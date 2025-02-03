import pytz
from datetime import datetime, timedelta
from urllib.request import urlopen
import certifi
import json
import ujson
import schedule
import time
import subprocess
from pocketbase import PocketBase  # Client also works the same
import asyncio
import aiohttp
import pytz
import pandas as pd
import numpy as np
import requests
import hashlib
import orjson
import sqlite3
from tqdm import tqdm

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')
stocknear_api_key = os.getenv('STOCKNEAR_API_KEY')

pb_admin_email = os.getenv('POCKETBASE_ADMIN_EMAIL')
pb_password = os.getenv('POCKETBASE_PASSWORD')


berlin_tz = pytz.timezone('Europe/Berlin')
pb = PocketBase('http://127.0.0.1:8090')
admin_data = pb.collection('_superusers').auth_with_password(pb_admin_email, pb_password)


# Define the URL and the API key
origin = "http://localhost:5173" #"http://localhost:5173"
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


def generate_unique_id(date, text):
    # Concatenate the title and date to form a string
    unique_str = f"{date}-{text}"
    
    # Hash the concatenated string to ensure uniqueness
    unique_id = hashlib.md5(unique_str.encode()).hexdigest()
    
    return unique_id

async def push_notification(title, text, user_id):
    data = {
        "title": title,
        "body": text,
        "url": f"{origin}/notifications",
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
                        all_notification = pb.collection("notifications").get_full_list(query_params={"filter": f"opUser='{user_id}'"})
                        exist = any(notify_item.push_hash == unique_id for notify_item in all_notification)
                        

                        if exist == False:
                            #check if user is subscribed to pushSubscription to receive push notifications
                            check_subscription = pb.collection("pushSubscription").get_full_list(query_params={"filter": f"user='{user_id}'"})
                            user_subscribed = False
                            for item in check_subscription:
                                if item.user == user_id:
                                    user_subscribed = True
                                    break

                            if user_subscribed:
                                #create notification in pb and push notification
                                newNotification = {
                                    'opUser': user_id,
                                    'user': '9ncz4wunmhk0k52', #stocknear bot id
                                    'notifyType': 'wiim',
                                    'sent': True,
                                    'pushHash': unique_id,
                                    'liveResults': {'symbol': symbol, 'assetType': 'stocks' if symbol in stocks_symbols else 'etf'},
                                }
                                notify_item = pb.collection('notifications').create(newNotification)
                                await push_notification(f'⚡News Update for {symbol}', data['text'], user_id)
                        
                except:
                    pass
    except Exception as e:
        print(e)


async def run():
    all_users = pb.collection("users").get_full_list(query_params={"filter": "tier='Pro'"})
    for item in tqdm(all_users):
        user_id = item.id
        await push_wiim(user_id=user_id)
       

try:
    asyncio.run(run())
except Exception as e:
    print(e)