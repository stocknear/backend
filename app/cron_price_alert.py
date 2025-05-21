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
origin = "https://stocknear.com" #"http://localhost:5173"
url = f"{origin}/api/sendPushSubscription"

headers = {"Content-Type": "application/json"}

async def push_notification(symbol, user_id):
    data = {
        "title": f"🚨 {symbol} Price Alert triggered",
        "body": "",
        "url": f"{origin}/notifications",
        "userId": user_id,
        "key": stocknear_api_key,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))


async def run():
    result =  pb.collection("priceAlert").get_full_list(query_params={"filter": 'triggered=false'})
    if len(result) != 0:
        for item in result:
            try:
                symbol = item.symbol
                with open(f"json/quote/{symbol}.json", 'r') as file:
                    data = ujson.load(file)
                    current_price = round(data['price'],2)
                    target_price = round(item.target_price,2)
                    if (item.condition == 'below') and target_price >= current_price:
                        #print('below true', symbol, target_price)
                        pb.collection("priceAlert").update(item.id, {"triggered": True})
                        
                        newNotification = {
                        'user': item.user,
                        'notifyType': 'priceAlert',
                        'priceAlert': item.id,
                        'liveResults': {'symbol': symbol, 'assetType': item.asset_type, 'condition': item.condition, 'targetPrice': target_price, 'currentPrice': current_price},
                        }

                        notify_item = pb.collection('notifications').create(newNotification)
                        
                        await push_notification(symbol, item.user)
                        pb.collection('notifications').update(notify_item.id, {"sent": True})

                    elif (item.condition == 'above') and target_price <= current_price:
                        #print('above true', symbol, target_price)
                        pb.collection("priceAlert").update(item.id, {"triggered": True})
                        
                        newNotification = {
                        'user': item.user,
                        'notifyType': 'priceAlert',
                        'priceAlert': item.id,
                        'liveResults': {'symbol': symbol, 'assetType': item.asset_type, 'condition': item.condition, 'targetPrice': target_price, 'currentPrice': current_price},
                        }

                        notify_item = pb.collection('notifications').create(newNotification)
                        await push_notification(symbol, item.user)
                        pb.collection('notifications').update(notify_item.id, {"sent": True})
            
            except Exception as e:
                print(e)
                    

try:
    asyncio.run(run())
except Exception as e:
    print(e)