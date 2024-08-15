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

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import boto3
from botocore.exceptions import NoCredentialsError
from bs4 import BeautifulSoup

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')
pb_admin_email = os.getenv('POCKETBASE_ADMIN_EMAIL')
pb_password = os.getenv('POCKETBASE_PASSWORD')

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

berlin_tz = pytz.timezone('Europe/Berlin')
pb = PocketBase('http://127.0.0.1:8090')
admin_data = pb.admins.auth_with_password(pb_admin_email, pb_password)


#Send price alert via email
def send_email(recipient, symbol, asset_type, current_price,target_price, condition):
    # Replace the placeholders with your AWS SES credentials
    region_name = 'eu-north-1' #email-smtp.eu-north-1.amazonaws.com

    # Replace the placeholders with your sender email and password
    sender_email = 'mrahimi@stocknear.com'

    to_email = recipient # user email address
    subject = f'Price Alert triggered for ${symbol}'
    
    # Read the index.html file
    with open('html_template/price_alert.html', 'r') as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the body element
    html_body = str(soup.body)
    # Get the current date
    current_date = datetime.now()
    # Define the format string
    date_format = "%A - %B %d, %Y"
    # Format the date
    formatted_date = current_date.strftime(date_format)

    if asset_type == 'stock':
        asset_type = 'stocks'
    elif asset_type == 'etf':
        asset_type = 'etf'
    elif asset_type == 'crypto':
        asset_type = 'crypto'

    html_body = html_body.replace('currentDate', formatted_date)
    html_body = html_body.replace('addingSentence', f'The price of ${current_price} is at/{condition} your target price of ${target_price}')
    html_body = html_body.replace('symbol', symbol)
    html_body = html_body.replace('asset-link', f'/{asset_type}/{symbol}')

    # Create a MIMEMultipart object
    message = MIMEMultipart('alternative')
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = to_email

    #Preheader text
    preheader = MIMEText("This is a price alert notification.", 'plain')
    
    message.attach(MIMEText(html_body, 'html'))

    # Use Amazon SES to send the email
    ses_client = boto3.client(
        'ses',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    try:
        # Send the email
        response = ses_client.send_raw_email(
            Source=message['From'],
            Destinations=[message['To']],
            RawMessage={'Data': message.as_string()},
        )
        print("Email sent successfully!")
    except NoCredentialsError:
        print("AWS credentials not available")
    except Exception as e:
        print(f"Error sending email: {e}")


async def run():
    result =  pb.collection("priceAlert").get_full_list(query_params={"filter": 'triggered=false'})
    if len(result) != 0:
        for item in result:
            symbol = item.symbol
            with open(f"json/quote/{symbol}.json", 'r') as file:
                data = ujson.load(file)
                current_price = round(data['price'],2)
                target_price = round(item.target_price,2)
                if (item.condition == 'below') and target_price >= current_price:
                    #print('below true', symbol, target_price)
                    pb.collection("priceAlert").update(item.id, {"triggered": True})
                    
                    newNotification = {
                    'opUser': item.user,
                    'user': '9ncz4wunmhk0k52', #stocknear bot id
                    'notifyType': 'priceAlert',
                    'priceAlert': item.id,
                    'liveResults': {'symbol': symbol, 'assetType': item.asset_type, 'condition': item.condition, 'targetPrice': target_price, 'currentPrice': current_price},
                    }
                    pb.collection('notifications').create(newNotification)
                    #send alert via email
                    recipient = (pb.collection('users').get_one(item.user)).email
                    send_email(recipient, symbol, item.asset_type, current_price, target_price, item.condition)

                elif (item.condition == 'above') and target_price <= current_price:
                    #print('above true', symbol, target_price)
                    pb.collection("priceAlert").update(item.id, {"triggered": True})
                    
                    newNotification = {
                    'opUser': item.user,
                    'user': '9ncz4wunmhk0k52', #stocknear bot id
                    'notifyType': 'priceAlert',
                    'priceAlert': item.id,
                    'liveResults': {'symbol': symbol, 'assetType': item.asset_type, 'condition': item.condition, 'targetPrice': target_price, 'currentPrice': current_price},
                    }
                    pb.collection('notifications').create(newNotification)
                    #send alert via email
                    recipient = item.email
                    send_email(recipient, symbol, item.asset_type, current_price, target_price, item.condition)
try:
    asyncio.run(run())
except Exception as e:
    print(e)