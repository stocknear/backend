import sys
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
from tqdm import tqdm

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import boto3
from botocore.exceptions import NoCredentialsError
from bs4 import BeautifulSoup

from dotenv import load_dotenv
import os

load_dotenv()
pb_admin_email = os.getenv('POCKETBASE_ADMIN_EMAIL')
pb_password = os.getenv('POCKETBASE_PASSWORD')

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

berlin_tz = pytz.timezone('Europe/Berlin')
pb = PocketBase('http://127.0.0.1:8090')
admin_data = pb.collection('_superusers').auth_with_password(pb_admin_email, pb_password)


now = datetime.today().date()
#one_month_ago = now - timedelta(days=30)
time_threshold = now - timedelta(days=7)


def send_email(recipient):
    # Replace the placeholders with your AWS SES credentials
    region_name = 'eu-north-1' #email-smtp.eu-north-1.amazonaws.com

    # Replace the placeholders with your sender email and password
    sender_email = 'mrahimi@stocknear.com'

    to_email = recipient # user email address
    subject = f'Your Free Trial expired'
    
    # Read the index.html file
    with open('html_template/free_trial.html', 'r') as file:
        html_content = file.read()

  
    # Create a MIMEMultipart object
    message = MIMEMultipart('alternative')
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = to_email

    #Preheader text
    preheader = MIMEText("Your free trial ended.", 'plain')
    
    message.attach(MIMEText(html_content, 'html'))

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


async def update_free_trial():

    data = pb.collection("users").get_full_list(query_params = {"filter": f'freeTrial = True'})

    for item in data:
        created_date = item.created
        # Check if the created date is more than N days ago
        if created_date < time_threshold:
            # Update the user record
            pb.collection("users").update(item.id, {
                "tier": 'Free',
                "freeTrial": False,
            })
            try:
                send_email(item.email)
            except Exception as e:
                print(e)



async def downgrade_user():

    user_data =  pb.collection('users').get_full_list()
    for item in tqdm(user_data):
        if item.tier not in ['Pro', 'Plus']:
            try:
                pb.collection("users").update(item.id, {
                        "credits": 10,
                    })

                stock_screener_data = pb.collection("stockscreener").get_full_list(query_params = {"filter": f"user = '{item.id}'"})
                for screener in stock_screener_data:
                    pb.collection('stockscreener').delete(screener.id)

                options_watchlist_data = pb.collection("optionsWatchlist").get_full_list(query_params = {"filter": f"user = '{item.id}'"})
                for watchlist in options_watchlist_data:
                    pb.collection('optionsWatchlist').delete(watchlist.id)


                payment_data = pb.collection("payments").get_full_list(query_params = {"filter": f"user = '{item.id}'"})
                for item in payment_data:
                    pb.collection('payments').delete(item.id)
            except:
                pass


async def delete_old_notifications():
    # Get the current datetime
    now = datetime.utcnow()
    # Calculate the threshold datetime
    time_threshold = now - timedelta(days=7)

    # Fetch all notifications
    data = pb.collection("notifications").get_full_list()
    for item in data:
        try:
            # Ensure both are datetime objects before comparison
            if item.created < time_threshold:
                pb.collection('notifications').delete(item.id)
        except:
            pass

async def refresh_bulk_credits():
    user_data =  pb.collection('users').get_full_list()
    for item in tqdm(user_data):
        try:
            if item.tier == 'Plus':
                pb.collection("users").update(item.id, {
                    "credits": 500,
                })
            elif item.tier == 'Pro':
                pb.collection("users").update(item.id, {
                    "credits": 1000,
                })

            else:
                pb.collection("users").update(item.id, {
                    "credits": 10,
                })
        except Exception as e:
            print(e)



async def run_all_except_refresh():
    await update_free_trial()
    await downgrade_user()
    await delete_old_notifications()

def main():
    if '--refresh' in sys.argv:
        asyncio.run(refresh_bulk_credits())
    else:
        asyncio.run(run_all_except_refresh())

if __name__ == '__main__':
    main()