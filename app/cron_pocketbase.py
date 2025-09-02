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

import discord
from discord.ext import commands

from dotenv import load_dotenv
import os

load_dotenv()
pb_admin_email = os.getenv('POCKETBASE_ADMIN_EMAIL')
pb_password = os.getenv('POCKETBASE_PASSWORD')

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region_ses = os.getenv("AWS_REGION_SES")

berlin_tz = pytz.timezone('Europe/Berlin')
pb = PocketBase('http://127.0.0.1:8090')
admin_data = pb.collection('_superusers').auth_with_password(pb_admin_email, pb_password)


now = datetime.today().date()
#one_month_ago = now - timedelta(days=30)
time_threshold = now - timedelta(days=7)

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)


def send_email(recipient, template_name='free_trial', subject=None):

    # Replace the placeholders with your sender email and password
    sender_email = 'mrahimi@stocknear.com'

    to_email = recipient # user email address
    
    # Set default subject if not provided
    if subject is None:
        subject = f'Your Free Trial expired'
    
    # Read the index.html file
    with open(f'html_template/{template_name}.html', 'r') as file:
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
    # Extract region from SMTP endpoint if it's in that format
    if aws_region_ses and 'email-smtp' in aws_region_ses:
        # Extract region from format like 'email-smtp.eu-north-1.amazonaws.com'
        region = aws_region_ses.split('.')[1] if '.' in aws_region_ses else 'eu-north-1'
    else:
        region = aws_region_ses if aws_region_ses else 'eu-north-1'
    
    ses_client = boto3.client(
        'ses',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region
    )

    try:
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



async def downgrade_discord_role_of_user():
    """Process the user list and assign roles accordingly"""
    data = pb.collection("users").get_full_list()
    
    USER_LIST = []
    for item in data:
        try:
            if item.discord and len(item.discord) > 0:
                USER_LIST.append({**item.discord, 'tier': item.tier, 'user_id': item.id})
        except:
            pass
        
    for guild in bot.guilds:
        print(f"\nProcessing server: {guild.name}")
        
        # Get Pro role (Free is not a role, just means no Pro role)
        pro_role = discord.utils.get(guild.roles, name="Pro")
        
        if not pro_role:
            print(f"  Warning: 'Pro' role not found in {guild.name}")
            continue  # Skip this server if Pro role doesn't exist
        
        # Process each user in the list
        for user_data in USER_LIST:
            user_id = user_data['id']
            tier = user_data['tier']
            db_user_id = user_data['user_id']
            changes_made = []
            
            try:
                # Get the member
                member = guild.get_member(user_id)
                if not member:
                    try:
                        member = await guild.fetch_member(user_id)
                    except discord.NotFound:
                        print(f"  User {user_id} not found in {guild.name}")
                        continue
                
                # Handle role assignment based on tier
                if tier == 'Pro':
                    # User should have Pro role
                    if pro_role not in member.roles:
                        await member.add_roles(pro_role)
                        changes_made.append(f"added {pro_role.name}")
                    else:
                        print(f"  - {member.display_name} ({user_id}): already has Pro role")
                        continue
                        
                elif tier != 'Pro':
                    # User should NOT have Pro role, remove it if they have it
                    if pro_role in member.roles:
                        await member.remove_roles(pro_role)
                        changes_made.append(f"removed {pro_role.name}")
                        
                        # Update database to set access to false within discord object
                        try:
                            # Get current discord data and update the access field
                            current_discord = item.discord if hasattr(item, 'discord') else user_data
                            current_discord['access'] = False
                            
                            pb.collection("users").update(db_user_id, {
                                "discord": current_discord,
                            })
                            changes_made.append("set discord.access to false in database")
                        except Exception as db_e:
                            print(f"  ✗ Failed to update database for user {db_user_id}: {db_e}")
                    else:
                        print(f"  - {member.display_name} ({user_id}): already Free (no Pro role)")
                        continue
                
                # Report changes
                if changes_made:
                    print(f"  ✓ {member.display_name} ({user_id}): {', '.join(changes_made)}")
                
            except discord.Forbidden:
                print(f"  ✗ No permission to assign roles to user {user_id} in {guild.name}")
            except discord.HTTPException as e:
                print(f"  ✗ Failed to assign role to user {user_id} in {guild.name}: {e}")
            except Exception as e:
                print(f"  ✗ Unexpected error for user {user_id} in {guild.name}: {e}")
    
    print(f"\nRole assignment complete!")



@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f'Bot is in {len(bot.guilds)} servers')
    
    try:
        # Process all users and assign roles
        await downgrade_discord_role_of_user()
    except Exception as e:
        print(f"Error during role assignment: {e}")
    finally:
        # Close the bot connection and exit
        print("Closing bot connection...")
        await bot.close()

async def update_discord_roles():
    """Run the Discord bot and handle role updates"""
    try:
        await bot.start(TOKEN)
    except KeyboardInterrupt:
        print("\nBot interrupted by user")
    except Exception as e:
        print(f"Bot error: {e}")
    finally:
        print("Bot has finished running.")
        if not bot.is_closed():
            await bot.close()




async def update_free_trial():

    data = pb.collection("users").get_full_list(query_params = {"filter": f'freeTrial = True'})

    for item in data:
        created_date = item.created
        # Check if the created date is more than N days ago
        if created_date.date() < time_threshold:
            # Update the user record
            pb.collection("users").update(item.id, {
                "tier": 'Free',
                "freeTrial": False,
                "credits": 10,
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
                '''
                frontend is doing this part
                pb.collection("users").update(item.id, {
                        "credits": 10,
                    })
                '''
                
                stock_screener_data = pb.collection("stocksScreener").get_full_list(query_params = {"filter": f"user = '{item.id}'"})
                for screener in stock_screener_data:
                    pb.collection('stocksScreener').delete(screener.id)

                option_screener_data = pb.collection("optionsScreener").get_full_list(query_params = {"filter": f"user = '{item.id}'"})
                for screener in option_screener_data:
                    pb.collection('optionsScreener').delete(screener.id)

                options_watchlist_data = pb.collection("optionsWatchlist").get_full_list(query_params = {"filter": f"user = '{item.id}'"})
                for watchlist in options_watchlist_data:
                    pb.collection('optionsWatchlist').delete(watchlist.id)

                backtesting_data = pb.collection("backtesting").get_full_list(query_params = {"filter": f"user = '{item.id}'"})
                for strategy in backtesting_data:
                    pb.collection('backtesting').delete(strategy.id)


                payment_data = pb.collection("payments").get_full_list(query_params = {"filter": f"user = '{item.id}'"})
                for payment_item in payment_data:
                    pb.collection('payments').delete(payment_item.id)
                

                watchlist_data = pb.collection("watchlist").get_full_list(query_params={"filter": f"user = '{item.id}'"})

                if len(watchlist_data) > 1:

                    # Keep the first one
                    keep = watchlist_data[0]

                    # Limit tickers to max 5
                    tickers = keep.ticker  # assuming list
                    if len(tickers) > 5:
                        new_tickers = tickers[:5]
                        pb.collection("watchlist").update(keep.id, {"ticker": new_tickers})

                    # Delete all others
                    for watchlist in watchlist_data[1:]:
                        pb.collection("watchlist").delete(watchlist.id)

                elif len(watchlist_data) == 1:
                    watchlist = watchlist_data[0]
                    tickers = watchlist.ticker  # assuming list

                    if len(tickers) > 5:
                        new_tickers = tickers[:5]
                        # Update the watchlist with the first 5 tickers only
                        pb.collection("watchlist").update(watchlist.id, {"ticker": new_tickers})





                #for watchlist in watchlist_data:


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

async def email_marketing():
    """
    Send marketing emails to free tier users based on account age.
    Day 1 after signup: day_1.html - Welcome & Value Proposition
    Day 2 after signup: day_2.html - Stock Screener
    Day 3 after signup: day_3.html - Price Alerts
    Day 4 after signup: day_4.html - AI Analysis
    Day 5 after signup: day_5.html - Dark Pool & Options Flow
    Day 6 after signup: day_6.html - Hedge Fund & Congress Tracking
    """
    print("Starting email marketing campaign...")
    
    # Get all users who are not Plus or Pro tier
    data = pb.collection("users").get_full_list(
        query_params={"filter": f"tier != 'Plus' && tier != 'Pro'"}
    )

    # Email subject lines for each day
    email_subjects = {
        1: "Welcome to Stocknear! Your Trading Game Just Changed",
        2: "Discover Winning Stocks in Seconds",
        3: "Never Miss a Breakout Again",
        4: "Your AI Analyst Works 24/7",
        5: "See What Wall Street Doesn't Want You to See",
        6: "Copy the Smartest Money in the Market"
    }
    
    today = datetime.now(berlin_tz).date()
    emails_sent = 0
    emails_skipped = 0
    
    for user in data:
        try:
            # Calculate days since account creation
            created_date = user.created.date()
            days_since_created = (today - created_date).days
            # Check if we should send an email (days 1-6)
            if 1 <= days_since_created <= 6:
                template_name = f"day_{days_since_created}"
                subject = email_subjects.get(days_since_created, f"Day {days_since_created}: Unlock Your Trading Potential")
                
                # Get existing marketing field or initialize as empty dict
                try:
                    marketing_data = user.marketing if hasattr(user, 'marketing') and user.marketing else {}
                except:
                    marketing_data = {}
                
                # Check if this specific email has already been sent
                if template_name in marketing_data and marketing_data[template_name] == True:
                    emails_skipped += 1
                    print(f"Skipping {template_name} email for {user.email} - already sent")
                    continue
                
                try:
                    # Send the email
                    send_email(user.email, template_name=template_name, subject=subject)
                    
                    # Update the marketing field to track that this email was sent
                    marketing_data[template_name] = True
                    pb.collection("users").update(user.id, {
                        "marketing": marketing_data
                    })
                    
                    emails_sent += 1
                    print(f"Sent {template_name} email to {user.email} and updated tracking")
                    
                except Exception as e:
                    print(f"Failed to send {template_name} email to {user.email}: {e}")
                    
        except Exception as e:
            print(f"Error processing user {user.id}: {e}")
    
    print(f"Email marketing completed. Sent {emails_sent} emails, skipped {emails_skipped} (already sent).")

async def refresh_bulk_credits():
    user_data =  pb.collection('users').get_full_list()
    for item in tqdm(user_data):
        try:
            if item.tier == 'Plus':
                pb.collection("users").update(item.id, {
                    "credits": 150,
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
    #await update_free_trial()
    #await email_marketing()
    await update_discord_roles()
    await downgrade_user()
    await delete_old_notifications()

def main():
    if '--refresh' in sys.argv:
        asyncio.run(refresh_bulk_credits())
    else:
        asyncio.run(run_all_except_refresh())

if __name__ == '__main__':
    main()