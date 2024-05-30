import pytz
from datetime import datetime, timedelta
from urllib.request import urlopen
import certifi
import json
import ujson
import schedule
import time
import subprocess
import asyncio
import aiohttp
import pytz
import sqlite3
import pandas as pd
import numpy as np
import threading  # Import threading module for parallel execution


from dotenv import load_dotenv
import os
load_dotenv()

useast_ip_address = os.getenv('USEAST_IP_ADDRESS')



# Set the system's timezone to Berlin at the beginning
subprocess.run(["timedatectl", "set-timezone", "Europe/Berlin"])


def run_json_job():
    # Run the asynchronous function inside an asyncio loop
    subprocess.run(["python3", "restart_json.py"])
    subprocess.run(["pm2", "restart","fastapi"])

def run_cron_insider_trading():
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["python3", "cron_insider_trading.py"])
        command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/insider-trading",
            f"root@{useast_ip_address}:/root/backend/app/json"
        ]
        subprocess.run(command)

def run_congress_trading():
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["python3", "cron_congress_trading.py"])
        command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/congress-trading",
            f"root@{useast_ip_address}:/root/backend/app/json"
        ]
        subprocess.run(command)

def run_cron_var():
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["python3", "cron_var.py"])
        command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/var",
            f"root@{useast_ip_address}:/root/backend/app/json"
        ]
        subprocess.run(command)

def run_cron_market_movers():
    week = datetime.today().weekday()
    if week <= 4:
        subprocess.run(["python3", "cron_market_movers.py"])
        command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/market-movers",
            f"root@{useast_ip_address}:/root/backend/app/json"
        ]
        subprocess.run(command)

    
def run_cron_market_news():
    subprocess.run(["python3", "cron_market_news.py"])
    command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/market-news",
            f"root@{useast_ip_address}:/root/backend/app/json"
    ]
    subprocess.run(command)

def run_cron_heatmap():
    week = datetime.today().weekday()
    if week <= 4:
        subprocess.run(["python3", "cron_heatmap.py"])
        command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/heatmaps",
            f"root@{useast_ip_address}:/root/backend/app/json"
        ]
        subprocess.run(command)

def run_cron_quote():
    week = datetime.today().weekday()
    if week <= 6:
        subprocess.run(["python3", "cron_quote.py"])
        command = ["sudo", "rsync", "-avz", "-e", "ssh", "/root/backend/app/json/quote", f"root@{useast_ip_address}:/root/backend/app/json"]
        subprocess.run(command)

def run_cron_price_alert():
    week = datetime.today().weekday()
    if week <= 4:
        subprocess.run(["python3", "cron_price_alert.py"])

def run_cron_portfolio():
    week = datetime.today().weekday()
    if week <= 4:
        subprocess.run(["python3", "cron_portfolio.py"])

def run_cron_options_flow():
    week = datetime.today().weekday()
    if week <= 4:
        subprocess.run(["python3", "cron_options_flow.py"])        
        
        command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/options-flow/feed/",
            f"root@{useast_ip_address}:/root/backend/app/json/options-flow/feed/"
        ]
        subprocess.run(command)
        
        
def run_cron_options_zero_dte():
    week = datetime.today().weekday()
    if week <= 4:
        subprocess.run(["python3", "cron_options_zero_dte.py"])
        command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/options-flow/zero-dte/",
            f"root@{useast_ip_address}:/root/backend/app/json/options-flow/zero-dte/"
        ]
        subprocess.run(command)
                  

def run_ta_rating():
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["python3", "cron_ta_rating.py"])
        command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/ta-rating",
            f"root@{useast_ip_address}:/root/backend/app/json"
        ]
        subprocess.run(command)

def run_stockdeck():
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["python3", "cron_stockdeck.py"])
        command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/stockdeck",
            f"root@{useast_ip_address}:/root/backend/app/json"
        ]
        subprocess.run(command)

def run_similar_stocks():
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["python3", "cron_similar_stocks.py"])
        command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/similar-stocks",
            f"root@{useast_ip_address}:/root/backend/app/json"
        ]
        subprocess.run(command)

def run_historical_price():
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["python3", "cron_historical_price.py"])
        command = [
            "sudo", "rsync", "-avz", "-e", "ssh",
            "/root/backend/app/json/historical-price",
            f"root@{useast_ip_address}:/root/backend/json"
        ]
        subprocess.run(command)

def run_one_day_price():
    week = datetime.today().weekday()
    if week <= 6:
        subprocess.run(["python3", "cron_one_day_price.py"])
        command = ["sudo", "rsync", "-avz", "-e", "ssh", "/root/backend/app/json/one-day-price/", f"root@{useast_ip_address}:/root/backend/app/json/one-day-price/"]
        subprocess.run(command)

def run_options_bubble_ticker():
    week = datetime.today().weekday()
    if week <= 4:
        subprocess.run(["python3", "cron_options_bubble.py"])

        command = ["sudo", "rsync", "-avz", "-e", "ssh", "/root/backend/app/json/options-bubble/", f"root@{useast_ip_address}:/root/backend/app/json/options-bubble/"]
        subprocess.run(command)

        command = ["sudo", "rsync", "-avz", "-e", "ssh", "/root/backend/app/json/options-flow/company/", f"root@{useast_ip_address}:/root/backend/app/json/options-flow/company/"]
        subprocess.run(command)

def run_analyst_rating():
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["python3", "cron_analyst_db.py"])
        subprocess.run(["python3", "cron_analyst_ticker.py"])
        command = ["sudo", "rsync", "-avz", "-e", "ssh", "/root/backend/app/json/analyst", f"root@{useast_ip_address}:/root/backend/app/json"]
        subprocess.run(command)

def run_market_moods():
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["python3", "cron_bull_bear_say.py"])
        subprocess.run(["python3", "cron_wiim.py"])
        command = ["sudo", "rsync", "-avz", "-e", "ssh", "/root/backend/app/json/bull_bear_say", f"root@{useast_ip_address}:/root/backend/app/json"]
        subprocess.run(command)
        command = ["sudo", "rsync", "-avz", "-e", "ssh", "/root/backend/app/json/wiim", f"root@{useast_ip_address}:/root/backend/app/json"]
        subprocess.run(command)


def run_db_schedule_job():
    #update db daily
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["bash", "run_universe.sh"])

def run_restart_cache():
    #update db daily
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["pm2", "restart","fastapi"])
        subprocess.run(["pm2", "restart","fastify"])
        #subprocess.run(["python3", "cache_endpoints.py"])

# Create functions to run each schedule in a separate thread
def run_threaded(job_func):
    job_thread = threading.Thread(target=job_func)
    job_thread.start()

# Schedule the job to run

schedule.every().day.at("01:00").do(run_threaded, run_options_bubble_ticker).tag('options_ticker_job')
schedule.every().day.at("02:00").do(run_threaded, run_db_schedule_job)
schedule.every().day.at("06:00").do(run_threaded, run_historical_price).tag('historical_job')
schedule.every().day.at("07:00").do(run_threaded, run_ta_rating).tag('ta_rating_job')
schedule.every().day.at("08:00").do(run_threaded, run_cron_insider_trading).tag('insider_trading_job')
schedule.every().day.at("09:00").do(run_threaded, run_congress_trading).tag('congress_job')

schedule.every().day.at("13:30").do(run_threaded, run_stockdeck).tag('stockdeck_job')
schedule.every().day.at("13:45").do(run_threaded, run_similar_stocks).tag('similar_stocks_job')
schedule.every().day.at("14:00").do(run_threaded, run_cron_var).tag('var_job')


schedule.every().day.at("15:45").do(run_threaded, run_restart_cache)
schedule.every(1).minutes.do(run_threaded, run_cron_portfolio).tag('portfolio_job')
schedule.every(5).minutes.do(run_threaded, run_cron_market_movers).tag('market_movers_job')

schedule.every(15).minutes.do(run_threaded, run_cron_market_news).tag('market_news_job')
schedule.every(10).minutes.do(run_threaded, run_one_day_price).tag('one_day_price_job')
schedule.every(5).minutes.do(run_threaded, run_cron_heatmap).tag('heatmap_job')

schedule.every(1).minutes.do(run_threaded, run_cron_quote).tag('quote_job')
schedule.every(1).minutes.do(run_threaded, run_cron_price_alert).tag('price_alert_job')
schedule.every(15).minutes.do(run_threaded, run_market_moods).tag('market_moods_job')
schedule.every(3).hours.do(run_threaded, run_json_job).tag('json_job')
schedule.every(6).hours.do(run_threaded, run_analyst_rating).tag('analyst_job')

schedule.every(10).seconds.do(run_threaded, run_cron_options_flow).tag('options_flow_job')
schedule.every(10).seconds.do(run_threaded, run_cron_options_zero_dte).tag('options_zero_dte_job')


# Run the scheduled jobs indefinitelyp
while True:
    schedule.run_pending()
    time.sleep(3)