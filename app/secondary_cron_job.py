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
api_key = os.getenv('FMP_API_KEY')

berlin_tz = pytz.timezone('Europe/Berlin')

# Set the system's timezone to Berlin at the beginning
subprocess.run(["timedatectl", "set-timezone", "Europe/Berlin"])

def run_restart_cache():
    #update db daily
    week = datetime.today().weekday()
    if week <= 5:
        subprocess.run(["pm2", "restart","fastapi"])
        subprocess.run(["pm2", "restart","fastify"])


def run_json_job():
    # Run the asynchronous function inside an asyncio loop
    subprocess.run(["python3", "restart_json.py"])
    subprocess.run(["pm2", "restart","fastapi"])


# Create functions to run each schedule in a separate thread
def run_threaded(job_func):
    job_thread = threading.Thread(target=job_func)
    job_thread.start()


schedule.every().day.at("15:45").do(run_threaded, run_restart_cache)
schedule.every(3).hours.do(run_threaded, run_json_job).tag('json_job')

while True:
    schedule.run_pending()
    time.sleep(3)