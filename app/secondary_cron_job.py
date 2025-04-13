import schedule
import time
import subprocess
import threading
from datetime import datetime
import pytz

berlin_tz = pytz.timezone('Europe/Berlin')

# Set the system's timezone to Berlin at the beginning
subprocess.run(["timedatectl", "set-timezone", "Europe/Berlin"])


def run_pocketbase():
    subprocess.run(["python3", "cron_pocketbase.py"])
    subprocess.run(["python3", "cron_notification_channel.py"])

def run_restart_cache():
    subprocess.run(["pm2", "restart", "fastapi"])
    subprocess.run(["pm2", "restart", "fastify"])
    subprocess.run(["pm2", "restart", "websocket"])

def run_json_job():
    subprocess.run(["python3", "restart_json.py"])
    subprocess.run(["pm2", "restart", "fastapi"])
    subprocess.run(["pm2", "restart", "fastify"])
    subprocess.run(["pm2", "restart", "websocket"])

def run_cron_price_alert():
    week = datetime.today().weekday()
    if week <= 4:
        subprocess.run(["python3", "cron_price_alert.py"])

def run_refresh_pocketbase():
    """Runs cron_pocketbase.py with --refresh at the start of each month."""
    now = datetime.now(berlin_tz)
    if now.day == 1:
        subprocess.run(["python3", "cron_pocketbase.py", "--refresh"])


# Run each job in a separate thread
def run_threaded(job_func):
    job_thread = threading.Thread(target=job_func)
    job_thread.start()


# Existing scheduled tasks
schedule.every().day.at("06:30").do(run_threaded, run_pocketbase).tag('pocketbase_job')
schedule.every().day.at("15:30").do(run_threaded, run_restart_cache)
schedule.every().day.at("23:00").do(run_threaded, run_restart_cache)
schedule.every(2).hours.do(run_threaded, run_json_job).tag('json_job')
schedule.every(1).minutes.do(run_threaded, run_cron_price_alert).tag('price_alert_job')

schedule.every().day.at("00:30").do(run_threaded, run_refresh_pocketbase)

# Keep the scheduler running
while True:
    schedule.run_pending()
    time.sleep(3)
