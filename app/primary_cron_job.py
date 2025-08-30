
from datetime import datetime, time as datetime_time
import schedule
import time
import subprocess
import threading  # Import threading module for parallel execution
#import logging  # Import logging module
#from logging.handlers import RotatingFileHandler
from pytz import timezone

from dotenv import load_dotenv
import os
load_dotenv()

# Create a dictionary to store the status of each job
job_status = {
    'options_flow_job': {'running': False},
}

ny_tz = timezone('America/New_York')
berlin_tz = timezone('Europe/Berlin')

# Setup logging
'''
log_file = 'logs/cron_job.log'
logger = logging.getLogger('CronJobLogger')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)  # 5MB per file, 5 backup files
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
'''


# Set the system's timezone to Berlin at the beginning
subprocess.run(["timedatectl", "set-timezone", "Europe/Berlin"])


def run_if_not_running(job_func, job_tag):
    def wrapper():
        if not job_status[job_tag]['running']:
            job_status[job_tag]['running'] = True
            try:
                job_func()
            finally:
                job_status[job_tag]['running'] = False
    return wrapper

# Function to run commands and log output
def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # Log stdout and stderr
    '''
    logger.info(f"Command: {' '.join(command)}")
    logger.info("Output:\n" + stdout)
    if stderr:
        logger.error("Error:\n" + stderr)
    '''

def run_dark_pool_flow():
    now = datetime.now(ny_tz)
    week = now.weekday()
    hour = now.hour
    if week <= 4 and 8 <= hour < 17:
        run_command(["python3", "cron_dark_pool_flow.py"])

def run_market_flow():
    now = datetime.now(ny_tz)
    week = now.weekday()
    current_time = now.time()
    hour = now.hour
    if week <= 4 and 8 <= hour < 17:
        run_command(["python3", "cron_market_flow.py"])


def run_dark_pool_level():
    now = datetime.now(ny_tz)
    week = now.weekday()
    hour = now.hour
    if week <= 4 and 8 <= hour <= 22:
        run_command(["python3", "cron_dark_pool_level.py"])

def run_dark_pool_ticker():
    now = datetime.now(ny_tz)
    week = now.weekday()
    if week <= 5:
        run_command(["python3", "cron_dark_pool_ticker.py"])


def run_options_jobs():
    now = datetime.now(ny_tz)
    week = now.weekday()
    if week <= 5:
        run_command(["python3"," cron_unusual_activity.py"])
        run_command(["python3", "cron_options_single_contract.py"])
        run_command(["python3", "cron_options_max_pain.py"])
        run_command(["python3", "cron_options_chain_statistics.py"])
        run_command(["python3", "cron_options_historical_volume.py"])
        run_command(["python3", "cron_options_hottest_contracts.py"])
        run_command(["python3", "cron_options_oi.py"])
        run_command(["python3", "cron_options_screener.py"])

        run_command(["python3", "cron_implied_volatility.py"])
        run_command(["python3", "cron_options_gex_dex.py"])
        run_command(["python3", "cron_options_contract_lookup.py"])

def run_historical_employees():
    now = datetime.now(ny_tz)
    week = now.weekday()
    if week <= 5:
        run_command(["python3", "cron_historical_employees.py"])

def run_cron_insider_trading():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_insider_trading.py"])

def run_congress_trading():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_congress_trading.py"])
        run_command(["python3", "restart_json.py"])
        run_command(["python3", "cron_average_value.py"])

def run_dividend_list():
    week = datetime.today().weekday()
    current_time = datetime.now().time()
    start_time = datetime_time(15, 30)
    end_time = datetime_time(22, 30)

    if week <= 4 and start_time <= current_time < end_time:
        run_command(["python3", "cron_dividend_kings.py"])
        run_command(["python3", "cron_dividend_aristocrats.py"])

def run_cron_var():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_var.py"])

def run_cron_sector():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_sector.py"])

def run_cron_industry():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_industry.py"])

def run_analyst_estimate():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_analyst_estimate.py"])

def run_shareholders():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_shareholders.py"])

def run_profile():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_profile.py"])

def run_share_statistics():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_share_statistics.py"])

def run_cron_market_news():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_market_news.py"])
        run_command(["python3", "cron_ipo_news.py"])

def run_company_news():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_company_news.py"])

def run_press_releases():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_press_releases.py"])

def run_cron_options_flow():
    run_command(["python3", "cron_options_flow.py"])

        
def run_ta_rating():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_ta_rating.py"])


def run_similar_stocks():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_similar_stocks.py"])

def run_historical_price():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_historical_price.py"])
        run_command(["python3","cron_historical_adj_price.py"])

def run_one_day_price():
    now = datetime.now(ny_tz)
    # only Mon–Fri
    if now.weekday() <= 4:
        # between 09:00 (inclusive) and 16:30 (exclusive)
        if datetime_time(9, 30) <= now.time() < datetime_time(16, 30):
            run_command(["python3", "cron_one_day_price.py"])


def run_analyst_rating():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_analyst_insight.py"])
        run_command(["python3", "cron_analyst_db.py"])
        run_command(["python3", "cron_analyst_ticker.py"])
        run_command(["python3", "cron_analyst_flow.py"])

def run_market_moods():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_wiim.py"])

def run_db_schedule_job():
    #update db daily
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["bash", "run_universe.sh"])


def run_ownership_stats():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_ownership_stats.py"])


def run_options_historical_flow():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_options_historical_flow.py"])
        
    

def run_hedge_fund():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_hedge_funds.py"])

def run_dashboard():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_quote.py"])
        run_command(["python3", "cron_market_movers.py"])
        run_command(["python3", "cron_dashboard.py"])

def run_quote():
    now = datetime.now(ny_tz)
    # only Mon–Fri
    if now.weekday() <= 4:
        current_time = now.time()
        # pre-market: 4:00–9:30
        if datetime_time(4, 0) <= current_time < datetime_time(9, 30):
            run_command(["python3", "cron_quote.py"])
        # post-market: 16:00–20:00
        elif datetime_time(16, 0) <= current_time < datetime_time(20, 0):
            run_command(["python3", "cron_quote.py"])


def run_tracker():

    scripts = [
        "cron_reddit_tracker.py",
        "cron_reddit_statistics.py",
        "cron_potus_tracker.py",
        "cron_fear_and_greed.py",
    ]
    for script in scripts:
        run_command(["python3", script])

    week = datetime.today().weekday()
    if week <= 4:
        scripts = [
            #"cron_cramer_tracker.py",
            #"cron_lobbying_tracker.py",
            #"cron_sentiment_tracker.py",
            "cron_insider_tracker.py",
        ]
        for script in scripts:
            run_command(["python3", script])

        #update screener for moneyness
        run_command(["python3", "cron_options_screener.py","update"])


    #now = datetime.now(ny_tz)
    #week = now.weekday()
    #current_time = now.time()
    #hour = now.hour
    #if week <= 4 and 8 <= hour < 17:
    #    run_command(["python3", "cron_unusual_activity.py"])



def run_list():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_list.py"])
        run_command(["python3", "cron_heatmap.py"])

def run_bot():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_discord_bot.py"])
        run_command(["python3", "cron_twitter_bot.py"])
        run_command(["python3", "cron_facebook_bot.py"])
        run_command(["python3", "cron_bsky_bot.py"])
        #run_command(["python3", "cron_linkedin_bot.py"])
        run_command(["python3", "cron_reddit_bot.py"])



def run_financial_statements():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_financial_statements.py"])
        run_command(["python3", "compute_ttm.py"])

def run_financial_score():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_financial_score.py"])
        run_command(["python3", "cron_business_metrics.py"])
 

def run_market_cap():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_market_cap.py"])
        run_command(["python3", "cron_revenue.py"])


def run_dividends():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_dividends.py"])


def run_earnings():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_earnings.py"])

def run_price_reaction():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_earnings_price_reaction.py"])


def run_economy_indicator():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_economic_indicator.py"])


def run_weekly_and_daily_jobs():
    now = datetime.now(ny_tz)
    weekday = now.weekday()
    if weekday == 5:
        run_command(["python3", "cron_ai_score.py"])
        run_command(["python3", "cron_price_analysis.py"])
        run_command(["python3","cron_etf_sector.py"])
    
    run_command(["python3", "cron_stockdeck.py"])
    run_command(["python3", "restart_json.py"])
    run_command(["python3", "cron_statistics.py"])
    run_command(["python3", "cron_bull_vs_bear.py"])

def run_push_notifications():
    now = datetime.now(ny_tz)
    week = now.weekday()
    hour = now.hour
    if week <= 4 and 5 <= hour < 21:
        run_command(["python3", "cron_push_notifications.py"])
    

# Create functions to run each schedule in a separate thread
def run_threaded(job_func):
    job_thread = threading.Thread(target=job_func)
    job_thread.start()

# Schedule the job to run


schedule.every().day.at("00:00").do(run_threaded, run_db_schedule_job)
schedule.every().day.at("23:30").do(run_threaded, run_options_jobs).tag('options_job')
schedule.every().day.at("04:00").do(run_threaded, run_historical_employees).tag('employees_job')
schedule.every().day.at("08:00").do(run_threaded, run_options_historical_flow).tag('options_historical_flow_job')


schedule.every().day.at("06:00").do(run_threaded, run_historical_price).tag('historical_job')
schedule.every().day.at("06:30").do(run_threaded, run_weekly_and_daily_jobs).tag('weekly_daily_job')

schedule.every().day.at("07:00").do(run_threaded, run_ta_rating).tag('ta_rating_job')
schedule.every().day.at("08:00").do(run_threaded, run_price_reaction).tag('price_reaction_job')
schedule.every().day.at("08:00").do(run_threaded, run_dark_pool_ticker).tag('dark_pool_ticker_job')
schedule.every().day.at("09:00").do(run_threaded, run_hedge_fund).tag('hedge_fund_job')
schedule.every().day.at("07:30").do(run_threaded, run_financial_statements).tag('financial_statements_job')
#schedule.every().day.at("08:00").do(run_threaded, run_economy_indicator).tag('economy_indicator_job')
schedule.every().day.at("08:00").do(run_threaded, run_cron_insider_trading).tag('insider_trading_job')
schedule.every().day.at("08:30").do(run_threaded, run_dividends).tag('dividends_job')
schedule.every().day.at("09:00").do(run_threaded, run_shareholders).tag('shareholders_job')
schedule.every().day.at("09:30").do(run_threaded, run_profile).tag('profile_job')
schedule.every().day.at("10:00").do(run_threaded, run_share_statistics).tag('share_statistics_job')



schedule.every().day.at("12:00").do(run_threaded, run_market_cap).tag('market_cap_job')



schedule.every().day.at("13:40").do(run_threaded, run_analyst_estimate).tag('analyst_estimate_job')
schedule.every().day.at("13:45").do(run_threaded, run_similar_stocks).tag('similar_stocks_job')
schedule.every().day.at("14:00").do(run_threaded, run_cron_var).tag('var_job')
schedule.every().day.at("14:00").do(run_threaded, run_cron_sector).tag('sector_job')


schedule.every(2).days.at("08:30").do(run_threaded, run_financial_score).tag('financial_score_job')
schedule.every().saturday.at("05:00").do(run_threaded, run_ownership_stats).tag('ownership_stats_job')
#schedule.every().saturday.at("06:00").do(run_threaded, run_sentiment_analysis).tag('sentiment_analysis_job')
#schedule.every().saturday.at("10:00").do(run_threaded, run_price_analysis).tag('price_analysis_job')


schedule.every(30).minutes.do(run_threaded, run_dividend_list).tag('dividend_list_job')
schedule.every(3).hours.do(run_threaded, run_congress_trading).tag('congress_job')
schedule.every(30).minutes.do(run_threaded, run_cron_market_news).tag('market_news_job')

schedule.every(30).minutes.do(run_threaded, run_cron_industry).tag('industry_job')

schedule.every(8).minutes.do(run_threaded, run_one_day_price).tag('one_day_price_job')


schedule.every(20).minutes.do(run_threaded, run_tracker).tag('tracker_job')


schedule.every(10).minutes.do(run_threaded, run_market_moods).tag('market_moods_job')
schedule.every(10).minutes.do(run_threaded, run_earnings).tag('earnings_job')


schedule.every(2).hours.do(run_threaded, run_analyst_rating).tag('analyst_job')
schedule.every(1).hours.do(run_threaded, run_company_news).tag('company_news_job')
schedule.every(3).hours.do(run_threaded, run_press_releases).tag('press_release_job')


schedule.every(5).minutes.do(run_threaded, run_push_notifications).tag('push_notifications_job')

schedule.every(5).minutes.do(run_threaded, run_market_flow).tag('market_flow_job')
schedule.every(5).minutes.do(run_threaded, run_list).tag('stock_list_job')
schedule.every(2).minutes.do(run_threaded, run_bot).tag('social_media_bot')


schedule.every(15).minutes.do(run_threaded, run_dark_pool_level).tag('dark_pool_level_job')
schedule.every(10).minutes.do(run_threaded, run_dark_pool_flow).tag('dark_pool_flow_job')

schedule.every(2).minutes.do(run_threaded, run_dashboard).tag('dashboard_job')
schedule.every(3).minutes.do(run_threaded, run_quote).tag('quote_job')


schedule.every(15).seconds.do(run_threaded, run_if_not_running(run_cron_options_flow, 'options_flow_job')).tag('options_flow_job')



# Run the scheduled jobs indefinitelyp
while True:
    schedule.run_pending()
    time.sleep(3)