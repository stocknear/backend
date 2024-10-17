
from datetime import datetime, time as datetime_time
import schedule
import time
import subprocess
import threading  # Import threading module for parallel execution
import logging  # Import logging module
from logging.handlers import RotatingFileHandler


from dotenv import load_dotenv
import os
load_dotenv()

# Create a dictionary to store the status of each job
job_status = {
    'options_flow_job': {'running': False},
}

useast_ip_address = os.getenv('USEAST_IP_ADDRESS')


# Setup logging
log_file = 'logs/cron_job.log'
logger = logging.getLogger('CronJobLogger')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)  # 5MB per file, 5 backup files
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



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
    logger.info(f"Command: {' '.join(command)}")
    logger.info("Output:\n" + stdout)
    if stderr:
        logger.error("Error:\n" + stderr)



def run_cron_insider_trading():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_insider_trading.py"])

def run_congress_trading():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_congress_trading.py"])

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
    if week <= 5:
        run_command(["python3", "cron_var.py"])

def run_cron_sector():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_sector.py"])

def run_cron_industry():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_industry.py"])

def run_analyst_estimate():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_analyst_estimate.py"])

def run_shareholders():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_shareholders.py"])

def run_share_statistics():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_share_statistics.py"])


def run_cron_market_news():
    run_command(["python3", "cron_market_news.py"])
    run_command(["python3", "cron_company_news.py"])

def run_cron_heatmap():
    run_command(["python3", "cron_heatmap.py"])


def run_cron_options_flow():
    week = datetime.today().weekday()
    current_time = datetime.now().time()
    start_time = datetime_time(15, 30)
    end_time = datetime_time(22, 30)

    if week <= 4 and start_time <= current_time < end_time:
        run_command(["python3", "cron_options_flow.py"])

        
def run_ta_rating():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_ta_rating.py"])


def run_similar_stocks():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_similar_stocks.py"])

def run_historical_price():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_historical_price.py"])

def run_one_day_price():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_one_day_price.py"])

def run_sec_filings():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_sec_filings.py"])

def run_executive():
    week = datetime.today().weekday()
    if week <= 4:
        run_command(["python3", "cron_executive.py"])

def run_options_bubble_ticker():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_options_bubble.py"])


def run_analyst_rating():
    run_command(["python3", "cron_analyst_insight.py"])
    run_command(["python3", "cron_analyst_db.py"])
    run_command(["python3", "cron_analyst_ticker.py"])

def run_market_moods():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_bull_bear_say.py"])
        run_command(["python3", "cron_wiim.py"])


def run_db_schedule_job():
    #update db daily
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["bash", "run_universe.sh"])


def run_dark_pool():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_dark_pool.py"])


def run_dark_pool_flow():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_dark_pool_flow.py"])

def run_market_maker():
    run_command(["python3", "cron_market_maker.py"])

def run_ownership_stats():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_ownership_stats.py"])

def run_clinical_trial():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_clinical_trial.py"])

def run_fda_calendar():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_fda_calendar.py"])

def run_borrowed_share():
    run_command(["python3", "cron_borrowed_share.py"])

'''
def run_implied_volatility():
    run_command(["python3", "cron_implied_volatility.py"])
    command = [
        "sudo", "rsync", "-avz", "-e", "ssh",
        "/root/backend/app/json/implied-volatility",
        f"root@{useast_ip_address}:/root/backend/app/json"
    ]
    run_command(command)
'''

def run_options_net_flow():
    run_command(["python3", "cron_options_net_flow.py"])

def run_options_gex():
    run_command(["python3", "cron_options_gex.py"])
    run_command(["python3", "cron_options_historical_flow.py"])
    


def run_government_contract():
    run_command(["python3", "cron_government_contract.py"])

def run_hedge_fund():
    run_command(["python3", "cron_hedge_funds.py"])

def run_dashboard():
    week = datetime.today().weekday()
    if week <= 5:
        run_command(["python3", "cron_quote.py"])
        run_command(["python3", "cron_market_movers.py"])
        run_command(["python3", "cron_dashboard.py"])

def run_tracker():
    # Run Python scripts
    scripts = [
        "cron_reddit_tracker.py",
        "cron_reddit_statistics.py",
        "cron_cramer_tracker.py",
        "cron_lobbying_tracker.py",
        "cron_sentiment_tracker.py"
    ]
    for script in scripts:
        run_command(["python3", script])

def run_financial_statements():
    run_command(["python3", "cron_financial_statements.py"])

def run_financial_score():
    run_command(["python3", "cron_financial_score.py"])
 

def run_market_cap():
    run_command(["python3", "cron_market_cap.py"])



def run_dividends():
    run_command(["python3", "cron_dividends.py"])


def run_earnings():
    run_command(["python3", "cron_earnings.py"])


def run_fomc_impact():
    run_command(["python3", "cron_fomc_impact.py"])


def run_economy_indicator():
    run_command(["python3", "cron_economic_indicator.py"])



def run_sentiment_analysis():
    run_command(["python3", "cron_sentiment_analysis.py"])


def run_price_analysis():
    run_command(["python3", "cron_price_analysis.py"])


def run_ai_score():
    run_command(["python3", "cron_ai_score.py"])
    run_command(["python3", "cron_stockdeck.py"])
    run_command(["python3", "restart_json.py"])

# Create functions to run each schedule in a separate thread
def run_threaded(job_func):
    job_thread = threading.Thread(target=job_func)
    job_thread.start()

# Schedule the job to run

schedule.every().day.at("01:00").do(run_threaded, run_options_bubble_ticker).tag('options_ticker_job')
schedule.every().day.at("02:00").do(run_threaded, run_db_schedule_job)
schedule.every().day.at("03:00").do(run_threaded, run_dark_pool)
schedule.every().day.at("05:00").do(run_threaded, run_options_gex).tag('options_gex_job')
schedule.every().day.at("06:00").do(run_threaded, run_historical_price).tag('historical_job')

schedule.every().day.at("06:30").do(run_threaded, run_ai_score).tag('ai_score_job')

schedule.every().day.at("07:00").do(run_threaded, run_ta_rating).tag('ta_rating_job')
schedule.every().day.at("09:00").do(run_threaded, run_hedge_fund).tag('hedge_fund_job')
schedule.every().day.at("07:30").do(run_threaded, run_government_contract).tag('government_contract_job')
schedule.every().day.at("07:30").do(run_threaded, run_financial_statements).tag('financial_statements_job')
schedule.every().day.at("08:00").do(run_threaded, run_economy_indicator).tag('economy_indicator_job')
schedule.every().day.at("08:00").do(run_threaded, run_cron_insider_trading).tag('insider_trading_job')
schedule.every().day.at("08:30").do(run_threaded, run_dividends).tag('dividends_job')
schedule.every().day.at("08:30").do(run_threaded, run_fomc_impact).tag('fomc_impact_job')
schedule.every().day.at("09:00").do(run_threaded, run_congress_trading).tag('congress_job')
schedule.every().day.at("10:00").do(run_threaded, run_shareholders).tag('shareholders_job')
schedule.every().day.at("10:30").do(run_threaded, run_sec_filings).tag('sec_filings_job')
schedule.every().day.at("11:00").do(run_threaded, run_executive).tag('executive_job')
schedule.every().day.at("11:45").do(run_threaded, run_clinical_trial).tag('clinical_trial_job')
schedule.every().day.at("12:00").do(run_threaded, run_market_cap).tag('market_cap_josb')

#schedule.every().day.at("05:00").do(run_threaded, run_implied_volatility).tag('implied_volatility_job')


schedule.every().day.at("13:40").do(run_threaded, run_analyst_estimate).tag('analyst_estimate_job')
schedule.every().day.at("13:45").do(run_threaded, run_similar_stocks).tag('similar_stocks_job')
schedule.every().day.at("14:00").do(run_threaded, run_cron_var).tag('var_job')
schedule.every().day.at("14:00").do(run_threaded, run_cron_sector).tag('sector_job')


schedule.every(2).days.at("01:00").do(run_threaded, run_market_maker).tag('markt_maker_job')
schedule.every(2).days.at("08:30").do(run_threaded, run_financial_score).tag('financial_score_job')
schedule.every().saturday.at("05:00").do(run_threaded, run_ownership_stats).tag('ownership_stats_job')
schedule.every().saturday.at("06:00").do(run_threaded, run_sentiment_analysis).tag('sentiment_analysis_job')
schedule.every().saturday.at("10:00").do(run_threaded, run_price_analysis).tag('price_analysis_job')


schedule.every(30).minutes.do(run_threaded, run_dividend_list).tag('dividend_list_job')
schedule.every(15).minutes.do(run_threaded, run_cron_market_news).tag('market_news_job')
schedule.every(30).minutes.do(run_threaded, run_cron_industry).tag('industry_job')

schedule.every(10).minutes.do(run_threaded, run_one_day_price).tag('one_day_price_job')
schedule.every(15).minutes.do(run_threaded, run_cron_heatmap).tag('heatmap_job')


schedule.every(10).minutes.do(run_threaded, run_tracker).tag('tracker_job')


schedule.every(15).minutes.do(run_threaded, run_market_moods).tag('market_moods_job')
schedule.every(10).minutes.do(run_threaded, run_earnings).tag('earnings_job')
#schedule.every(10).minutes.do(run_threaded, run_dark_pool_flow).tag('dark_pool_flow_job')

#schedule.every(2).hours.do(run_threaded, run_fda_calendar).tag('fda_calendar_job')
schedule.every(3).hours.do(run_threaded, run_options_net_flow).tag('options_net_flow_job')
#schedule.every(4).hours.do(run_threaded, run_share_statistics).tag('share_statistics_job')
#schedule.every(2).days.at("01:00").do(run_borrowed_share).tag('borrowed_share_job')

schedule.every(12).hours.do(run_threaded, run_analyst_rating).tag('analyst_job')


schedule.every(2).minutes.do(run_threaded, run_dashboard).tag('dashboard_job')

schedule.every(20).seconds.do(run_threaded, run_if_not_running(run_cron_options_flow, 'options_flow_job')).tag('options_flow_job')


# Run the scheduled jobs indefinitelyp
while True:
    schedule.run_pending()
    time.sleep(3)