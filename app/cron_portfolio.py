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

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')
pb_admin_email = os.getenv('POCKETBASE_ADMIN_EMAIL')
pb_password = os.getenv('POCKETBASE_PASSWORD')

#berlin_tz = pytz.timezone('Europe/Berlin')
new_york_tz = pytz.timezone('America/New_York')
pb = PocketBase('http://127.0.0.1:8090')
admin_data = pb.admins.auth_with_password(pb_admin_email, pb_password)

# Set the system's timezone to Berlin at the beginning
subprocess.run(["timedatectl", "set-timezone", "Europe/Berlin"])

async def get_quote_of_stocks(ticker_list):
    ticker_str = ','.join(ticker_list)
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v3/quote/{ticker_str}?apikey={api_key}" 
        async with session.get(url) as response:
            df = await response.json()
    return df

def check_number_of_shares(holdings, trading_history):
    # Create a dictionary to track the current number of shares for each symbol
    share_count = {}
    # Update share count based on history
    for transaction in trading_history:
        symbol = transaction["symbol"]
        num_shares = transaction["numberOfShares"]
        if transaction["type"] == "buy":
            # Increment the share count for the symbol
            share_count[symbol] = share_count.get(symbol, 0) + num_shares
        elif transaction["type"] == "sell":
            # Decrement the share count for the symbol
            share_count[symbol] = share_count.get(symbol, 0) - num_shares

    # Update the holdings list based on the share count
    for holding in holdings:
        symbol = holding["symbol"]
        if symbol in share_count:
            holding["numberOfShares"] = share_count[symbol]

    return holdings

def compute_available_cash(transactions):
    available_cash = 100000  # Initial available cash
    for transaction in transactions:
        if transaction['type'] == 'buy':
            shares_bought = transaction['numberOfShares']
            price_per_share = transaction['price']
            total_cost = shares_bought * price_per_share
            available_cash -= total_cost
        elif transaction['type'] == 'sell':
            shares_sold = transaction['numberOfShares']
            price_per_share = transaction['price']
            total_gain = shares_sold * price_per_share
            available_cash += total_gain

    return available_cash


def compute_overall_return(initial_budget, transactions):
    current_budget = initial_budget

    for transaction in transactions:
        if transaction["type"] == "buy":
            current_budget -= transaction["numberOfShares"] * transaction["price"]
        elif transaction["type"] == "sell":
            current_budget += transaction["numberOfShares"] * transaction["price"]

    overall_return = (current_budget - initial_budget) / initial_budget * 100
    #print('overall return: ', overall_return)
    return overall_return


async def update_portfolio():
    current_time = datetime.now(new_york_tz)
    current_weekday = current_time.weekday()

    initial_budget = 100000

    opening_hour = 9
    opening_minute = 30
    closing_hour = 17

    is_market_open = ( current_time.hour > opening_hour or (current_time.hour == opening_hour and current_time.minute >= opening_minute)) and current_time.hour < closing_hour
    if current_weekday <= 5 and is_market_open:
        # Get the current date
        current_month = datetime.today()
        # Set the day to 1 to get the beginning of the current month
        beginning_of_month = current_month.replace(day=1)
        # Format it as a string if needed
        formatted_date = beginning_of_month.strftime("%Y-%m-%d")

        result =  pb.collection("portfolios").get_full_list(query_params = {"filter": f'created >= "{formatted_date}"'})

        ranking_list = []
        ticker_list = []
        if len(result) != 0:
            #get all tickers from all portfolios
            for port in result:
                if len(port.holdings) != 0:
                    ticker_list += [i['symbol'] for i in port.holdings]
            ticker_list = list(set(ticker_list))
            #unique ticker_list
            data = await get_quote_of_stocks(ticker_list)
            #Get all quotes in bulks to save api calls
            
            for x in result:
                if len(x.trading_history) > 0:
                    try:
                        if len(x.holdings) != 0:

                            #compute the correct available cash to avoid bugs
                            x.available_cash = compute_available_cash(x.trading_history)

                            account_value = x.available_cash
                            quote_data_dict = {dd['symbol']: dd for dd in data}
                            #compute the correct number of shares to avoid bugs
                            x.holdings = check_number_of_shares(x.holdings, x.trading_history)
                            
                            for item in x.holdings:
                                dd = quote_data_dict.get(item['symbol'])
                                if dd:
                                    current_price = dd['price']
                                    since_bought_change = round((current_price/ item['boughtPrice']  - 1) * 100, 2)
                                    account_value += current_price * item['numberOfShares']

                                    # Update holdings_list
                                    item['currentPrice'] = current_price
                                    item['sinceBoughtChange'] = since_bought_change

                            overall_return = round( ( account_value/initial_budget -1) * 100 ,2)
                            #Update Pocketbase with new values

                            pb.collection("portfolios").update(x.id, {
                              "accountValue": account_value,
                              "overallReturn": overall_return,
                              "availableCash": x.available_cash,
                              "holdings": x.holdings,
                            })
                        else:
                            #overall_return = x.overall_return
                            overall_return = compute_overall_return(initial_budget, x.trading_history)
                            account_value = round(initial_budget*(1+overall_return/100),2)
                            available_cash = account_value

                            pb.collection("portfolios").update(x.id, {
                              "accountValue": account_value,
                              "overallReturn": overall_return,
                              "availableCash": available_cash,
                            })
                        
                        ranking_list.append({'userId': x.id, 'overallReturn': overall_return})

                    except Exception as e:
                        print(e)
                else:
                    pass

            #Apply ranking to each user
            sorted_ranking_list = sorted(ranking_list, key=lambda x: x['overallReturn'], reverse=True)
            for rank, item in enumerate(sorted_ranking_list):
                pb.collection("portfolios").update(item['userId'], {
                      "rank": rank+1,
                    })
            print('Done')
        else:
            print('Market Closed')


asyncio.run(update_portfolio())