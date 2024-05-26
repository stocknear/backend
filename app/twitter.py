from dotenv import load_dotenv
import os
import tweepy
from requests_oauthlib import OAuth1Session
from benzinga import financial_data
from datetime import datetime, timedelta
from collections import defaultdict
import requests
import json
import ujson
import sqlite3
load_dotenv()

api_key = os.getenv('BENZINGA_API_KEY')
fin = financial_data.Benzinga(api_key)

consumer_key = os.getenv('TWITTER_API_KEY')
consumer_secret = os.getenv('TWITTER_API_SECRET')
access_token = os.getenv('TWITTER_ACCESS_TOKEN')
access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

con = sqlite3.connect('stocks.db')
cursor = con.cursor()
cursor.execute("PRAGMA journal_mode = wal")
cursor.execute("SELECT DISTINCT symbol FROM stocks")
stock_symbols = [row[0] for row in cursor.fetchall()]


def send_tweet(message):
    # Be sure to add replace the text of the with the text you wish to Tweet. You can also add parameters to post polls, quote Tweets, Tweet with reply settings, and Tweet to Super Followers in addition to other features.
    payload = {"text": message}

    # Make the request
    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=access_token,
        resource_owner_secret=access_token_secret,
    )

    # Making the request
    response = oauth.post(
        "https://api.twitter.com/2/tweets",
        json=payload,
    )

    if response.status_code != 201:
        raise Exception(
            "Request returned an error: {} {}".format(response.status_code, response.text)
        )

    print("Response code: {}".format(response.status_code))

    # Saving the response as JSON
    json_response = response.json()
    print(json.dumps(json_response, indent=4, sort_keys=True))


def get_news():
    start_date = datetime.today().strftime('%Y-%m-%d')
    end_date = start_date
    url = "https://api.benzinga.com/api/v2/news"
    querystring = {"token":api_key,"channels":"WIIM","dateFrom":start_date,"dateTo":end_date}
    headers = {"accept": "application/json"}
    response = requests.request("GET", url, headers=headers, params=querystring)
    data = ujson.loads(response.text)
    
    res_list = []
    for item in data:
        title = item['title']
        stock_names = ' '.join(['$' + stock['name'] for stock in item['stocks']])
        message = '{} {}'.format(stock_names, title)
        send_tweet(message)
        print(message)

def get_analyst_ratings():
    url = "https://api.benzinga.com/api/v2.1/calendar/ratings"
    querystring = {"token":api_key,"parameters[date_from]":"2024-04-16","parameters[date_to]":"2024-04-16"}
    headers = {"accept": "application/json"}
    response = requests.request("GET", url, headers=headers, params=querystring)
    data = ujson.loads(response.text)['ratings']
    
    for item in data:
        symbol = item['ticker']
        try:
            item['adjusted_pt_current'] = round(float(item['adjusted_pt_current']))
            item['adjusted_pt_prior'] = round(float(item['adjusted_pt_prior']))
        except:
            pass
        if item['rating_current'] == 'Strong Sell' or item['rating_current'] == 'Strong Buy':
            pass
        elif item['rating_current'] == 'Neutral':
            item['rating_current'] = 'Hold'
        elif item['rating_current'] == 'Equal-Weight' or item['rating_current'] == 'Sector Weight' or item['rating_current'] == 'Sector Perform':
            item['rating_current'] = 'Hold'
        elif item['rating_current'] == 'In-Line':
            item['rating_current'] = 'Hold'
        elif item['rating_current'] == 'Outperform' and item['action_company'] == 'Downgrades':
            item['rating_current'] = 'Hold'
        elif item['rating_current'] == 'Negative':
            item['rating_current'] = 'Sell'
        elif (item['rating_current'] == 'Outperform' or item['rating_current'] == 'Overweight') and (item['action_company'] == 'Reiterates' or item['action_company'] == 'Initiates Coverage On'):
            item['rating_current'] = 'Buy'
            item['action_comapny'] = 'Initiates'
        elif item['rating_current'] == 'Market Outperform' and (item['action_company'] == 'Maintains' or item['action_company'] == 'Reiterates'):
            item['rating_current'] = 'Buy'
        elif item['rating_current'] == 'Outperform' and (item['action_company'] == 'Maintains' or item['action_pt'] == 'Announces' or item['action_company'] == 'Upgrades'):
            item['rating_current'] = 'Buy'
        elif item['rating_current'] == 'Buy' and (item['action_company'] == 'Raises' or item['action_pt'] == 'Raises'):
            item['rating_current'] = 'Strong Buy'
        elif item['rating_current'] == 'Overweight' and (item['action_company'] == 'Maintains' or item['action_company'] == 'Upgrades' or item['action_company'] == 'Reiterates' or item['action_pt'] == 'Raises'):
            item['rating_current'] = 'Buy'
        elif item['rating_current'] == 'Positive' or item['rating_current'] == 'Sector Outperform':
            item['rating_current'] = 'Buy'
        elif item['rating_current'] == 'Underperform' or item['rating_current'] == 'Underweight':
            item['rating_current'] = 'Sell'
        elif item['rating_current'] == 'Reduce' and (item['action_company'] == 'Downgrades' or item['action_pt'] == 'Lowers'):
            item['rating_current'] = 'Sell'
        elif item['rating_current'] == 'Sell' and item['action_pt'] == 'Announces':
            item['rating_current'] = 'Strong Sell'
        elif item['rating_current'] == 'Market Perform':
            item['rating_current'] = 'Hold'
        elif item['rating_prior'] == 'Outperform' and item['action_company'] == 'Downgrades':
            item['rating_current'] = 'Hold'
        elif item['rating_current'] == 'Peer Perform' and item['rating_prior'] == 'Peer Perfrom':
            item['rating_current'] = 'Hold'
        elif item['rating_current'] == 'Peer Perform' and item['action_pt'] == 'Announces':
            item['rating_current'] = 'Hold'
            item['action_comapny'] = 'Initiates'
        if symbol in stock_symbols:
            message = f"{item['action_company']} {item['rating_current']} rating on ${item['ticker']} from ${item['adjusted_pt_prior']} to ${item['adjusted_pt_current']} new Price Target from {item['analyst']}."
            #print(message)
            send_tweet(message)


def get_analyst_insight():
    url = "https://api.benzinga.com/api/v1/analyst/insights"
    querystring = {"token": api_key}
    response = requests.request("GET", url, params=querystring)
    data = ujson.loads(response.text)['analyst-insights']
    #print(data)
    for item in data:
        try:
            symbol = item['security']['symbol']
            with open(f"json/trend-analysis/{symbol}.json", 'r') as file:
                ai_model = ujson.load(file)
                for i in ai_model:
                    if i['label']== 'threeMonth':
                        sentiment = i['sentiment']
                        accuracy = i['accuracy']
            if symbol in stock_symbols:
                #tweet = f"{item['action']} {item['rating']} rating on ${item['security']['symbol']} with ${item['pt']} Price Target from {item['firm']}. \
                #\nOur own AI Model predicts a {sentiment} Trend for the next 3 months with an accuracy of {accuracy}%."
                message = f"{item['action']} {item['rating']} rating on ${item['security']['symbol']} with ${item['pt']} Price Target from {item['firm']}."
                print(message)
                #send_tweet(message)
        except:
            pass

def get_biggest_options_activity():
    # Initialize dictionaries to store cumulative sums and counts
    call_volume_sum = defaultdict(int)
    put_volume_sum = defaultdict(int)
    volume_sum = defaultdict(int)
    open_interest_sum = defaultdict(int)
    price_sum = defaultdict(float)
    cost_basis_sum = defaultdict(float)
    call_count = defaultdict(int)
    put_count = defaultdict(int)

    try:
        end_date = datetime.today()
        start_date = (end_date -timedelta(10))
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        res_list = []
        for page in range(0,50):
            try:
                data = fin.options_activity(date_from=start_date_str, date_to=end_date_str, page=0, pagesize=1000)
                data = ujson.loads(fin.output(data))['option_activity']
                res_list +=data                
            except:
                break

        #filtered_data = [{key: value for key, value in item.items() if key in ['ticker','cost_basis','date_expiration','open_interest','price', 'put_call','strike_price', 'volume']} for item in res_list]
        # Iterate through the data
        for item in res_list:
            ticker = item['ticker']
            if item['put_call'] == 'CALL':
                call_volume_sum[ticker] += int(item['volume'])
                call_count[ticker] += 1
            elif item['put_call'] == 'PUT':
                put_volume_sum[ticker] += int(item['volume'])
                put_count[ticker] += 1
            volume_sum[ticker] += int(item['volume'])
            #open_interest_sum[ticker] += int(item['open_interest'])
            #price_sum[ticker] += float(item['price'])
            #cost_basis_sum[ticker] += float(item['cost_basis'])
        
        sorted_volume = sorted(volume_sum.items(), key=lambda x: x[1], reverse=True)
        output = []
        for i, (ticker, volume) in enumerate(sorted_volume[:3], 1):
            flow_sentiment = 'Neutral'
            if put_volume_sum[ticker] > call_volume_sum[ticker]:
                flow_sentiment = 'Bearish'
            elif put_volume_sum[ticker] < call_volume_sum[ticker]:
                flow_sentiment = 'Bullish'

            output.append(f"{i}) ${ticker}\n \
            - Call Flow: {call_volume_sum[ticker]:,}\n \
            - Put Flow: {put_volume_sum[ticker]:,}\n \
            - Put/Call Ratio: {round(put_volume_sum[ticker]/call_volume_sum[ticker],2)}\n \
            - Flow Sentiment: {flow_sentiment}")

        message = f"Market Recap: Top 3 Highest Options Activity from this Week\n\
        {output[0]}\n\
        {output[1]}\n\
        {output[2]}"
        print(message)
        #send_tweet(message)

    except Exception as e:
        print(e)




if __name__ == '__main__':
    get_news()
    #get_analyst_insight()
    #get_analyst_ratings()
    #get_biggest_options_activity()