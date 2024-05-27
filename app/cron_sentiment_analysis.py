from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm
from datetime import datetime, timedelta
import asyncio
import aiohttp
import sqlite3
import ujson
import time
import random
from dotenv import load_dotenv
import os
import re 

'''
import nltk
nltk.download('vader_lexicon')
'''

load_dotenv()
api_key = os.getenv('FMP_API_KEY')
sid = SentimentIntensityAnalyzer()


def convert_symbols(symbol_list):
    """
    Converts the symbols in the given list from 'BTCUSD' and 'USDTUSD' format to 'BTC-USD' and 'USDT-USD' format.
    
    Args:
        symbol_list (list): A list of strings representing the symbols to be converted.
    
    Returns:
        list: A new list with the symbols converted to the desired format.
    """
    converted_symbols = []
    for symbol in symbol_list:
        # Determine the base and quote currencies
        base_currency = symbol[:-3]
        quote_currency = symbol[-3:]
        
        # Construct the new symbol in the desired format
        new_symbol = f"{base_currency}-{quote_currency}"
        converted_symbols.append(new_symbol)
    
    return converted_symbols

async def get_news_of_stocks(ticker_list,page):
    ticker_str = ','.join(ticker_list)
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker_str}&page={page}&limit=2000&apikey={api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return []

async def get_news_of_cryptos(ticker_list,page):
    ticker_str = ','.join(ticker_list)
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v4/crypto_news?tickers={ticker_str}&page={page}&limit=2000&apikey={api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return []

def remove_duplicates(data, key):
    seen = set()
    new_data = []
    for item in data:
        if item[key] not in seen:
            seen.add(item[key])
            new_data.append(item)
    return new_data

def adjust_scaled_score(scaled_score):
    #adjustment = random.choice([-2,-1, 0, 1, 2])
    # Add the adjustment to the scaled_score
    #scaled_score += adjustment
    
    # Ensure the scaled_score stays within the range of 0 to 10
    scaled_score = max(0, min(10, scaled_score))
    
    return scaled_score

def compute_sentiment_score(sentence):
    # Compute sentiment score using VADER
    #sentiment_score = sid.polarity_scores(sentence)['compound']
    sentiment_score = TextBlob(sentence).sentiment.polarity
    # Scale the sentiment score to range from 0 to 10
    scaled_score = (sentiment_score + 1) * 5  # Map from [-1, 1] to [0, 10]
    return scaled_score

def get_sentiment(symbol, res_list, is_crypto=False):
    if is_crypto == True:
        time_format = '%Y-%m-%dT%H:%M:%S.%fZ'
    else:
        time_format = '%Y-%m-%d %H:%M:%S'

    end_date = datetime.now().date()
    end_date_datetime = datetime.combine(end_date, datetime.min.time())  # Convert end_date to datetime
    
    sentiment_scores_by_period = {}

    for time_period, days in {'oneWeek': 10, 'oneMonth': 30, 'threeMonth': 90, 'sixMonth': 180, 'oneYear': 365}.items():
        start_date = end_date - timedelta(days=days)
        title_data = [item['title'] for item in res_list if start_date <= datetime.strptime(item['publishedDate'], time_format).date() <= end_date_datetime.date()]
        text_data = [item['text'] for item in res_list if start_date <= datetime.strptime(item['publishedDate'], time_format).date() <= end_date_datetime.date()]
        

        sentiment_scores_title = [compute_sentiment_score(sentence) for sentence in title_data]
        if sentiment_scores_title:  # Handle case when sentiment_scores is empty
            average_sentiment_title_score = round(sum(sentiment_scores_title) / len(sentiment_scores_title))
        else:
            average_sentiment_title_score = 0

        sentiment_scores_text = [compute_sentiment_score(sentence) for sentence in text_data]
        if sentiment_scores_text:  # Handle case when sentiment_scores is empty
            average_sentiment_text_score = round(sum(sentiment_scores_text) / len(sentiment_scores_text))
        else:
            average_sentiment_text_score = 0

        sentiment_scores_by_period[time_period] = adjust_scaled_score(round((average_sentiment_title_score+average_sentiment_text_score)/2))

    
    label_mapping = {'oneWeek': '1W', 'oneMonth': '1M', 'threeMonth': '3M', 'sixMonth': '6M', 'oneYear': '1Y'}
    result = [{'label': label_mapping[key], 'value': value} for key, value in sentiment_scores_by_period.items()]

    if any(item['value'] != 0 for item in result):

        if is_crypto == True:
            symbol = symbol.replace('-','') #convert back from BTC-USD to BTCUSD

        with open(f"json/sentiment-analysis/{symbol}.json", 'w') as file:
            ujson.dump(result, file)


async def run():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')
    crypto_con = sqlite3.connect('crypto.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stocks_symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    crypto_cursor = crypto_con.cursor()
    crypto_cursor.execute("PRAGMA journal_mode = wal")
    crypto_cursor.execute("SELECT DISTINCT symbol FROM cryptos")
    crypto_symbols = [row[0] for row in crypto_cursor.fetchall()]


    con.close()
    etf_con.close()
    crypto_con.close()

    #chunk not necessary at the moment
    
    res_list = []
    for page in tqdm(range(0,100)):
        data = await get_news_of_cryptos(crypto_symbols, page)
        if len(data) == 0:
            break
        else:
            res_list+=data

    crypto_symbols = convert_symbols(crypto_symbols)#The News article has the symbol format BTC-USD

    for symbol in crypto_symbols:
        filtered_ticker = [item for item in res_list if item['symbol'] == symbol]
        filtered_ticker = remove_duplicates(filtered_ticker, 'publishedDate')
        get_sentiment(symbol, filtered_ticker, is_crypto=True)

    
    total_symbols = stocks_symbols+etf_symbols
    
    chunk_size = len(total_symbols) // 70  # Divide the list into N chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
    for chunk in tqdm(chunks):
        res_list = []
        for page in tqdm(range(0,100)):
            data = await get_news_of_stocks(chunk, page)
            if len(data) == 0:
                break
            else:
                res_list+=data
        for symbol in chunk:
            filtered_ticker = [item for item in res_list if item['symbol'] == symbol]
            filtered_ticker = remove_duplicates(filtered_ticker, 'publishedDate')
            get_sentiment(symbol, filtered_ticker, is_crypto=False)
            
    

try:
    asyncio.run(run())
except Exception as e:
    print(e)


