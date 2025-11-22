# Standard library imports
import random
import io
import gzip
import csv
import re
import os
import secrets
import hashlib
import math
from typing import List, Dict, Set, Optional, Any
from concurrent.futures import ThreadPoolExecutor
# Third-party library imports
import numpy as np
import zipfile
import pandas as pd
import orjson
import json
import aiohttp
import aiofiles
import redis
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests
from pathlib import Path
import asyncio
import httpx

# Database related imports
import sqlite3
from contextlib import contextmanager
from pocketbase import PocketBase

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Security, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse, JSONResponse

from functools import partial
from datetime import datetime

from functions import *
from functions import FUNCTION_SOURCE_METADATA
from portfolio.calculator import calculate_portfolio_health, calculate_portfolio_fundamentals
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents.stream_events import RunItemStreamEvent
from agents import Agent, Runner, ModelSettings
from llm.agents import *
from contextlib import asynccontextmanager
from hashlib import md5
from bs4 import BeautifulSoup
from collections import Counter

from backtesting.backtest_engine import BacktestingEngine
from rule_extractor import extract_screener_rules, format_rules_for_screener
from stock_screener_engine import python_screener

# DB constants & context manager
API_URL = "http://localhost:8000"

STOCK_DB = 'stocks'
ETF_DB = 'etf'
INDEX_DB = 'index'
INSTITUTE_DB = 'institute'

OPTIONS_WATCHLIST_DIR = Path("json/options-historical-data/watchlist")

# Prioritization strategy dictionary
PRIORITY_STRATEGIES = {
    'exact_symbol_match': 0,
    'symbol_prefix_match': 1,
    'exact_name_match': 2,
    'name_prefix_match': 3,
    'symbol_contains': 4,
    'name_contains': 5
}

client = httpx.AsyncClient(http2=True, timeout=10.0)

#================LLM Configuration====================#
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
with open("json/llm/chat_instruction.txt","r",encoding="utf-8") as file:
    CHAT_INSTRUCTION = file.read()
with open("json/llm/options_insight_instruction.txt","r",encoding="utf-8") as file:
    OPTIONS_INSIGHT_INSTRUCTION = file.read()

with open("json/llm/backtesting_instruction.txt","r",encoding="utf-8") as file:
    BACKTESTING_INSTRUCTION = file.read()

with open("json/llm/portfolio_insight_instruction.txt","r",encoding="utf-8") as file:
    PORTFOLIO_INSIGHT_INSTRUCTION = file.read()


all_tools = [get_stocknear_platform_data, search_new_listed_companies, get_news_flow, get_reddit_tracker, get_fear_and_greed_index, get_ticker_earnings_call_transcripts, get_all_sector_overview, get_bitcoin_etfs, get_most_shorted_stocks, get_penny_stocks, get_ipo_calendar, get_dividend_calendar, get_ticker_trend_forecast, get_monthly_dividend_stocks, get_top_rated_dividend_stocks, get_dividend_aristocrats, get_dividend_kings, get_overbought_tickers, get_oversold_tickers, get_ticker_owner_earnings, get_ticker_financial_score, get_ticker_key_metrics, get_ticker_statistics, get_ticker_dividend, get_ticker_dark_pool, get_ticker_unusual_activity, get_ticker_open_interest_by_strike_and_expiry, get_ticker_max_pain, get_ticker_options_overview_data, get_ticker_shareholders, get_ticker_insider_trading, get_ticker_pre_post_quote, get_ticker_quote, get_market_flow, get_market_news, get_analyst_tracker, get_latest_congress_trades, get_insider_tracker, get_potus_tracker, get_top_active_stocks, get_top_aftermarket_losers, get_top_premarket_losers, get_top_losers, get_top_aftermarket_gainers, get_top_premarket_gainers, get_top_gainers, get_ticker_analyst_rating, get_ticker_news, get_latest_dark_pool_feed, get_latest_options_flow_feed, get_ticker_bull_vs_bear, get_ticker_earnings, get_ticker_earnings_price_reaction, get_top_rating_stocks, get_economic_calendar, get_earnings_releases, get_ticker_analyst_estimate, get_ticker_business_metrics, get_why_priced_moved, get_ticker_short_data, get_company_data, get_ticker_hottest_options_contracts, get_ticker_ratios_statement, get_ticker_cash_flow_statement, get_ticker_income_statement, get_ticker_balance_sheet_statement, get_congress_activity]


#======================================================#

def calculate_score(item: Dict, search_query: str) -> int:
    name_lower = item['name'].lower()
    symbol_lower = item['symbol'].lower()
    query_lower = search_query.lower()

    if len(query_lower) == 1:
        if symbol_lower == query_lower:
            base_score = PRIORITY_STRATEGIES['exact_symbol_match']
        elif name_lower == query_lower:
            base_score = PRIORITY_STRATEGIES['exact_name_match']
        else:
            base_score = len(PRIORITY_STRATEGIES)
    else:
        if symbol_lower == query_lower:
            base_score = PRIORITY_STRATEGIES['exact_symbol_match']
        elif symbol_lower.startswith(query_lower):
            base_score = PRIORITY_STRATEGIES['symbol_prefix_match']
        elif name_lower == query_lower:
            base_score = PRIORITY_STRATEGIES['exact_name_match']
        elif name_lower.startswith(query_lower):
            base_score = PRIORITY_STRATEGIES['name_prefix_match']
        elif query_lower in symbol_lower:
            base_score = PRIORITY_STRATEGIES['symbol_contains']
        elif query_lower in name_lower:
            base_score = PRIORITY_STRATEGIES['name_contains']
        else:
            base_score = len(PRIORITY_STRATEGIES)

    dot_penalty = 1 if '.' in symbol_lower else 0
    return base_score + dot_penalty


@contextmanager
def db_connection(db_name):
  conn = sqlite3.connect(f'{db_name}.db')
  cursor = conn.cursor()
  cursor.execute("PRAGMA journal_mode = wal")
  try:
    yield cursor
  finally:
    conn.commit()
    cursor.close()
    conn.close()

################# Redis #################
redis_client = redis.Redis(host='localhost', port=6380, db=0)

# Compression executor for better performance
compress_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="compress")
redis_client.flushdb() # TECH DEBT
caching_time = 3600*12 #Cache data for 12 hours

#########################################

#------Start Stocks DB------------#
with db_connection(STOCK_DB) as cursor:
  cursor.execute("SELECT DISTINCT symbol FROM stocks")
  symbols = [row[0] for row in cursor.fetchall()]

  cursor.execute("SELECT symbol, name, type, marketCap FROM stocks")
  raw_data = cursor.fetchall()
  stock_list_data = [{
    'symbol': row[0],
    'name': row[1],
    'type': row[2].capitalize(),
    'marketCap': row[3],
  } for row in raw_data if row[0] is not None and row[1] is not None and row[3] is not None]
#------End Stocks DB------------#

#------Start ETF DB------------#
with db_connection(ETF_DB) as cursor:
  cursor.execute("SELECT DISTINCT symbol FROM etfs")
  etf_symbols = [row[0] for row in cursor.fetchall()]

  cursor.execute("SELECT symbol, name, type FROM etfs")
  raw_data = cursor.fetchall()
  etf_list_data = [{
    'symbol': row[0],
    'name': row[1],
    'type': row[2].upper(),
  } for row in raw_data]
#------End ETF DB------------#


#------Start Index DB------------#

with db_connection(INDEX_DB) as cursor:
  cursor.execute("SELECT DISTINCT symbol FROM indices")
  index_symbols = [row[0] for row in cursor.fetchall()]

  cursor.execute("SELECT symbol, name, type FROM indices")
  raw_data = cursor.fetchall()
  index_list_data = [{
    'symbol': row[0],
    'name': row[1],
    'type': 'Index',
  } for row in raw_data]

#------End Index DB------------#

#------Start Institute DB------------#
with db_connection(INSTITUTE_DB) as cursor:
  cursor.execute("SELECT cik FROM institutes")
  cik_list = [row[0] for row in cursor.fetchall()]
#------End Institute DB------------#

#------Start Stock Screener--------#
with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())

# Convert stock_screener_data into a dictionary keyed by symbol
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

#------End Stock Screener--------#


#------Init Searchbar Data------------#
#index_list_data = [{'symbol': '^SPX','name': 'S&P 500 Index', 'type': 'Index'}, {'symbol': '^VIX','name': 'CBOE Volatility Index', 'type': 'Index'},]

searchbar_data = stock_list_data + etf_list_data + index_list_data

for item in searchbar_data:
    try:
        # Look up the symbol in the stock_screener_data_dict
        symbol = item['symbol']
        item['isin'] = stock_screener_data_dict[symbol]['isin']
    except Exception as e:
        item['isin'] = None

etf_set = set(etf_symbols)


### TECH DEBT ###
con = sqlite3.connect('stocks.db')
etf_con = sqlite3.connect('etf.db')

load_dotenv()

pb = PocketBase('http://127.0.0.1:8090')

FMP_API_KEY = os.getenv('FMP_API_KEY')
Benzinga_API_KEY = os.getenv('BENZINGA_API_KEY')

app = FastAPI(docs_url=None, redoc_url=None, openapi_url = None)

origins = ["http://www.stocknear.com","https://www.stocknear.com","http://stocknear.com","https://stocknear.com","http://localhost:5173","http://localhost:4173","http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



security = HTTPBasic()

#Hide /docs and /openai.json behind authentication
fastapi_username = os.getenv('FASTAPI_USERNAME')
fastapi_password = os.getenv('FASTAPI_PASSWORD')

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, fastapi_username)
    correct_password = secrets.compare_digest(credentials.password, fastapi_password)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


STOCKNEAR_API_KEY = os.getenv('STOCKNEAR_API_KEY')
api_key_header = APIKeyHeader(name="X-API-KEY")



async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != STOCKNEAR_API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


@app.get("/docs")
async def get_documentation(username: str = Depends(get_current_username)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


@app.get("/openapi.json")
async def openapi(username: str = Depends(get_current_username)):
    return get_openapi(title = "FastAPI", version="0.1.0", routes=app.routes)

class TickerData(BaseModel):
    ticker: str


class GeneralData(BaseModel):
    params: str
    etf: Optional[str] = "SPY"

class OptionContract(BaseModel):
    ticker: str
    contract: str

class ParamsData(BaseModel):
    params: str
    category: str

class GreekExposureData(BaseModel):
    params: str
    category: str
    type: str

class CompareData(BaseModel):
    tickerList: list
    category: dict

class Backtesting(BaseModel):
    strategyData: dict

class PortfolioOverview(BaseModel):
    portfolioData: list
    timePeriod: str

class PortfolioHealth(BaseModel):
    portfolioData: list

class PortfolioAnalysis(BaseModel):
    portfolioId: str
    holdings: list

class MarketNews(BaseModel):
    newsType: str

class OptionsFlowData(BaseModel):
    ticker: str = ''
    start_date: str = ''
    end_date: str = ''
    pagesize: int = Field(default=1000)
    page: int = Field(default=0)

class OptionsFlowFeed(BaseModel):
    orderList: List
    
class OptionsInsight(BaseModel):
    optionsData: Dict

class HistoricalPrice(BaseModel):
    ticker: str
    timePeriod: str

class BulkDownload(BaseModel):
    tickers: list
    bulkData: list

class CustomSettings(BaseModel):
    customSettings: list

class AnalystId(BaseModel):
    analystId: str

class PoliticianId(BaseModel):
    politicianId: str

class TranscriptData(BaseModel):
    ticker: str
    year: int
    quarter: int

class GetWatchList(BaseModel):
    watchListId: str
    ruleOfList: list

class GetList(BaseModel):
    listId: str
    ruleOfList: list

class UserId(BaseModel):
    userId: str

class GetCIKData(BaseModel):
    cik: str

class CreateStrategy(BaseModel):
    title: str
    user: str
    rules: str

class GetBatchPost(BaseModel):
    userId: str
    startPage: int
    sortingPosts: str
    seenPostId: list

class Liste(BaseModel):
    unreadList: list

class FilterStockList(BaseModel):
    filterList: str

class ETFProviderData(BaseModel):
    etfProvider: str

class IPOData(BaseModel):
    year: str

class HeatMapData(BaseModel):
    index: str

class StockScreenerData(BaseModel):
    ruleOfList: List[str]

class OptionsScreenerData(BaseModel):
    selectedDates: List[str]

class IndicatorListData(BaseModel):
    ruleOfList: list
    tickerList: list

class TransactionId(BaseModel):
    transactionId: str

class InfoText(BaseModel):
    parameter: str

class HistoricalDate(BaseModel):
    date: str

class FinancialStatement(BaseModel):
    ticker: str
    statement: str

class OptionsWatchList(BaseModel):
    optionsIdList: list

class BulkList(BaseModel):
    ticker: str
    endpoints: list

class ChatRequest(BaseModel):
    query: str
    messages: list
    reasoning: bool

# Replace NaN values with None in the resulting JSON object
def replace_nan_inf_with_none(obj):
    if isinstance(obj, list):
        return [replace_nan_inf_with_none(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: replace_nan_inf_with_none(value) for key, value in obj.items()}
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj

def load_json(file_path):
    try:
        with open(file_path, 'rb') as file:
            return orjson.loads(file.read())
    except FileNotFoundError:
        return None

async def load_json_async(file_path):
    # Check if the data is cached in Redis
    cached_data = redis_client.get(file_path)
    if cached_data:
        return orjson.loads(cached_data)
    
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
            data = orjson.loads(content)

            # Store in Redis without wrapping in create_task
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: redis_client.set(file_path, orjson.dumps(data), ex=600)
            )

            return data
    except:
        return None



@app.get("/")
async def hello_world():
    return {"stocknear api"}

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

@app.post("/historical-adj-price")
async def get_stock(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"historical-adj-price-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/historical-price/max/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except Exception as e:
        # if file reading fails, initialize to an empty list
        res = []
    
    try:
        with open(f"json/historical-price/adj/{ticker}.json", 'rb') as file:
            adj_res = orjson.loads(file.read())
    except Exception as e:
        # if file reading fails, initialize to an empty list
        adj_res = []

    # Create a dictionary mapping date (or time) to the corresponding adj price entry.
    # Assuming "date" in adj_res corresponds to "time" in res.
    adj_by_date = { entry["date"]: entry for entry in adj_res if "date" in entry }

    # Loop over the historical price records and add the adjusted prices if the date matches.
    for record in res:
        date_key = record.get("time")
        if date_key in adj_by_date:
            adj_entry = adj_by_date[date_key]
            # add adjusted data to record; adjust field names as necessary.
            #record["adjOpen"]  = adj_entry.get("adjOpen")
            #record["adjHigh"]  = adj_entry.get("adjHigh")
            #record["adjLow"]   = adj_entry.get("adjLow")
            record["adjClose"] = adj_entry.get("adjClose")

    # Serialize and cache the result.
    res_json = orjson.dumps(res)
    compressed_data = gzip.compress(res_json)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*20)  # cache for 24 hours

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/historical-price")
async def get_stock(data: HistoricalPrice, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    time_period = data.timePeriod

    cache_key = f"historical-price-{ticker}-{time_period}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/historical-price/{time_period}/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    res_json = orjson.dumps(res)
    compressed_data = gzip.compress(res_json)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*24) # Set cache expiration time to Infinity

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )
    

@app.post("/one-day-price")
async def get_stock(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()
    cache_key = f"one-day-price-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/one-day-price/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    res_json = orjson.dumps(res)
    compressed_data = gzip.compress(res_json)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*3)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



def shuffle_list(lst):
    for i in range(len(lst) - 1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
    return lst



@app.post("/similar-stocks")
async def similar_stocks(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"similar-stocks-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/similar-stocks/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*24)  # Set cache expiration time to 1 day
    return res


@app.post("/market-movers")
async def get_market_movers(data: GeneralData, api_key: str = Security(get_api_key)):
    params = data.params
    cache_key = f"market-movers-{params}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/market-movers/markethours/{params}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 5*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.post("/market-news")
async def get_market_news(data: MarketNews, api_key: str = Security(get_api_key)):
    news_type = data.newsType

    cache_key = f"market-news-{news_type}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/market-news/{news_type}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*5)  # Set cache expiration time to 15 min

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/stock-news")
async def stock_news(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"stock-news-{ticker}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})


    try:
        with open(f"json/market-news/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*5)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/news-videos")
async def stock_news(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"news-video-{ticker}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})


    try:
        with open(f"json/market-news/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    res = [item for item in res if 'youtube.com' in item['url'] or 'youtu.be' in item['url']]

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*5)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/stock-press-release")
async def stock_news(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"press-releases-{ticker}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})


    try:
        with open(f"json/market-news/press-releases/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.post("/stock-dividend")
async def stock_dividend(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"stock-dividend-{ticker}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/dividends/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {'history': []}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*15)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.post("/stock-quote")
async def stock_dividend(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"stock-quote-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/quote/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/historical-employees")
async def economic_calendar(data:TickerData, api_key: str = Security(get_api_key)):

    ticker = data.ticker.upper()

    cache_key = f"historical-employees-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/historical-employees/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60 * 15)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/financial-statement")
async def get_data(data: FinancialStatement, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    statement = data.statement.lower()
    cache_key = f"financial-statement-{ticker}-{statement}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    keys_to_remove = {"symbol", "cik", "filingDate", "acceptedDate"}
    base_path = f"json/financial-statements/{statement}"

    def load_and_clean(path: str):
        if not os.path.exists(path):
            return []
        with open(path, "rb") as f:
            raw_data = orjson.loads(f.read())
            cleaned = []
            for sublist in raw_data:
                if isinstance(sublist, list):
                    cleaned_sublist = []
                    for item in sublist:
                        if isinstance(item, dict):
                            cleaned_item = {}
                            for k, v in item.items():
                                if k in keys_to_remove:
                                    continue
                                # Remove "TTM" from the key
                                new_key = k.replace("TTM", "")
                                # Optionally, if you want to remove "TTM" from string values of a specific key,
                                # you could also do:
                                if new_key == "key" and isinstance(v, str):
                                    v = v.replace("TTM", "")
                                cleaned_item[new_key] = v
                            cleaned_sublist.append(cleaned_item)
                        else:
                            cleaned_sublist.append(item)
                    cleaned.append(cleaned_sublist)
                elif isinstance(sublist, dict):
                    cleaned_item = {}
                    for k, v in sublist.items():
                        if k in keys_to_remove:
                            continue
                        new_key = k.replace("TTM", "")
                        if new_key == "key" and isinstance(v, str):
                            v = v.replace("TTM", "")
                        cleaned_item[new_key] = v
                    cleaned.append(cleaned_item)
                else:
                    cleaned.append(sublist)
            return cleaned

    quarter_res = load_and_clean(f"{base_path}/quarter/{ticker}.json")
    annual_res = load_and_clean(f"{base_path}/annual/{ticker}.json")
    if statement == 'ratios':
        ttm_res = load_and_clean(f"{base_path}/ttm-updated/{ticker}.json")
    else:
        ttm_res = load_and_clean(f"{base_path}/ttm/{ticker}.json")

    compressed_data = gzip.compress(
        orjson.dumps({'quarter': quarter_res, 'annual': annual_res, 'ttm': ttm_res})
    )

    redis_client.setex(cache_key, 86400, compressed_data)  # 1 day expiry

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/stock-ratios")
async def stock_ratios(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"stock-ratios-{ticker}"
    cached_result = redis_client.get(cache_key)

    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/financial-statements/ratios/quarter/{ticker}.json", 'rb') as file:
            quarter_res = orjson.loads(file.read())
    except:
        quarter_res = []

    try:
        with open(f"json/financial-statements/ratios/annual/{ticker}.json", 'rb') as file:
            annual_res = orjson.loads(file.read())
    except:
        annual_res = []

    res = {'quarter': quarter_res, 'annual': annual_res}

    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    ) 



@app.get("/economic-calendar")
async def economic_calendar(api_key: str = Security(get_api_key)):

    cache_key = f"economic-calendar"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/economic-calendar/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60 * 15)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/earnings-calendar")
async def earnings_calendar(api_key: str = Security(get_api_key)):
    
    cache_key = f"earnings-calendar"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/earnings-calendar/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []
    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/dividends-calendar")
async def dividends_calendar(api_key: str = Security(get_api_key)):

    cache_key = f"dividends-calendar"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/dividends-calendar/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/stockdeck")
async def rating_stock(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"stockdeck-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/stockdeck/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*24)  # Set cache expiration time to 1 day
    return res


@app.post("/analyst-summary-rating")
async def get_analyst_rating(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"analyst-summary-rating-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/analyst/summary/all_analyst/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60*60)  # Set cache expiration time to 1 day
    return res

@app.post("/top-analyst-summary-rating")
async def get_analyst_rating(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"top-analyst-summary-rating-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/analyst/summary/top_analyst/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60*60)  # Set cache expiration time to 1 day
    return res

@app.post("/analyst-ticker-history")
async def get_analyst_ticke_history(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"analyst-ticker-history-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/analyst/history/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    # Compress the JSON data
    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*60)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/indicator-data")
async def get_indicator(data: IndicatorListData, api_key: str = Security(get_api_key)):
    rule_of_list = data.ruleOfList or ['volume', 'marketCap', 'changesPercentage', 'price', 'symbol', 'name']
    # Ensure 'symbol' and 'name' are always included in the rule_of_list
    rule_of_list_set = set(rule_of_list)
    rule_of_list_set.update(['symbol', 'name'])
    rule_of_list = list(rule_of_list_set)
    
    # Pre-filter and deduplicate ticker list
    ticker_set = {t.upper() for t in data.tickerList if t is not None}
    ticker_list = list(ticker_set)
    
    # Early return for empty ticker list
    if not ticker_list:
        return StreamingResponse(
            io.BytesIO(gzip.compress(b'[]')),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    
    # Load quote data in parallel with proper error handling
    quote_tasks = [load_json_async(f"json/quote/{ticker}.json") for ticker in ticker_list]
    quote_data = await asyncio.gather(*quote_tasks, return_exceptions=True)
    
    # Build quote_dict more efficiently, filtering out None/exceptions
    quote_dict = {
        ticker: data for ticker, data in zip(ticker_list, quote_data) 
        if data is not None and not isinstance(data, Exception)
    }
    
    # Early return if no valid quote data
    if not quote_dict:
        return StreamingResponse(
            io.BytesIO(gzip.compress(b'[]')),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    
    # Pre-compute screener data lookup and filter keys once
    screener_keys = [key for key in rule_of_list if key not in {'volume', 'marketCap', 'changesPercentage', 'price', 'symbol', 'name'}]
    screener_dict = {}
    if screener_keys:
        screener_keys_set = set(screener_keys)
        screener_dict = {
            item['symbol']: {k: v for k, v in item.items() if k in screener_keys_set} 
            for item in stock_screener_data 
            if item.get('symbol') in quote_dict  # Only process symbols we actually need
        }
    
    # Convert to set for O(1) lookup
    etf_set_lookup = etf_set
    rule_of_list_set = set(rule_of_list)
    
    # Build results more efficiently
    combined_results = []
    for ticker, quote in quote_dict.items():
        # Determine ticker type with single lookup
        ticker_type = 'etf' if ticker in etf_set_lookup else 'stock'
        
        # Build result dict in single pass
        result = {'type': ticker_type}
        
        # Add quote data for keys in rule_of_list
        for key in rule_of_list_set:
            if key in quote:
                result[key] = quote[key]
            else:
                result[key] = None
        
        # Merge screener data if available
        if ticker in screener_dict:
            result.update(screener_dict[ticker])
        
        combined_results.append(result)
    
    # Serialize and compress the response
    res = orjson.dumps(combined_results)
    compressed_data = gzip.compress(res)
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

async def process_watchlist_ticker(ticker, rule_of_list, quote_keys_to_include, screener_dict, etf_set, all_news_data):
    """Optimized single ticker processing with guaranteed rule_of_list keys."""
    ticker = ticker.upper()
    ticker_type = 'stocks'
    if ticker in etf_set:
        ticker_type = 'etf'

    try:
        # Combine I/O operations into single async read
        quote_dict, earnings_dict = await asyncio.gather(
            load_json_async(f"json/quote/{ticker}.json"),
            load_json_async(f"json/earnings/next/{ticker}.json")
        )
    except Exception:
        return None, [], None

    if not quote_dict:
        return None, [], None

    # Merge data from multiple sources
    filtered_quote = {
        key: quote_dict.get(key)
        for key in rule_of_list + quote_keys_to_include
        if key in quote_dict
    }

    # Add core fields with priority to quote data
    core_keys = ['price', 'volume', 'changesPercentage', 'symbol']
    filtered_quote.update({k: quote_dict[k] for k in core_keys if k in quote_dict})

    # Merge screener data for non-core fields
    symbol = filtered_quote.get('symbol')
    if symbol and symbol in screener_dict:
        filtered_quote.update({
            k: v for k, v in screener_dict[symbol].items()
            if k not in filtered_quote
        })

    # Ensure all rule_of_list keys exist with None as default
    for key in rule_of_list:
        filtered_quote.setdefault(key, None)

    filtered_quote['type'] = ticker_type

    # Process news from centralized data - filter by ticker and transform format
    news = []
    if all_news_data:
        for item in all_news_data:
            if ticker in item.get('symbolList', []):
                # Transform to match frontend expected format
                news_item = {
                    'symbol': ticker,
                    'publishedDate': item.get('date'),
                    'title': item.get('text'),
                    'type': ticker_type
                }
                news.append(news_item)

        # Limit to 5 most recent news items
        news = news[:5]

    earnings = {**earnings_dict, 'symbol': symbol} if earnings_dict else None

    return filtered_quote, news, earnings

@app.post("/get-watchlist")
async def get_watchlist(data: GetWatchList, api_key: str = Security(get_api_key)):
    """Optimized endpoint with complete key enforcement."""
    try:
        watchlist = pb.collection("watchlist").get_one(data.watchListId)
        ticker_list = watchlist.ticker
    except Exception:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    # Configure data collection parameters
    rule_of_list = list(set(
        (data.ruleOfList or []) +
        ['symbol', 'name']  # Ensure mandatory fields
    ))
    quote_keys_to_include = ['volume', 'marketCap', 'changesPercentage', 'price']

    # Load centralized news data once
    all_news_data = await load_json_async("json/wiim/flow/data.json") or []

    # Preprocess screener data for O(1) lookups
    screener_dict = {
        item['symbol']: {k: item.get(k) for k in rule_of_list if k in item}
        for item in stock_screener_data
    }

    # Parallel processing pipeline
    results = await asyncio.gather(*[
        process_watchlist_ticker(
            ticker,
            rule_of_list,
            quote_keys_to_include,
            screener_dict,
            etf_set,
            all_news_data
        ) for ticker in ticker_list
    ])

    # Efficient response assembly
    response_data = {
        'data': [r[0] for r in results if r[0]],
        'news': [item for r in results for item in r[1]],
        'earnings': [r[2] for r in results if r[2]]
    }

    return StreamingResponse(
        io.BytesIO(gzip.compress(orjson.dumps(response_data))),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/get-portfolio")
async def get_portfolio(data: GetList, api_key: str = Security(get_api_key)):

    try:
        portfolio = pb.collection("portfolio").get_one(data.listId)
        ticker_list = [item['symbol'] for item in portfolio.ticker]
    except Exception:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Configure data collection parameters
    rule_of_list = list(set(
        (data.ruleOfList or []) +
        ['symbol', 'name']  # Ensure mandatory fields
    ))
    quote_keys_to_include = ['changesPercentage', 'price']

    # Load centralized news data once
    all_news_data = await load_json_async("json/wiim/flow/data.json") or []

    # Preprocess screener data for O(1) lookups
    screener_dict = {
        item['symbol']: {k: item.get(k) for k in rule_of_list if k in item}
        for item in stock_screener_data
    }

    # Parallel processing pipeline
    results = await asyncio.gather(*[
        process_watchlist_ticker(
            ticker,
            rule_of_list,
            quote_keys_to_include,
            screener_dict,
            etf_set,
            all_news_data
        ) for ticker in ticker_list
    ])

    # Efficient response assembly
    response_data = {
        'data': [r[0] for r in results if r[0]],
        'news': [item for r in results for item in r[1]],
        'earnings': [r[2] for r in results if r[2]]
    }

    return StreamingResponse(
        io.BytesIO(gzip.compress(orjson.dumps(response_data))),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

def _to_float(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None

def _to_int(x):
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None

async def _return_empty_portfolio(cache_key: str, cache_ttl: int = 60):
    """Helper function to return empty portfolio response"""
    payload = {
        "portfolio": [],
        "benchmark": [],
        "dividends": {"annualDividends": 0, "dividendYield": 0}
    }
    compressed = await asyncio.get_event_loop().run_in_executor(
        compress_executor,
        lambda: gzip.compress(orjson.dumps(payload))
    )
    redis_client.setex(cache_key, cache_ttl, compressed)
    return StreamingResponse(
        io.BytesIO(compressed),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/portfolio-overview")
async def get_data(data: PortfolioOverview, api_key: str = Security(get_api_key)):
    portfolio_data = data.portfolioData or []
    time_period = data.timePeriod

    # Validate position count to prevent excessive processing
    if len(portfolio_data) > 100:
        return JSONResponse(
            {"error": "Too many positions. Maximum 100 allowed."},
            status_code=400
        )

    # Create efficient cache key using hash
    portfolio_tuple = tuple(sorted([
        (item.get("symbol"), item.get("shares"), item.get("avgPrice"))
        for item in portfolio_data if item
    ]))
    portfolio_hash = hashlib.md5(orjson.dumps(portfolio_tuple)).hexdigest()
    cache_key = f"portfolio-overview-{portfolio_hash}-{time_period}"

    # Cache
    cached = redis_client.get(cache_key)
    if cached:
        return StreamingResponse(
            io.BytesIO(cached),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    # Keep only positions with BOTH shares and avgPrice
    positions = []
    for item in portfolio_data:
        symbol = (item or {}).get("symbol")
        shares = _to_int((item or {}).get("shares"))
        avg_price = _to_float((item or {}).get("avgPrice"))
        if symbol and shares is not None and avg_price is not None:
            positions.append({"symbol": symbol, "shares": shares, "avgPrice": avg_price})

    if not positions:
        return await _return_empty_portfolio(cache_key, cache_ttl=300)

    # Always load one-year price histories (portfolio + SPY)
    tasks = [load_json_async(f"json/historical-price/one-year/{p['symbol']}.json") for p in positions]
    tasks.append(load_json_async("json/historical-price/one-year/SPY.json"))  # Add SPY
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Build date->close maps per symbol & intersect dates across all usable symbols
    symbol_closes: Dict[str, Dict[str, float]] = {}
    all_dates_intersection: Optional[set] = None
    usable_positions = []
    spy_closes: Dict[str, float] = {}

    # Process portfolio positions
    for pos, bars_or_exc in zip(positions, results[:-1]):  # Exclude last result (SPY)
        if isinstance(bars_or_exc, Exception) or not bars_or_exc:
            continue  # skip symbols that failed to load
        try:
            dmap: Dict[str, float] = {}
            for b in bars_or_exc:
                d = str(b.get("time"))  # "YYYY-MM-DD"
                c = _to_float(b.get("close"))
                if d and c is not None:
                    dmap[d] = c
            if not dmap:
                continue
            symbol_closes[pos["symbol"]] = dmap
            usable_positions.append(pos)

            if all_dates_intersection is None:
                all_dates_intersection = set(dmap.keys())
            else:
                all_dates_intersection &= set(dmap.keys())
        except Exception:
            continue

    # Process SPY data
    spy_bars = results[-1]
    if not isinstance(spy_bars, Exception) and spy_bars:
        try:
            for b in spy_bars:
                d = str(b.get("time"))
                c = _to_float(b.get("close"))
                if d and c is not None:
                    spy_closes[d] = c
            # Intersect SPY dates with portfolio dates
            if all_dates_intersection and spy_closes:
                all_dates_intersection &= set(spy_closes.keys())
        except Exception:
            pass

    if not usable_positions or not all_dates_intersection:
        payload = {
            "portfolio": [],
            "benchmark": [],
            "dividends": {
                "annualDividends": 0,
                "dividendYield": 0
            }
        }
        compressed = await asyncio.to_thread(lambda: gzip.compress(orjson.dumps(payload)))
        redis_client.setex(cache_key, 60, compressed)
        return StreamingResponse(io.BytesIO(compressed), media_type="application/json",
                                 headers={"Content-Encoding": "gzip"})

    # Constant total invested across the series
    total_invested = sum(p["avgPrice"] * p["shares"] for p in usable_positions)
    if total_invested == 0:
        payload = {
            "portfolio": [],
            "benchmark": [],
            "dividends": {
                "annualDividends": 0,
                "dividendYield": 0
            }
        }
        compressed = await asyncio.to_thread(lambda: gzip.compress(orjson.dumps(payload)))
        redis_client.setex(cache_key, 60, compressed)
        return StreamingResponse(io.BytesIO(compressed), media_type="application/json",
                                 headers={"Content-Encoding": "gzip"})

    # Compute portfolio value per date (intersection of all symbols' dates)
    dates_sorted = sorted(all_dates_intersection)  # "YYYY-MM-DD"

    # Filter dates based on time period
    from datetime import datetime, timedelta
    if dates_sorted:
        end_date = datetime.strptime(dates_sorted[-1], "%Y-%m-%d")

        # Calculate start date based on time period
        if time_period == "1W":
            start_date = end_date - timedelta(weeks=1)
        elif time_period == "1M":
            start_date = end_date - timedelta(days=30)
        elif time_period == "3M":
            start_date = end_date - timedelta(days=90)
        elif time_period == "6M":
            start_date = end_date - timedelta(days=180)
        else:  # "1Y" or default
            start_date = end_date - timedelta(days=365)

        # Filter dates to only include those within the time period
        dates_sorted = [d for d in dates_sorted if datetime.strptime(d, "%Y-%m-%d") >= start_date]

    # Calculate portfolio market values (raw prices)
    portfolio_series: List[Dict[str, Any]] = []
    for d in dates_sorted:
        total_mv = 0.0
        for p in usable_positions:
            c = symbol_closes[p["symbol"]][d]  # safe due to intersection
            total_mv += c * p["shares"]
        portfolio_series.append({
            "date": d,
            "value": round(total_mv, 2)  # Send raw portfolio value
        })

    # Send SPY benchmark prices (raw close prices)
    benchmark_series: List[Dict[str, Any]] = []
    if spy_closes and dates_sorted:
        for d in dates_sorted:
            spy_close = spy_closes.get(d)
            if spy_close:
                benchmark_series.append({
                    "date": d,
                    "close": round(spy_close, 2)  # Send raw SPY close price
                })

    # Calculate annual dividends
    total_annual_dividends = 0.0
    dividend_yield_percentage = 0.0

    for pos in usable_positions:
        symbol = pos["symbol"]
        shares = pos["shares"]

        # Get dividend info from stock screener data
        stock_data = stock_screener_data_dict.get(symbol, {})
        dividend_yield = _to_float(stock_data.get('dividendYield'))
        annual_dividend_per_share = _to_float(stock_data.get('annualDividend'))
        current_price = _to_float(stock_data.get('price'))

        if annual_dividend_per_share and annual_dividend_per_share > 0:
            total_annual_dividends += shares * annual_dividend_per_share
        elif dividend_yield and current_price and dividend_yield > 0 and current_price > 0:
            # Fallback: derive annual dividend per share from yield percentage
            annual_dividend = shares * current_price * (dividend_yield / 100)
            total_annual_dividends += annual_dividend

    # Calculate portfolio dividend yield
    if total_invested > 0:
        dividend_yield_percentage = (total_annual_dividends / total_invested) * 100

    # Return both portfolio and benchmark data with raw prices and dividends
    result = {
        "portfolio": portfolio_series,
        "benchmark": benchmark_series,
        "dividends": {
            "annualDividends": round(total_annual_dividends, 2),
            "dividendYield": round(dividend_yield_percentage, 2)
        }
    }

    compressed = await asyncio.to_thread(lambda: gzip.compress(orjson.dumps(result)))
    redis_client.setex(cache_key, 60, compressed)
    return StreamingResponse(io.BytesIO(compressed), media_type="application/json",
                             headers={"Content-Encoding": "gzip"})




@app.post("/portfolio-health-score")
async def get_data(data: PortfolioHealth, api_key: str = Security(get_api_key)):
    portfolio_data = data.portfolioData or []

    # Create cache key based on holdings
    holdings_str = ','.join([f"{item.get('symbol','')}-{item.get('shares','')}-{item.get('avgPrice','')}"
                             for item in portfolio_data if item])
    cache_key = f"portfolio-health-{holdings_str}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return StreamingResponse(
            io.BytesIO(cached),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        # Calculate health scores (now async with batch file loading)
        health_scores = await calculate_portfolio_health(portfolio_data)

        # Format result
        result = {
            "categories": ["Moat", "Trend", "Growth", "Fundamentals", "Volatility"],
            "values": [
                health_scores['Moat'],
                health_scores['Trend'],
                health_scores['Growth'],
                health_scores['Fundamentals'],
                health_scores['Volatility']
            ]
        }

        # Compress and cache
        compressed = await asyncio.to_thread(lambda: gzip.compress(orjson.dumps(result)))
        redis_client.setex(cache_key, 3600, compressed)  # Cache for 1 hour

        return StreamingResponse(
            io.BytesIO(compressed),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    except Exception as e:
        # Return default scores on error
        print(f"Error calculating portfolio health: {e}")
        default_result = {
            "categories": ["Moat", "Trend", "Growth", "Fundamentals", "Volatility"],
            "values": [50.0, 50.0, 50.0, 50.0, 50.0]
        }
        compressed = await asyncio.to_thread(lambda: gzip.compress(orjson.dumps(default_result)))
        return StreamingResponse(
            io.BytesIO(compressed),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )


@app.post("/portfolio-analysis")
async def portfolio_analysis(data: PortfolioAnalysis, api_key: str = Security(get_api_key)):
    """
    Calculate portfolio-weighted fundamental metrics for analysis page
    Returns valuation, growth, efficiency, and margins data
    """
    holdings = data.holdings or []
    
    # Create cache key
    holdings_str = ','.join([f"{item.get('symbol','')}-{item.get('shares','')}-{item.get('avgPrice','')}"
                             for item in holdings if item])
    cache_key = f"portfolio-analysis-{holdings_str}"
    
    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return StreamingResponse(
            io.BytesIO(cached),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    
    # Calculate portfolio fundamentals using refactored function
    result = await asyncio.to_thread(
        calculate_portfolio_fundamentals,
        holdings,
        stock_screener_data_dict
    )


    # Compress and cache
    compressed = gzip.compress(orjson.dumps(result))
    redis_client.setex(cache_key, 3600, compressed)  # Cache for 1 hour

    return StreamingResponse(
        io.BytesIO(compressed),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/portfolio-bull-bear")
async def portfolio_bull_bear(data: PortfolioAnalysis, api_key: str = Security(get_api_key)):
    holdings = data.holdings or []

    # Helper function to safely convert to float
    def safe_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # Filter out watchlist items and enrich with current market data
    active_holdings = []
    enriched_holdings = []
    total_market_value = 0

    for h in holdings:
        try:
            avg_price = safe_float(h.get('avgPrice'))
            shares = safe_float(h.get('shares'))
            symbol = h.get('symbol', '').upper()

            if avg_price is not None and shares is not None and avg_price > 0 and shares > 0:
                active_holdings.append(h)

                # Get current price from stock screener data
                stock_data = stock_screener_data_dict.get(symbol, {})
                current_price = safe_float(stock_data.get('price'))

                if current_price and current_price > 0:
                    market_value = shares * current_price
                    cost_basis = shares * avg_price
                    profit_loss = market_value - cost_basis
                    profit_loss_percent = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0

                    enriched_holdings.append({
                        'symbol': symbol,
                        'shares': shares,
                        'avgPrice': avg_price,
                        'currentPrice': current_price,
                        'marketValue': market_value,
                        'costBasis': cost_basis,
                        'profitLoss': profit_loss,
                        'profitLossPercent': profit_loss_percent,
                    })
                    total_market_value += market_value
                else:
                    # If no current price, include basic info only
                    enriched_holdings.append({
                        'symbol': symbol,
                        'shares': shares,
                        'avgPrice': avg_price,
                    })
        except:
            pass

    # If no active holdings, return empty analysis as stream
    if not active_holdings:
        async def empty_stream():
            yield orjson.dumps({"event": "complete", "bullSay": "", "bearSay": ""}) + b"\n"

        return StreamingResponse(
            empty_stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"}
        )

    # Calculate portfolio weights
    for holding in enriched_holdings:
        if 'marketValue' in holding and total_market_value > 0:
            holding['weight'] = (holding['marketValue'] / total_market_value * 100)

    # Format enriched portfolio data for AI analysis
    portfolio_json = orjson.dumps(enriched_holdings, option=orjson.OPT_INDENT_2).decode('utf-8')
    # Prepare messages for agent with enriched data context
    data_description = """
        Each position includes:
        - symbol: Stock ticker
        - shares: Number of shares owned
        - avgPrice: Average purchase price
        - currentPrice: Current market price
        - marketValue: Current value (shares × currentPrice)
        - costBasis: Original investment (shares × avgPrice)
        - profitLoss: Unrealized P/L in dollars
        - profitLossPercent: Return percentage from entry
        - weight: Position size as % of total portfolio
"""

    messages = [
        {
            "role": "system",
            "content": PORTFOLIO_INSIGHT_INSTRUCTION
        },
        {
            "role": "user",
            "content": f"Analyze this portfolio and provide bull and bear arguments.\n\n{data_description}\nPortfolio:\n{portfolio_json}"
        }
    ]

    # Add today's date as context
    today_date = datetime.now().strftime("%B %d, %Y")
    date_context = {
        "role": "system",
        "content": f"Today's date is {today_date}. Use this for any date-related analysis or market context."
    }
    messages = [date_context] + messages

    # Agent setup for portfolio analysis
    model_settings = ModelSettings(
        tool_choice="auto",
        parallel_tool_calls=True,
        reasoning={"effort": "medium"},
        text={"verbosity": "low"}
    )

    agent = Agent(
        name="Portfolio Analyst",
        instructions=PORTFOLIO_INSIGHT_INSTRUCTION,
        model=os.getenv("CHAT_MODEL"),
        tools=[
            get_ticker_bull_vs_bear,
            get_ticker_quote,
            get_company_data,
        ],
        model_settings=model_settings
    )

    async def event_generator():
        full_content = ""

        try:
            result = Runner.run_streamed(agent, input=messages)
            async for event in result.stream_events():
                try:
                    # Process only raw_response_event events
                    if event.type == "raw_response_event":
                        delta = getattr(event.data, "delta", "")
                        if not delta:
                            continue

                        full_content += delta
                        yield orjson.dumps({"event": "progress", "content": full_content}) + b"\n"

                except Exception as e:
                    print(f"Event processing error: {e}")
                    continue

            # Parse and validate the complete response
            if full_content:
                try:
                    # Extract JSON from response (it might be wrapped in markdown code blocks)
                    json_content = full_content.strip()
                    if json_content.startswith("```json"):
                        json_content = json_content.split("```json")[1].split("```")[0].strip()
                    elif json_content.startswith("```"):
                        json_content = json_content.split("```")[1].split("```")[0].strip()

                    parsed_result = orjson.loads(json_content)

                    # Validate the response has required fields
                    if "bullSay" not in parsed_result or "bearSay" not in parsed_result:
                        raise ValueError("Invalid response format from AI")


                    # Send complete event with parsed result
                    yield orjson.dumps({"event": "complete", **parsed_result}) + b"\n"

                except Exception as e:
                    print(f"Error parsing portfolio analysis result: {e}")
                    yield orjson.dumps({
                        "event": "error",
                        "message": f"Failed to parse analysis: {str(e)}"
                    }) + b"\n"

        except Exception as e:
            print(f"Streaming error: {e}")
            yield orjson.dumps({
                "event": "error",
                "message": f"Failed to generate analysis"
            }) + b"\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"}
    )


@app.post("/get-price-alert")
async def get_price_alert(data: dict, api_key: str = Security(get_api_key)):
    user_id = data.get('userId')
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    # Fetch all alerts for the user in a single database call
    try:
        result = pb.collection("priceAlert").get_full_list(
            query_params={"filter": f"user='{user_id}' && triggered=false"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

    # Extract unique tickers
    unique_tickers = {item.symbol for item in result if hasattr(item, 'symbol')}

    # Load centralized news data once
    all_news_data = await load_json_async("json/wiim/flow/data.json") or []

    async def fetch_ticker_data(ticker, all_news_data):
        try:
            earnings_task = load_json_async(f"json/earnings/next/{ticker}.json")
            earnings_dict = await earnings_task

            # Process news from centralized data - filter by ticker
            news = []
            if all_news_data:
                for item in all_news_data:
                    if ticker in item.get('symbolList', []):
                        # Transform to match frontend expected format
                        news_item = {
                            'symbol': ticker,
                            'publishedDate': item.get('date'),
                            'title': item.get('text'),
                            'type': item.get('assetType', 'stocks')
                        }
                        news.append(news_item)

                # Limit to 5 most recent news items
                news = news[:5]
                print(f"DEBUG: Found {len(news)} news items for {ticker}")

            # Process earnings
            earnings = None
            if earnings_dict:
                earnings = {**earnings_dict, 'symbol': ticker}

            return news, earnings
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return [], None
    
    async def fetch_quote_data(item):
        try:
            async with aiofiles.open(f"json/quote/{item.symbol}.json", mode='r') as file:
                quote_data = orjson.loads(await file.read())
            
            return {
                'symbol': item.symbol,
                'name': getattr(item, 'name', ''),
                'id': item.id,
                'type': getattr(item, 'asset_type', '').lower().replace("stock", "stocks"),
                'targetPrice': getattr(item, 'target_price', None),
                'condition': getattr(item, 'condition', '').capitalize(),
                'priceWhenCreated': getattr(item, 'price_when_created', None),
                'price': quote_data.get("price"),
                'changesPercentage': quote_data.get("changesPercentage"),
                'volume': quote_data.get("volume"),
            }
        except FileNotFoundError:
            print(f"Quote file not found for {item.symbol}")
            return None
        except Exception as e:
            print(f"Error processing {item.symbol}: {e}")
            return None
    
    try:
        # Run all tasks concurrently
        ticker_tasks = [fetch_ticker_data(ticker, all_news_data) for ticker in unique_tickers]
        quote_tasks = [fetch_quote_data(item) for item in result]

        ticker_results = await asyncio.gather(*ticker_tasks)
        quote_results = await asyncio.gather(*quote_tasks)
        
        # Process results
        combined_results = [res for res in quote_results if res]
        combined_news = [news_item for news, _ in ticker_results for news_item in news]
        combined_earnings = [earnings for _, earnings in ticker_results if earnings]
        
        # Final response structure
        res = {
            'data': combined_results,
            'news': combined_news,
            'earnings': combined_earnings,
        }
        # Serialize and compress the response data
        res_serialized = orjson.dumps(res)
        compressed_data = gzip.compress(res_serialized)

        return StreamingResponse(
            io.BytesIO(compressed_data),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



def process_option_activity(item):
    item['put_call'] = 'Calls' if item['put_call'] == 'CALL' else 'Puts'
    item['underlying_type'] = item['underlying_type'].lower()
    item['price'] = round(float(item['price']), 2)
    item['strike_price'] = round(float(item['strike_price']), 2)
    item['cost_basis'] = round(float(item['cost_basis']), 2)
    item['underlying_price'] = round(float(item['underlying_price']), 2)
    item['option_activity_type'] = item['option_activity_type'].capitalize()
    item['sentiment'] = item['sentiment'].capitalize()
    item['execution_estimate'] = item['execution_estimate'].replace('_', ' ').title()
    item['tradeCount'] = item.get('trade_count', 0)
    return item

async def fetch_option_data(option_id: str):
    url = "https://api.benzinga.com/api/v1/signal/option_activity"
    headers = {"accept": "application/json"}
    querystring = {"token": Benzinga_API_KEY, "parameters[id]": option_id}
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = orjson.loads(response.text)
        option_activity = data.get('option_activity', [])
        
        if isinstance(option_activity, list):
            return [process_option_activity(item) for item in option_activity]
        else:
            print(f"Unexpected response format for {option_id}: {option_activity}")
            return []
    except Exception as e:
        print(f"Error fetching data for {option_id}: {e}")
        return []

@app.post("/get-options-watchlist")
async def get_options_watchlist(data: OptionsWatchList, api_key: str = Security(get_api_key)):
    options_list_id = sorted(data.optionsIdList)
    cache_key = f"options-watchlist-{options_list_id}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    result = []

    for option_id in options_list_id:
        file_path = OPTIONS_WATCHLIST_DIR / f"{option_id}.json"
        
        if file_path.exists():
            with open(file_path, 'rb') as json_file:
                option_data = orjson.loads(json_file.read())
                result.extend(option_data)
        else:
            option_activity = await fetch_option_data(option_id)
            if option_activity:
                with open(file_path, 'wb') as file:
                    file.write(orjson.dumps(option_activity))
                result.extend(option_activity)

    compressed_data = gzip.compress(orjson.dumps(result))
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60 * 30)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/price-prediction")
async def brownian_motion(data:TickerData, api_key: str = Security(get_api_key)):

    data= data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"price-prediction-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    
    if ticker in etf_symbols:
        table_name = 'etfs'
    else:
        table_name = 'stocks'
    
    query = f"""
    SELECT 
        pricePrediction
    FROM 
        {table_name}
    WHERE
        symbol = ?
    """

    df = pd.read_sql_query(query, etf_con if table_name == 'etfs' else con, params=(ticker,))
    df_dict = df.to_dict()
    try:
        price_dict = orjson.loads(df_dict['pricePrediction'][0])
    except:
        price_dict = {'1W': {'min': 0, 'mean': 0, 'max': 0}, '1M': {'min': 0, 'mean': 0, 'max': 0}, '3M': {'min': 0, 'mean': 0, 'max': 0}, '6M': {'min': 0, 'mean': 0, 'max': 0}}

    redis_client.set(cache_key, orjson.dumps(price_dict))
    redis_client.expire(cache_key, 3600*24) # Set cache expiration time to 1 hour
    return price_dict



@app.post("/stock-screener-data")
async def stock_finder(data: StockScreenerData, api_key: str = Security(get_api_key)):
    # Use frozenset for consistent, hashable cache key
    rule_set = frozenset(data.ruleOfList)
    cache_key = f"stock-screener-data-{hash(rule_set)}"
    
    # Check cache first
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    
    # Pre-compute filtering sets
    always_include = {'symbol', 'marketCap', 'price', 'changesPercentage', 'name', 'volume', 'priceToEarningsRatio'}
    all_keys = always_include | set(data.ruleOfList)
    
    try:
        # Single-pass filter with list comprehension optimization
        # Pre-filter for US stocks and extract required fields in one operation
        filtered_data = [
            {key: item[key] for key in all_keys if key in item}
            for item in stock_screener_data 
            if item.get('exchange') != 'OTC'
        ]
    except Exception:
        filtered_data = []
    
    # Serialize and compress
    res = orjson.dumps(filtered_data)
    compressed_data = gzip.compress(res)
    
    # Set cache with expiration in single call (more efficient)
    redis_client.setex(cache_key, 86400, compressed_data)  # 24 hours in seconds
    
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.post("/options-screener-data")
async def get_data(data: OptionsScreenerData, api_key: str = Security(get_api_key)):
    # Sort selected dates
    selected_dates = sorted(data.selectedDates)
    
    # Prepare cache key
    cache_key = f"options-screener-data-{selected_dates}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    # Index filtering is now handled client-side

    #need to load the data everytime since moneyness is computed every 5 min

    with open(f"json/screener/options-screener.json", 'rb') as file:
        options_screener_data = orjson.loads(file.read())

    # Compute counts per expiration
    expirations = [item.get('expiration') for item in options_screener_data if 'expiration' in item]
    count_by_date = Counter(expirations)
    # Build unique_expirations list with date and contractLength
    unique_expirations = [
        {'date': date, 'contractLength': count_by_date[date]}
        for date in sorted(count_by_date)
    ]

    try:
        # Filter by selected dates if provided, else use earliest
        if selected_dates:
            filtered_data = [item for item in options_screener_data if item.get('expiration') in selected_dates]
        else:
            earliest = unique_expirations[0]['date'] if unique_expirations else None
            filtered_data = [
                item for item in options_screener_data 
                if item.get('expiration') == earliest
            ] if earliest else []
        
        # Index filtering is handled client-side for better performance

        # Sort filtered data by totalPrem descending
        filtered_data.sort(key=lambda x: x.get("totalPrem", 0), reverse=True)

    except Exception as e:
        print(f"Error filtering data: {e}")
        filtered_data = []

    # Serialize response including unique_expirations
    payload = {
        'expirationList': unique_expirations,
        'data': filtered_data
    }
    serialized = orjson.dumps(payload)
    compressed_data = gzip.compress(serialized)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/congress-trading-ticker")
async def get_fair_price(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"get-congress-trading-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/congress-trading/company/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.post("/shareholders")
async def get_shareholders(data: TickerData, api_key: str = Security(get_api_key)):

    data = data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"shareholders-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/shareholders/{ticker}.json", 'rb') as file:
            shareholder_list = orjson.loads(file.read())
    except:
        shareholder_list = []

    try:
        with open(f"json/ownership-stats/{ticker}.json", 'rb') as file:
            stats = orjson.loads(file.read())
    except:
        stats = {}

    try:
        res = {**stats, 'shareholders': shareholder_list}
    except:
        res = {}

    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/cik-data")
async def get_hedge_funds_data(data: GetCIKData, api_key: str = Security(get_api_key)):
    data = data.dict()
    cik = data['cik']

    cache_key = f"{cik}-hedge-funds"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    
    try:
        with open(f"json/hedge-funds/companies/{cik}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 3600) # Set cache expiration time to Infinity

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


def _load_json_snapshot(cache_key: str, file_path: str, *, ttl: Optional[int] = None, compress: bool = True) -> list[dict]:
    cached_result = redis_client.get(cache_key)
    if cached_result:
        try:
            raw_bytes = gzip.decompress(cached_result) if compress else cached_result
            return orjson.loads(raw_bytes)
        except Exception:
            pass

    try:
        with open(file_path, "rb") as file:
            dataset = orjson.loads(file.read())
    except Exception:
        dataset = []

    serialized = orjson.dumps(dataset)
    redis_client.set(cache_key, gzip.compress(serialized) if compress else serialized)
    if ttl:
        redis_client.expire(cache_key, ttl)
    return dataset


def _paginate_collection(
    dataset: List[dict],
    *,
    page: int,
    page_size: int,
    sort_key: str,
    sort_order: str,
    default_sort_key: str,
    valid_sort_keys: Set[str],
    field_types: Optional[Dict[str, str]] = None,
    search_term: Optional[str] = None,
    search_fields: Optional[List[str]] = None,
) -> dict:
    dataset = dataset or []
    page_size = max(1, min(page_size, 1000))

    normalized_order = (sort_order or "asc").lower()
    if normalized_order not in {"asc", "desc"}:
        normalized_order = "asc"
    reverse = normalized_order == "desc"

    effective_sort_key = sort_key if sort_key in valid_sort_keys else default_sort_key
    field_types = field_types or {}

    normalized_search = (search_term or "").strip().lower()
    filtered = dataset
    if normalized_search and search_fields:
        lowered_fields = [field for field in search_fields if field]
        if lowered_fields:
            filtered = [
                item
                for item in dataset
                if any(
                    normalized_search in str(item.get(field, "")).lower()
                    for field in lowered_fields
                )
            ]

    total_items = len(filtered)
    if total_items == 0:
        total_pages = 1
        page = 1
    else:
        total_pages = max(1, math.ceil(total_items / page_size))
        page = max(1, min(page, total_pages))

    def sort_accessor(item: dict):
        value = item.get(effective_sort_key)
        value_type = field_types.get(effective_sort_key, "number")
        if value_type == "string":
            if value is None:
                return "" if reverse else chr(0x10FFFF)
            return str(value).lower()
        else:
            if value is None:
                return float('-inf') if reverse else float('inf')
            try:
                return float(value)
            except (TypeError, ValueError):
                return float('-inf') if reverse else float('inf')

    try:
        sorted_data = sorted(filtered, key=sort_accessor, reverse=reverse)
    except Exception:
        sorted_data = filtered

    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    items = sorted_data[start_index:end_index]

    return {
        "items": items,
        "total": total_items,
        "page": page,
        "pageSize": page_size,
        "totalPages": total_pages,
        "sort": {"key": effective_sort_key, "order": "desc" if reverse else "asc"},
        "search": normalized_search,
    }


def _load_all_hedge_funds_snapshot() -> list[dict]:
    return _load_json_snapshot(
        "all-hedge-funds",
        "json/hedge-funds/all-hedge-funds.json",
        ttl=3600 * 3600,
    )


def _load_market_movers_snapshot(
    category: str,
    timeframe: str = "1D",
    session: str = "markethours",
) -> list[dict]:
    cache_key = f"market-movers:{session}:{category}"
    dataset = _load_json_snapshot(
        cache_key,
        f"json/market-movers/{session}/{category}.json",
        ttl=300,
    )
    if isinstance(dataset, dict):
        return dataset.get(timeframe, [])
    if timeframe == "1D":
        return dataset
    return []


def _cache_paginated_result(cache_key: str, payload: dict, ttl: int = 3600):
    redis_client.set(cache_key, orjson.dumps(payload))
    redis_client.expire(cache_key, ttl)


@app.get("/hedge-funds")
async def get_paginated_hedge_funds(
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=500),
    search: Optional[str] = Query(None),
    sortKey: str = Query("rank"),
    sortOrder: str = Query("asc"),
    api_key: str = Security(get_api_key),
):
    raw_search = (search or "").strip()
    normalized_search = raw_search.lower()
    hashed_search = md5(normalized_search.encode("utf-8")).hexdigest() if normalized_search else "none"
    cache_key = f"hedge-funds:{page}:{pageSize}:{sortKey}:{sortOrder}:{hashed_search}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        try:
            return JSONResponse(content=orjson.loads(cached_result))
        except Exception:
            pass

    dataset = _load_all_hedge_funds_snapshot()

    payload = _paginate_collection(
        dataset,
        page=page,
        page_size=pageSize,
        sort_key=sortKey,
        sort_order=sortOrder,
        default_sort_key="rank",
        valid_sort_keys={
            "rank",
            "name",
            "marketValue",
            "numberOfStocks",
            "turnover",
            "performancePercentage3Year",
            "winRate",
        },
        field_types={"name": "string"},
        search_term=raw_search,
        search_fields=["name", "cik"],
    )

    _cache_paginated_result(cache_key, payload, ttl=3600)

    return JSONResponse(content=payload)


@app.get("/market-movers/gainers")
async def get_paginated_market_movers_gainers(
    page: int = Query(1, ge=1),
    pageSize: int = Query(10, ge=1, le=500),
    search: Optional[str] = Query(None),
    sortKey: str = Query("changesPercentage"),
    sortOrder: str = Query("desc"),
    timeframe: str = Query("1D"),
    api_key: str = Security(get_api_key),
):
    raw_search = (search or "").strip()
    normalized_search = raw_search.lower()
    hashed_search = md5(normalized_search.encode("utf-8")).hexdigest() if normalized_search else "none"
    cache_key = f"market-movers:gainers:{timeframe}:{page}:{pageSize}:{sortKey}:{sortOrder}:{hashed_search}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        try:
            return JSONResponse(content=orjson.loads(cached_result))
        except Exception:
            pass

    dataset = _load_market_movers_snapshot("gainers", timeframe=timeframe)

    payload = _paginate_collection(
        dataset,
        page=page,
        page_size=pageSize,
        sort_key=sortKey,
        sort_order=sortOrder,
        default_sort_key="changesPercentage",
        valid_sort_keys={
            "rank",
            "symbol",
            "name",
            "companyName",
            "price",
            "changesPercentage",
            "volume",
            "marketCap",
            "avgVolume",
            "pe",
        },
        field_types={
            "symbol": "string",
            "name": "string",
            "companyName": "string",
        },
        search_term=raw_search,
        search_fields=["symbol", "name", "companyName"],
    )

    payload["timeframe"] = timeframe
    payload["session"] = "markethours"

    _cache_paginated_result(cache_key, payload, ttl=300)

    return JSONResponse(content=payload)


@app.get("/all-hedge-funds")
async def get_all_hedge_funds_data(api_key: str = Security(get_api_key)):

    cache_key = f"all-hedge-funds"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    res = _load_all_hedge_funds_snapshot()

    res = orjson.dumps(res)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 3600) # Set cache expiration time to Infinity

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/searchbar")
async def get_stock(
    query: str = Query(""),
    api_key: str = Security(lambda: None)
) -> JSONResponse:
    if not query:
        return JSONResponse(content=[])

    # Normalize query for index lookup: strip leading "^" then uppercase
    idx_key = query.upper().lstrip("^")
    candidate_symbol = f"^{idx_key}"

    # Try to find an exact symbol match for an index (e.g. "^SPX", "^VIX", "^DJI", etc.)
    index_result = next(
        (item for item in searchbar_data if item.get("symbol", "").upper() == candidate_symbol),
        None
    )
    if index_result:
        return JSONResponse(content=[index_result])

    # Exact ISIN match
    exact_match = next((item for item in searchbar_data if item.get("isin") == query), None)
    if exact_match:
        return JSONResponse(content=[exact_match])

    # Prepare case‑insensitive search regex
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    # Filter by name or symbol
    filtered = [
        item for item in searchbar_data
        if pattern.search(item.get("name") or "") or pattern.search(item.get("symbol") or "")
    ]

    # Score + sort: exact symbol hits first, then by descending marketCap
    results = sorted(
        filtered,
        key=lambda item: (
            0 if item.get("symbol", "").lower() == query.lower() else calculate_score(item, query),
            0 if item.get("marketCap") is None else -item["marketCap"]
        )
    )[:5]

    return JSONResponse(content=orjson.loads(orjson.dumps(results)))


@app.get("/full-searchbar")
async def get_data(api_key: str = Security(get_api_key)):
    
    cache_key = f"full-searchbar"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )


    res = orjson.dumps(searchbar_data)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 3600) # Set cache expiration time to Infinity

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/revenue-segmentation")
async def revenue_segmentation(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"revenue-segmentation-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        #redis_client.expire(cache_key, caching_time) 
        return orjson.loads(cached_result)


    query_template = """
        SELECT 
            revenue_product_segmentation, revenue_geographic_segmentation
        FROM 
            stocks 
        WHERE
            symbol = ?
    """

    query = query_template.format(ticker=ticker)
    cur = con.cursor()
    cur.execute(query, (ticker,))
    result = cur.fetchone()  # Get the first row

    if result is not None:
        product_list = orjson.loads(result[0])
        geographic_list = orjson.loads(result[1])
    else:
        product_list = []
        geographic_list = []

    res_list = [product_list, geographic_list]

    redis_client.set(cache_key, orjson.dumps(res_list))
    redis_client.expire(cache_key, 3600 * 24) # Set cache expiration time to Infinity

    return res_list



@app.post("/index-profile")
async def get_data(data: TickerData, api_key: str = Security(get_api_key)):

    data = data.dict()
    ticker = data['ticker'].upper()
    cache_key = f"index-profile-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/index/profile/{ticker}.json","r") as file:
            res = orjson.loads(file.read())
    except Exception as e:
        print(e)
        res = []
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*24)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/etf-profile")
async def get_data(data: TickerData, api_key: str = Security(get_api_key)):

    data = data.dict()
    ticker = data['ticker'].upper()
    cache_key = f"etf-profile-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/etf/profile/{ticker}.json","r") as file:
            res = orjson.loads(file.read())
    except Exception as e:
        print(e)
        res = []
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*5)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.post("/etf-holdings")
async def etf_holdings(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"etf-holdings-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/etf/holding/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {'holdings': []}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60*10)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/refresh-index-cache")
async def refresh_index_symbols_cache(api_key: str = Security(get_api_key)):
    """Refresh the in-memory index symbols cache"""
    try:
        refresh_index_cache(force=True)
        return {"message": "Index symbols cache refreshed successfully", "status": "success"}
    except Exception as e:
        return {"message": f"Error refreshing cache: {str(e)}", "status": "error"}

@app.get("/index-symbols")
async def get_index_symbols_endpoint(api_key: str = Security(get_api_key)):
    """Get all index symbols for client-side filtering"""
    cache_key = "index-symbols"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    
    with open("json/etf/holding/OEF.json", 'rb') as file:
        oef_data = orjson.loads(file.read())
        sp100_symbols = {holding['symbol'] for holding in oef_data.get('holdings', [])}
        
    # Load S&P 500 (SPY)
    with open("json/etf/holding/SPY.json", 'rb') as file:
        spy_data = orjson.loads(file.read())
        sp500_symbols = {holding['symbol'] for holding in spy_data.get('holdings', [])}
                
    response_data = {
        "sp100": list(sp100_symbols),
        "sp500": list(sp500_symbols)
    }
    
    serialized = orjson.dumps(response_data)
    compressed_data = gzip.compress(serialized)
    
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 24)  # 24 hours
    
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/etf-sector-weighting")
async def etf_holdings(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"etf-sector-weighting-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/etf-sector/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.get("/all-etf-tickers")
async def get_all_etf_tickers(api_key: str = Security(get_api_key)):
    cache_key = f"all-etf-tickers"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/all-symbols/etfs.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    # Compress the JSON data
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/congress-rss-feed")
async def get_congress_rss_feed(api_key: str = Security(get_api_key)):
    cache_key = f"congress-rss-feed"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    try:
        with open(f"json/congress-trading/rss-feed/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60 * 24)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )





@app.post("/historical-sector-price")
async def historical_sector_price(data:FilterStockList, api_key: str = Security(get_api_key)):
    data = data.dict()
    sector = data['filterList']
    cache_key = f"history-price-sector-{sector}"
    cached_result = redis_client.get(cache_key)

    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/sector/{sector}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*60)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



def remove_text_before_operator(text):
    # Find the index of the first occurrence of "Operator"
    operator_index = text.find("Operator")

    # If "Operator" was found, create a new string starting from that index
    if operator_index != -1:
        new_text = text[operator_index:]
        return new_text
    else:
        return "Operator not found in the text."



def extract_names_and_descriptions(text):
    pattern = r'([A-Z][a-zA-Z\s]+):\s+(.*?)(?=\n[A-Z][a-zA-Z\s]+:|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    extracted_data = []
    
    for match in matches:
        name = match[0].strip()
        description = match[1].strip()
        
        # Split the description into sentences
        sentences = re.split(r'(?<=[.!?])\s+', description)
        
        # Add line breaks every 3 sentences
        formatted_description = ""
        for i, sentence in enumerate(sentences, 1):
            formatted_description += sentence + " "
            if i % 3 == 0:
                formatted_description += "<br><br>"
        
        formatted_description = formatted_description.strip()
        
        extracted_data.append({'name': name, 'description': formatted_description})
    
    return extracted_data


@app.post("/earnings-call-transcripts")
async def get_earnings_call_transcripts(data:TranscriptData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker']
    year = data['year']
    quarter = data['quarter']
    cache_key = f"earnings-call-transcripts-{ticker}-{year}-{quarter}"
    cached_result = redis_client.get(cache_key)

    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}?year={year}&quarter={quarter}&apikey={FMP_API_KEY}"

            async with session.get(url) as response:
                data = (await response.json())[0]
        

        content = remove_text_before_operator(data['content'])
        chat = extract_names_and_descriptions(content)
        res = {'date': data['date'], 'chat': chat}
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*60)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/top-etf-ticker-holder")
async def top_etf_ticker_holder(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"top-etf-{ticker}-holder"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/top-etf-ticker-holder/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*24)  # Set cache expiration time to 1 day
    return res


@app.get("/popular-etfs")
async def get_popular_etfs(api_key: str = Security(get_api_key)):
    cache_key = "popular-etfs"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open("json/mini-plots-index/data.json", 'rb') as file:
            res = orjson.loads(file.read())
            for item in res:
                price_data = item["priceData"]
                last_price_data = price_data[-1]  # Get the last element of priceData
                if last_price_data['value'] == None:
                    last_price_data = price_data[-2]  # If last element is None, take the second last
                
                item["price"] = last_price_data["value"]  # Update priceData with just the value
                del item["priceData"]  # Remove the old key
                del item["previousClose"]  # Remove the old key
    except Exception as e:
        print(f"Error: {e}")
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60*5)  # Set cache expiration time to 5 minutes
    return res


@app.get("/all-etf-providers")
async def get_all_etf_providers(api_key: str = Security(get_api_key)):

    cache_key = f"get-all-etf-providers"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/all-etf-providers/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return res



@app.post("/etf-provider")
async def etf_holdings(data: ETFProviderData, api_key: str = Security(get_api_key)):
    etf_provider = data.etfProvider.lower()
    cache_key = f"etf-provider-{etf_provider}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/etf/provider/{etf_provider}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60*10)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/etf-new-launches")
async def etf_provider(api_key: str = Security(get_api_key)):
    cache_key = f"etf-new-launches"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/etf-new-launches/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60*10)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/etf-bitcoin-list")
async def get_etf_bitcoin_list(api_key: str = Security(get_api_key)):

    cache_key = f"get-etf-bitcoin-list"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/etf-bitcoin-list/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []
    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return res


@app.post("/analyst-estimate")
async def get_analyst_estimate(data:TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"get-analyst-estimates-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/analyst-estimate/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return res


@app.post("/insider-trading")
async def get_insider_trading(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"insider-trading-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/insider-trading/history/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
            res = [
                {
                    **{k: v for k, v in item.items() if k not in ("securityName", "formType", "url", "directOrIndirect", "acquisitionOrDisposition", "companyCik")},
                    "value": item["securitiesTransacted"] * item["price"]
                }
                for item in res
            ]


    except:
        res = []

    # Compress the JSON data
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

    

@app.post("/insider-trading-statistics")
async def get_insider_trading_statistics(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"insider-trading-statistics-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/insider-trading/statistics/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())[0]
    except:
        res = {}
    
    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return res

@app.post("/get-executives")
async def get_executives(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"get-executives-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/executives/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return res

@app.post("/get-sec-filings")
async def get_sec_filings(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"get-sec-filings-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/sec-filings/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    # Compress the JSON data
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.post("/ipo-calendar")
async def get_ipo_calendar(data:IPOData, api_key: str = Security(get_api_key)):
    year = data.year
    cache_key = f"ipo-calendar-{year}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/ipo-calendar/data.json", 'rb') as file:
            res = orjson.loads(file.read())
        if year != 'all':
            res = [entry for entry in res if entry['date'].startswith(year)]
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*5)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/heatmap")
async def get_heatmap(data: GeneralData, api_key: str = Security(get_api_key)):
    time_period = data.params
    etf_symbol = data.etf or "SPY"
    cache_key = f"heatmap-{etf_symbol}-{time_period}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    
    try:
        with open(f"json/heatmap/{etf_symbol}_{time_period}.html", 'r', encoding='utf-8') as file:
            html_content = file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Heatmap file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading heatmap file: {str(e)}")
    
    # Compress the HTML content
    compressed_data = gzip.compress(html_content.encode('utf-8'))
    
    # Cache the compressed HTML
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60 * 5)  # Set cache expiration time to 5 min
    
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={
            "Content-Encoding": "gzip",
        }
    )



@app.post("/pre-post-quote")
async def get_pre_post_quote(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"get-pre-post-quote-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/pre-post-quote/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60)  # Set cache expiration time to 1 day
    return res

@app.post("/get-quote")
async def get_pre_post_quote(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"get-quote-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/quote/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60)  # Set cache expiration time to 1 day
    return res



@app.post("/options-contract-history")
async def get_data(data:OptionContract, api_key: str = Security(get_api_key)):
    contract_id = data.contract
    ticker = data.ticker
    cache_key = f"options-contract-history-{ticker}-{contract_id}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/all-options-contracts/{ticker}/{contract_id}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*60)
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/options-gex-dex")
async def get_data(data:GreekExposureData, api_key: str = Security(get_api_key)):
    ticker = data.params.upper()
    category = data.category.lower()
    type = data.type

    cache_key = f"options-gex-dex-{ticker}-{category}-{type}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        if len(type) > 0:
            with open(f"json/gex-dex/{category}/{type}/{ticker}.json", 'rb') as file:
                data = orjson.loads(file.read())
        else:
            with open(f"json/gex-dex/{category}/{ticker}.json", 'rb') as file:
                data = orjson.loads(file.read())
    except:
        data = []


    data = orjson.dumps(data)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*60)
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/options-oi")
async def get_data(data: ParamsData, api_key: str = Security(get_api_key)):
    ticker = data.params.upper()
    category = data.category.lower()
    cache_key = f"options-oi-{ticker}-{category}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/oi/{category}/{ticker}.json", 'rb') as file:
            data = orjson.loads(file.read())
            
            if category == 'strike':
                # Collect all items across all expiration dates for threshold calculation
                all_items = []
                for expiration_date, items in data.items():
                    all_items.extend(items)
                
                # Calculate threshold based on combined call + put OI
                val_sums = [item["call_oi"] + item["put_oi"] for item in all_items]
                threshold = np.percentile(val_sums, 0.05)
                
                # Filter each expiration date's data based on threshold
                filtered_data = {}
                for expiration_date, items in data.items():
                    filtered_items = [
                        item for item in items 
                        if (item["call_oi"] + item["put_oi"]) >= threshold
                    ]
                    # Only include expiration dates that have items after filtering
                    if filtered_items:
                        filtered_data[expiration_date] = filtered_items
                
                data = filtered_data
                
    except:
        data = {}
        
    data = orjson.dumps(data)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*15)
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )





@app.post("/options-gex-ticker")
async def get_options_flow_ticker(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"options-gex-{ticker}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    try:
        with open(f"json/options-gex/companies/{ticker}.json", 'rb') as file:
            res_list = orjson.loads(file.read())
    except:
        res_list = []

    data = orjson.dumps(res_list)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 5 min

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/options-chain-statistics")
async def get_options_chain(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"options-chain-statistics-{ticker}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    try:
        with open(f"json/options-chain-statistics/{ticker}.json", 'rb') as file:
            res_list = orjson.loads(file.read())
    except:
        res_list = {}

    data = orjson.dumps(res_list)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*15)  # Set cache expiration time to 5 min

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/options-historical-flow")
async def get_options_chain(data:HistoricalDate, api_key: str = Security(get_api_key)):
    selected_date = data.date
    cache_key = f"options-historical-flow-{selected_date}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    try:
        with open(f"json/options-historical-data/flow-data/{selected_date}.json", 'rb') as file:
            res_list = orjson.loads(file.read())
    except:
        res_list = []
    data = orjson.dumps(res_list)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 5 min

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



fields_to_remove = {'exchange', 'tradeCount', 'description', 'aggressor_ind',"ask","bid","midpoint","trade_count"}
@app.get("/options-flow-feed")
async def get_options_flow_feed(api_key: str = Security(get_api_key)):
    cache_key = f"options-flow-feed"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    try:
        with open(f"json/options-flow/feed/data.json", 'rb') as file:
            data = orjson.loads(file.read())
            res_list = [{k: v for k, v in item.items() if k not in fields_to_remove} for item in data]
            if len(res_list) > 10_000:
                def _to_float(value):
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return 0.0

                def meets_threshold(item):
                    volume = _to_float(item.get("volume"))
                    premium_source = item.get("cost_basis")
                    if premium_source in (None, "", 0):
                        premium_source = item.get("premium")
                    premium = _to_float(premium_source)
                    return volume >= 100 and premium >= 50_000

                res_list = [item for item in res_list if meets_threshold(item)]
    except:
        res_list = []
    data = orjson.dumps(res_list)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.post("/options-insight")
async def get_options_flow_stream(data: OptionsInsight, api_key: str = Security(get_api_key)):
    options_data = data.optionsData

    # Check cache first
    cache_key = f"options-insight-{options_data}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        # If cached, return as a single chunk
        decompressed = gzip.decompress(cached_result)
        result = orjson.loads(decompressed)
        
        async def cached_stream():
            yield orjson.dumps({"content": result["analysis"]}) + b"\n"
        
        return StreamingResponse(
            cached_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive"
            }
        )
    
    # Format the options data as a string for analysis
    formatted_data = f"Analyze this options order flow data: {str(options_data)}"
    
    # Prepare messages for agent
    messages = [
        {"role": "user", "content": formatted_data}
    ]
    
    # Add today's date as context
    today_date = datetime.now().strftime("%B %d, %Y")
    date_context = {
        "role": "system",
        "content": f"Today's date is {today_date}. Use this for any date-related queries or when referring to current market conditions."
    }
    messages = [date_context] + messages
    
    # Agent setup for options insight
    model_settings = ModelSettings(
        tool_choice="auto",
        parallel_tool_calls=True,
        reasoning={"effort": "low"},
        text={ "verbosity": "low" }
    )

    agent = Agent(
        name="Stocknear AI Agent",
        instructions=OPTIONS_INSIGHT_INSTRUCTION,
        model=os.getenv("CHAT_MODEL"),
        tools=[],  # No tools needed for options insight analysis
        model_settings=model_settings
    )
    
    async def event_generator():
        full_content = ""
        
        try:
            result = Runner.run_streamed(agent, input=messages)
            async for event in result.stream_events():
                try:
                    # Process only raw_response_event events
                    if event.type == "raw_response_event":
                        delta = getattr(event.data, "delta", "")
                        if not delta:
                            continue
                        
                        full_content += delta
                        yield orjson.dumps({"content": full_content}) + b"\n"
                        
                except Exception as e:
                    print(f"Event processing error: {e}")
                    continue
            
            # Cache the complete response for future use
            if full_content:
                result = {"analysis": full_content}
                compressed_data = gzip.compress(orjson.dumps(result))
                redis_client.set(cache_key, compressed_data)
                redis_client.expire(cache_key, 60*5)
                
        except Exception as e:
            print(f"Streaming error: {e}")
            yield orjson.dumps({"error": str(e)}) + b"\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive"
        }
    )


@app.get("/dark-pool-flow-feed")
async def get_dark_pool_feed(api_key: str = Security(get_api_key)):
    cache_key = f"dark-pool-flow-feed"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/dark-pool/feed/data.json", "r") as file:
            res_list = orjson.loads(file.read())

        res_list = [item for item in res_list if float(item['premium']) >= 1E6]

    except:
        res_list = []
        
    data = orjson.dumps(res_list)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 10 * 60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/top-analysts")
async def get_all_analysts(api_key: str = Security(get_api_key)):
    cache_key = f"top-analysts"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/analyst/top-analysts.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*15)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/top-analysts-stocks")
async def get_all_analysts(api_key: str = Security(get_api_key)):
    cache_key = f"top-analysts-stocks"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/analyst/top-stocks.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*15)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/analyst-stats")
async def get_all_analysts(data:AnalystId, api_key: str = Security(get_api_key)):
    analyst_id = data.analystId

    cache_key = f"analyst-stats-{analyst_id}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/analyst/analyst-db/{analyst_id}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*15)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/analyst-flow")
async def get_all_analysts(api_key: str = Security(get_api_key)):

    cache_key = f"analyst-flow"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/analyst/flow-data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*5)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/wiim")
async def get_wiim(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"wiim-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/wiim/company/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())[:5]
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*2)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/dashboard-info")
async def get_dashboard_info(api_key: str = Security(get_api_key)):
    # Extract user-specified sections

    # Build cache key based on settings
    cache_key = f"dashboard-info"
    cached = redis_client.get(cache_key)
    if cached:
        return StreamingResponse(
            io.BytesIO(cached),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    # Load full JSON
    try:
        with open("json/dashboard/data.json", "rb") as f:
            full_res = orjson.loads(f.read())
    except FileNotFoundError:
        full_res = {}

    # Filter response based on user settings
    filtered_res = { key: full_res.get(key) for key in ['gainers','losers','optionsFlow','upcomingEarnings','marketStatus'] if key in full_res }

    # Serialize and compress
    compressed = gzip.compress(orjson.dumps(full_res))

    # Cache for 2 minutes
    redis_client.set(cache_key, compressed)
    redis_client.expire(cache_key, 60)

    return StreamingResponse(
        io.BytesIO(compressed),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/politician-stats")
async def get_politician_stats(data:PoliticianId, api_key: str = Security(get_api_key)):
    politician_id = data.politicianId.lower()
    cache_key = f"politician-stats-{politician_id}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/congress-trading/politician-db/{politician_id}.json", 'rb') as file:
            res_list = orjson.loads(file.read())
    except:
        res_list = {}

    data = orjson.dumps(res_list)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/all-politicians")
async def get_all_politician(api_key: str = Security(get_api_key)):
    
    cache_key = f"all-politician"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    
    try:
        with open(f"json/congress-trading/search_list.json", 'rb') as file:
            res_list = orjson.loads(file.read())
    except:
        res_list = []

    data = orjson.dumps(res_list)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/historical-dark-pool")
async def get_dark_pool(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"historical-dark-pool-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/dark-pool/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/dark-pool-level")
async def get_dark_pool(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"dark-pool-level-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/dark-pool/price-level/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*5)
    
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/fda-calendar")
async def get_market_maker(api_key: str = Security(get_api_key)):
    cache_key = f"fda-calendar"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/fda-calendar/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60*15)  # Set cache expiration time to 1 day
    return res


@app.post("/fail-to-deliver")
async def get_fail_to_deliver(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"fail-to-deliver-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/fail-to-deliver/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/analyst-insight")
async def get_analyst_insight(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"analyst-insight-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/analyst/insight/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*15)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/implied-volatility")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"implied-volatility-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/implied-volatility/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except Exception as e:
        print(e)
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*60)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/hottest-contracts")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"hottest-contracts-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/hottest-contracts/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*10)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/unusual-activity")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"unusual-activity-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/unusual-activity/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*10)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/max-pain")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"max-pain-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/max-pain/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*180)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/greeks")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"greeks-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/greeks/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*180)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.get("/reddit-tracker")
async def get_reddit_tracker(api_key: str = Security(get_api_key)):
    cache_key = f"reddit-tracker"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/reddit-tracker/wallstreetbets/trending.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*5)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.post("/historical-market-cap")
async def get_historical_market_cap(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"historical-market-cap-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/market-cap/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/historical-revenue")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"historical-revenue-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/revenue/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/economic-indicator")
async def get_economic_indicator(api_key: str = Security(get_api_key)):
    cache_key = f"economic-indicator"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/economic-indicator/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/sector-industry-overview")
async def get_industry_overview(api_key: str = Security(get_api_key)):
    cache_key = f"sector-industry-overview"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/industry/overview.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/sector-overview")
async def get_sector_overview(api_key: str = Security(get_api_key)):
    cache_key = f"sector-overview"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/industry/sector-overview.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/industry-stocks")
async def get_sector_overview(data: FilterStockList, api_key: str = Security(get_api_key)):
    filter_list = data.filterList.lower()
    cache_key = f"industry-stocks-{filter_list}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/industry/industries/{filter_list}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60*15)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/industry-overview")
async def get_industry_overview(api_key: str = Security(get_api_key)):
    cache_key = f"industry-overview"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/industry/industry-overview.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/earnings-statistics")
async def get_next_earnings(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"earnings-statistics-{ticker}"
    
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        # Boundaries (UTC)
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # --- Earnings ---
        with open(f"json/earnings/raw/{ticker}.json", "rb") as file:
            raw_earnings = orjson.loads(file.read())

        # Keep only entries within the last 5 years AND not in the future (fast string compare)
        earnings_data = [
            e for e in raw_earnings
            if isinstance(e.get("date"), str) and e["date"] <= today
        ]

        # Sort ascending by date (oldest -> newest)
        earnings_data.sort(key=lambda x: x["date"])
     

        res = {"historicalEarnings": earnings_data}
    except (FileNotFoundError, orjson.JSONDecodeError):
        res = {}

    compressed_data = gzip.compress(orjson.dumps(res))
    # setex in one call (TTL 15 minutes)
    redis_client.setex(cache_key, 15 * 60, compressed_data)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/next-earnings")
async def get_next_earnings(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"next-earnings-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/earnings/next/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,15*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/earnings-surprise")
async def get_surprise_earnings(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"earnings-surprise-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/earnings/surprise/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,15*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/price-action-earnings")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"price-action-earnings-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/earnings/past/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/info-text")
async def get_info_text(data:InfoText, api_key: str = Security(get_api_key)):
    parameter = data.parameter
    cache_key = f"info-text-{parameter}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/info-text/data.json", 'rb') as file:
            res = orjson.loads(file.read())[parameter]
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/sentiment-tracker")
async def get_fomc_impact(api_key: str = Security(get_api_key)):

    cache_key = f"sentiment-tracker"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/tracker/sentiment/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,5*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

def compute_ttm_from_quarterly(quarterly_data):
    """
    Compute TTM (Trailing Twelve Months) from quarterly data.
    For cumulative metrics: sum last 4 quarters
    For snapshot metrics: take latest value
    """
    if not quarterly_data:
        return []

    ttm_metrics = []

    for metric in quarterly_data:
        metric_type = metric.get('type', 'cumulative')
        values = metric.get('values', [])

        if not values:
            continue

        # Sort values by date
        sorted_values = sorted(values, key=lambda x: x['date'])

        ttm_values = []

        # Use sliding window of 4 quarters for TTM
        for i in range(3, len(sorted_values)):
            window = sorted_values[i-3:i+1]  # Last 4 quarters including current

            if metric_type == 'snapshot':
                # For snapshot metrics, just use the latest value
                ttm_val = window[-1]['val']
                percent_revenue = window[-1].get('percentRevenue', 'N/A')
            else:
                # For cumulative metrics, sum the last 4 quarters
                ttm_val = sum(q['val'] for q in window if q.get('val') is not None)
                # Average the percentRevenue if available
                percent_revenues = [q.get('percentRevenue') for q in window if q.get('percentRevenue') not in [None, 'N/A']]
                percent_revenue = sum(percent_revenues) / len(percent_revenues) if percent_revenues else 'N/A'

            ttm_entry = {
                'date': sorted_values[i]['date'],
                'val': ttm_val,
                'valueType': window[-1].get('valueType', 'NUMBER')
            }

            if percent_revenue != 'N/A':
                ttm_entry['percentRevenue'] = percent_revenue
            else:
                ttm_entry['percentRevenue'] = 'N/A'

            ttm_values.append(ttm_entry)

        if ttm_values:
            ttm_metrics.append({
                'name': metric.get('name'),
                'type': metric_type,
                'category': metric.get('category'),
                'values': ttm_values
            })

    return ttm_metrics

'''
def compute_annual_from_quarterly(quarterly_data):
    """
    Compute annual data from quarterly data.
    For cumulative metrics: sum 4 quarters ending in Q4 (12-31)
    For snapshot metrics: take Q4 value
    """
    if not quarterly_data:
        return []

    annual_metrics = []

    for metric in quarterly_data:
        metric_type = metric.get('type', 'cumulative')
        values = metric.get('values', [])

        if not values:
            continue

        # Sort values by date
        sorted_values = sorted(values, key=lambda x: x['date'])

        # Group by fiscal year (based on Q4 ending date)
        annual_values = []

        for i, val in enumerate(sorted_values):
            date_str = val['date']
            # Check if this is a Q4 (ends in 12-31 or 12-30)
            if '-12-3' in date_str:
                # This is Q4, compute annual
                if i >= 3:
                    # We have at least 4 quarters
                    window = sorted_values[i-3:i+1]

                    if metric_type == 'snapshot':
                        annual_val = window[-1]['val']
                        percent_revenue = window[-1].get('percentRevenue', 'N/A')
                    else:
                        annual_val = sum(q['val'] for q in window if q.get('val') is not None)
                        percent_revenues = [q.get('percentRevenue') for q in window if q.get('percentRevenue') not in [None, 'N/A']]
                        percent_revenue = sum(percent_revenues) / len(percent_revenues) if percent_revenues else 'N/A'

                    annual_entry = {
                        'date': date_str,
                        'val': annual_val,
                        'valueType': window[-1].get('valueType', 'NUMBER')
                    }

                    if percent_revenue != 'N/A':
                        annual_entry['percentRevenue'] = percent_revenue
                    else:
                        annual_entry['percentRevenue'] = 'N/A'

                    annual_values.append(annual_entry)

        if annual_values:
            annual_metrics.append({
                'name': metric.get('name'),
                'type': metric_type,
                'category': metric.get('category'),
                'values': annual_values
            })

    return annual_metrics
'''

@app.post("/business-metrics")
async def get_data(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"business-metrics-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    # Load quarterly data
    quarterly_data = await load_json_async(f"json/business-metrics/quarterly/{ticker}.json")

    if quarterly_data is None:
        quarterly_data = []

    ttm_data = compute_ttm_from_quarterly(quarterly_data)
    #annual_data = compute_annual_from_quarterly(quarterly_data)

    res = {
        'quarterly': quarterly_data,
        'ttm': ttm_data,
        #'annual': annual_data
    }

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/insider-tracker")
async def get_insider_tracker(api_key: str = Security(get_api_key)):
    cache_key = f"insider-tracker"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/tracker/insider/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,5*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/statistics")
async def get_statistics(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"statistics-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/statistics/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/valuation")
async def get_statistics(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"valuation-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/valuation/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60*1500)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/list-category")
async def get_statistics(data: FilterStockList, api_key: str = Security(get_api_key)):
    filter_list = data.filterList.lower()
    cache_key = f"filter-list-{filter_list}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    if filter_list in ['financial','healthcare','technology','industrials','consumer-cyclical','real-estate','basic-materials','communication-services','energy','consumer-defensive','utilities']:
        category_type = 'sector'
    elif filter_list == 'reits':
        category_type = 'industry'
    elif filter_list in ['covered-call-etfs','monthly-dividend-etfs','ethereum-etfs','spacs-stocks','highest-call-volume','highest-put-volume','monthly-dividend-stocks','most-buybacks','online-gambling','metaverse','sports-betting','virtual-reality','online-dating','pharmaceutical-stocks','gaming-stocks','augmented-reality','electric-vehicles','car-company-stocks','esports','clean-energy','mobile-games','social-media-stocks','ai-stocks','highest-option-premium','highest-option-iv-rank','highest-open-interest','highest-open-interest-change','most-shorted-stocks','most-ftd-shares','highest-income-tax','most-employees','highest-revenue','top-rated-dividend-stocks','penny-stocks','overbought-stocks','oversold-stocks','faang','magnificent-seven','ca','cn','de','gb','il','in','jp','nyse','nasdaq','amex','dowjones','sp500','nasdaq100','all-etf-tickers','all-stock-tickers']:
        category_type = 'stocks-list'
    elif filter_list in ['dividend-kings','dividend-aristocrats']:
        category_type = 'dividends'
    else:
        category_type = 'market-cap'
    try:
        with open(f"json/{category_type}/list/{filter_list}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60*10)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/pre-after-market-movers")
async def get_statistics(data: ParamsData, api_key: str = Security(get_api_key)):
    params = data.params
    category = data.category
    cache_key = f"pre-after-market-movers-{category}-{params}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/market-movers/{category}/{params}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {'gainers': [], 'losers': []}
        
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60*15)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/profile")
async def get_statistics(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"profile-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/profile/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
        
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/market-flow")
async def get_market_flow(api_key: str = Security(get_api_key)):
    cache_key = f"market-flow"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/market-flow/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
        
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,30)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/news-flow")
async def get_data(api_key: str = Security(get_api_key)):
    cache_key = f"news-flow"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/wiim/flow/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/potus-tracker")
async def get_data(api_key: str = Security(get_api_key)):
    cache_key = f"potus-tracker"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/tracker/potus/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
        
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60*15)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/fear-and-greed")
async def get_data(api_key: str = Security(get_api_key)):
    cache_key = f"fear-and-greed"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/fear-and-greed/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
        
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60*15)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/market-seasonality")
async def get_data(api_key: str = Security(get_api_key)):
    cache_key = f"market-seasonality"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/market-seasonality/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
        
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,60*15)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/price-analysis")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"price-analysis-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/price-analysis/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
    
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/ai-score")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"ai-score-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/ai-score/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
    
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/contract-lookup-summary")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"contract-lookup-summary-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/options-contract-lookup/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
    
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,15*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/short-interest")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"short-interest-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/share-statistics/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
    
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



async def fetch_data(client, endpoint, ticker):
    url = f"{API_URL}{endpoint}"
    try:
        response = await client.post(
            url,
            json={"ticker": ticker},
            headers={"X-API-KEY": STOCKNEAR_API_KEY}
        )
        response.raise_for_status()
        # Parse the JSON response
        return {endpoint: response.json()}
    except Exception as e:
        return {endpoint: {"error": str(e)}}

@app.post("/bulk-data")
async def get_stock_data(data: BulkList, api_key: str = Security(get_api_key)):
    endpoints = data.endpoints
    ticker = data.ticker.upper()

    # Create tasks for each endpoint concurrently.
    tasks = [fetch_data(client, endpoint, ticker) for endpoint in endpoints]
    results = await asyncio.gather(*tasks)

    # Combine the results into a single dictionary.
    combined_data = {k: v for result in results for k, v in result.items()}
    return combined_data


@app.post("/bulk-download")
async def get_data(data: BulkDownload, api_key: str = Security(get_api_key)):
    # Ensure tickers are uppercase.
    tickers = [ticker.upper() for ticker in data.tickers]
    selected_data_items = [item for item in data.bulkData if item.get("selected") is True]

    # Mapping file paths for non-Options data types.
    DATA_TYPE_PATHS = {
        "Stock Price": "json/historical-price/max/{ticker}.json",
        "Dividends": "json/dividends/companies/{ticker}.json",
        "Dark Pool": "json/dark-pool/companies/{ticker}.json",
    }

    # Create an in-memory binary stream for the zip archive.
    memory_file = io.BytesIO()

    # Open the zipfile for writing into the memory stream.
    with zipfile.ZipFile(memory_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ticker in tickers:
            for data_item in selected_data_items:
                data_type_name = data_item.get("name")
                
                # Special handling for Price Data.
                if data_type_name == "Price Data":
                    try:
                        with open(f"json/historical-price/max/{ticker}.json", 'rb') as file:
                            res = orjson.loads(file.read())
                    except Exception:
                        res = []
                    
                    # Read adjusted price data.
                    try:
                        with open(f"json/historical-price/adj/{ticker}.json", 'rb') as file:
                            adj_res = orjson.loads(file.read())
                    except Exception:
                        adj_res = []
                    
                    # Map adjusted entries by date.
                    adj_by_date = {entry["date"]: entry for entry in adj_res if "date" in entry}
                    
                    # Merge adjusted data into the regular records.
                    for record in res:
                        date_key = record.get("time")
                        if date_key in adj_by_date:
                            adj_entry = adj_by_date[date_key]
                            record["adjOpen"]  = adj_entry.get("adjOpen")
                            record["adjHigh"]  = adj_entry.get("adjHigh")
                            record["adjLow"]   = adj_entry.get("adjLow")
                            record["adjClose"] = adj_entry.get("adjClose")
                    
                    json_data = res

                    # Convert and write CSV for Price Data.
                    csv_buffer = io.StringIO()
                    csv_writer = csv.writer(csv_buffer)
                    if json_data and isinstance(json_data, list) and len(json_data) > 0:
                        headers = list(json_data[0].keys())
                        csv_writer.writerow(headers)
                        for row in json_data:
                            csv_writer.writerow([row.get(key, "") for key in headers])
                    else:
                        csv_writer.writerow(["No data available"])
                    
                    zip_csv_path = f"{data_type_name}/{ticker}.csv"
                    zf.writestr(zip_csv_path, csv_buffer.getvalue())

                # Special handling for Options.
                elif data_type_name == "Options":
                    # Load the options historical data.
                    try:
                        with open(f"json/options-historical-data/companies/{ticker}.json", 'rb') as file:
                            options_data = orjson.loads(file.read())
                    except Exception:
                        options_data = []
                    


                    # Add Unusual Activity data under Options/Unusual_Activity.
                    try:
                        with open(f"json/unusual-activity/{ticker}.json", 'rb') as file:
                            unusual_data = orjson.loads(file.read())
                    except Exception:
                        unusual_data = []
                    
                    csv_buffer_unusual = io.StringIO()
                    csv_writer_unusual = csv.writer(csv_buffer_unusual)
                    if unusual_data and isinstance(unusual_data, list) and len(unusual_data) > 0:
                        headers = list(unusual_data[0].keys())
                        csv_writer_unusual.writerow(headers)
                        for row in unusual_data:
                            csv_writer_unusual.writerow([row.get(key, "") for key in headers])
                    else:
                        csv_writer_unusual.writerow(["No data available"])
                    
                    zip_csv_unusual_path = f"Options/Unusual_Activity/{ticker}.csv"
                    zf.writestr(zip_csv_unusual_path, csv_buffer_unusual.getvalue())

                    # Also add the historical options data into a separate Historical folder.
                    csv_buffer_hist = io.StringIO()
                    csv_writer_hist = csv.writer(csv_buffer_hist)
                    if options_data and isinstance(options_data, list) and len(options_data) > 0:
                        headers = list(options_data[0].keys())
                        csv_writer_hist.writerow(headers)
                        for row in options_data:
                            csv_writer_hist.writerow([row.get(key, "") for key in headers])
                    else:
                        csv_writer_hist.writerow(["No data available"])
                    
                    zip_csv_hist_path = f"Options/Historical/{ticker}.csv"
                    zf.writestr(zip_csv_hist_path, csv_buffer_hist.getvalue())

                    # --- OI Data Handling ---
                    # Create two subfolders: one for "Strike" and another for "Expiry".
                    for category in ["strike", "expiry"]:
                        try:
                            with open(f"json/oi/{category}/{ticker}.json", 'rb') as file:
                                oi_data = orjson.loads(file.read())
                                # For "strike", filter data using the 85th percentile threshold.
                                if category == 'strike' and oi_data and isinstance(oi_data, list):
                                    val_sums = [item["call_oi"] + item["put_oi"] for item in oi_data]
                                    threshold = np.percentile(val_sums, 85)
                                    oi_data = [item for item in oi_data if (item["call_oi"] + item["put_oi"]) >= threshold]
                        except Exception:
                            oi_data = []
                        
                        csv_buffer_oi = io.StringIO()
                        csv_writer_oi = csv.writer(csv_buffer_oi)
                        if oi_data and isinstance(oi_data, list) and len(oi_data) > 0:
                            headers = list(oi_data[0].keys())
                            csv_writer_oi.writerow(headers)
                            for row in oi_data:
                                csv_writer_oi.writerow([row.get(key, "") for key in headers])
                        else:
                            csv_writer_oi.writerow(["No data available"])
                        
                        # Capitalize the folder name.
                        folder_category = category.capitalize()
                        zip_csv_oi_path = f"Options/OI/{folder_category}/{ticker}.csv"
                        zf.writestr(zip_csv_oi_path, csv_buffer_oi.getvalue())


                # Handling for other data types.
                else:
                    file_path_template = DATA_TYPE_PATHS.get(data_type_name)
                    if not file_path_template:
                        continue  # Skip if the data type is not mapped.
                    try:
                        with open(file_path_template.format(ticker=ticker), 'rb') as file:
                            json_data = orjson.loads(file.read())
                            if data_type_name == 'Dividends':
                                json_data = json_data.get('history', [])
                                json_data = sorted(json_data, key=lambda item: item['date'])
                    except Exception:
                        json_data = []

                    csv_buffer = io.StringIO()
                    csv_writer = csv.writer(csv_buffer)
                    if json_data and isinstance(json_data, list) and len(json_data) > 0:
                        headers = list(json_data[0].keys())
                        csv_writer.writerow(headers)
                        for row in json_data:
                            csv_writer.writerow([row.get(key, "") for key in headers])
                    else:
                        csv_writer.writerow(["No data available"])
                    
                    zip_csv_path = f"{data_type_name}/{ticker}.csv"
                    zf.writestr(zip_csv_path, csv_buffer.getvalue())

    memory_file.seek(0)
    return StreamingResponse(
        memory_file,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=bulk_data.zip"}
    )




CATEGORY_CONFIG = {
    "price": {
        "path": "json/historical-price/max/{ticker}.json",
        "processor": lambda raw, value_key="amount": sorted(
            [{"date": point["time"], "value": round(point.get(value_key, 0), 2)}
             for point in raw],
            key=lambda x: x["date"]
        )
    },
    "marketCap": {
        "path": "json/market-cap/companies/{ticker}.json",
        "processor": lambda raw, value_key="marketCap": sorted(
            [
                {"date": point["date"], "value": round(point.get(value_key, 0), 2)}
                for point in raw
                if isinstance(point, dict) and point.get("date", "") >= "2000-01-01"
            ],
            key=lambda x: x["date"]
        )
    },
    "dividend": {
        "path": "json/dividends/companies/{ticker}.json",
        "processor": lambda raw, value_key="amount": sorted(
            [{"date": point["date"], "value": round(point.get(value_key, 0), 2)}
             for point in raw['history'] if point["date"] >= "2000-01-01"],
            key=lambda x: x["date"]
        )
    },
    "ratios-quarter": {
        "path": "json/financial-statements/ratios/quarter/{ticker}.json",
        "processor": lambda raw, value_key="amount": sorted(
            [{"date": point["date"], "value": round(point.get(value_key, 0)*100, 2) if value_key in ["dividendPayoutRatio","returnOnAssets","returnOnEquity","returnOnInvestedCapital","grossProfitMargin","operatingProfitMargin","netProfitMargin","ebitdaMargin","effectiveTaxRate"] else round(point.get(value_key, 0), 2)}
             for point in raw if point["date"] >= "2000-01-01"],
            key=lambda x: x["date"]
        )
    },
    "ratios-ttm": {
        "path": "json/financial-statements/ratios/ttm-updated/{ticker}.json",
        "processor": lambda raw, value_key="amount": sorted(
            [{"date": point["date"], "value": round(point.get(value_key, 0)*100, 2) if value_key in ["dividendPayoutRatio","returnOnAssets","returnOnEquity","returnOnInvestedCapital"] else round(point.get(value_key, 0), 2)}
             for point in raw if point["date"] >= "2000-01-01"],
            key=lambda x: x["date"]
        )
    },
    "income": {
        "path": "json/financial-statements/income-statement/ttm/{ticker}.json",
        "processor": lambda raw, value_key="amount": sorted(
            [{"date": point["date"], "value": round(point.get(value_key, 0), 2) if value_key in ["epsDiluted"] else round(point.get(value_key, 0), 2)}
             for point in raw if point["date"] >= "2000-01-01"],
            key=lambda x: x["date"]
        )
    },
    "balance-sheet": {
        "path": "json/financial-statements/balance-sheet-statement/ttm/{ticker}.json",
        "processor": lambda raw, value_key="amount": sorted(
            [{"date": point["date"], "value": round(point.get(value_key, 0), 2) if value_key in ["epsDiluted"] else round(point.get(value_key, 0), 2)}
             for point in raw if point["date"] >= "2000-01-01"],
            key=lambda x: x["date"]
        )
    },
    "income-growth-ttm": {
        "path": "json/financial-statements/income-statement-growth/ttm-updated/{ticker}.json",
        "processor": lambda raw, value_key="amount": sorted(
            [{"date": point["date"], "value": round(point.get(value_key, 0)*100, 2)}
             for point in raw if point["date"] >= "2000-01-01"],
            key=lambda x: x["date"]
        )
    },
    "cash-flow": {
        "path": "json/financial-statements/cash-flow-statement/ttm/{ticker}.json",
        "processor": lambda raw, value_key="amount": sorted(
            [{"date": point["date"], "value": round(point.get(value_key, 0), 0)}
             for point in raw if point["date"] >= "2000-01-01"],
            key=lambda x: x["date"]
        )
    },
    "key-metrics": {
        "path": "json/financial-statements/key-metrics/quarter/{ticker}.json",
        "processor": lambda raw, value_key="amount": sorted(
            [{"date": point["date"], "value": round(point.get(value_key, 0), 0)}
             for point in raw if point["date"] >= "2000-01-01"],
            key=lambda x: x["date"]
        )
    },
    "share-statistics": {
        "path": "json/share-statistics/{ticker}.json",
        "processor": lambda raw, value_key="amount": sorted(
            [{"date": point["recordDate"], "value": round(point.get(value_key, 0), 2)}
             for point in raw['history']],
            key=lambda x: x["date"]
        )
    },
}

INDICATOR_RULES = [
    "marketCap",
    "price",
    "changesPercentage",
    "volume",
    "priceToEarningsRatio",
    "revenue",
    "grossProfit"
]
INDICATOR_DATA_URL = "http://localhost:8000/indicator-data"


async def load_ticker_data(ticker, category):
    """
    Load ticker data based on category object with type and value fields
    
    Args:
        ticker: The ticker symbol
        category: A dictionary with 'type', 'value', and 'name' keys
    """
    # Get category type to determine which configuration to use
    category_type = category.get("type") if isinstance(category, dict) else category
    
    config = CATEGORY_CONFIG.get(category_type)
    if not config:
        return []
    
    try:
        raw_data = await load_json_async(config["path"].format(ticker=ticker))
        value_key = category.get("value",None)
        processed_data = config["processor"](raw_data, value_key=value_key)
        return processed_data
    except Exception as e:
        print(e)
        return []

def create_merged_structure(tickers: list, histories: list, stock_data: dict) -> dict:
    """Create merged data structure for response"""
    merged = {}
    for ticker, history in zip(tickers, histories):
        screener = stock_data.get(ticker, {})
        merged[ticker] = {
            "history": history,
            "changesPercentage": [
                screener.get("cagr1MReturn"),
                screener.get("cagrYTDReturn"),
                screener.get("cagr1YReturn"),
                screener.get("cagr5YReturn"),
                screener.get("cagrMaxReturn"),
            ]
        }
    return merged

@app.post("/compare-data")
async def compare_data_endpoint(data: CompareData, api_key: str = Security(get_api_key)):
    tickers = data.tickerList
    category = data.category
    # Validate input
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")
    
    # Clean and validate tickers
    tickers = [ticker.strip().upper() for ticker in tickers if ticker and ticker.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="No valid tickers provided")
        
    # Use category value for the cache key since that identifies the specific data
    cache_key = f"compare-data-{','.join(tickers)}-{category.get('value', 'default')}"
    
    # Try to return cached response
    if cached := redis_client.get(cache_key):
        return StreamingResponse(
            io.BytesIO(cached),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"},
        )
    
    # Load data for all tickers in parallel
    loaders = [load_ticker_data(ticker, category) for ticker in tickers]
    histories = await asyncio.gather(*loaders, return_exceptions=True)
    
    # Handle any exceptions from the parallel loading
    processed_histories = []
    for i, result in enumerate(histories):
        if isinstance(result, Exception):
            print(f"Error loading data for ticker {tickers[i]}: {result}")
            processed_histories.append([])  # Empty list for failed loads
        else:
            processed_histories.append(result)
    
    # Create base response structure
    merged = create_merged_structure(tickers, processed_histories, stock_screener_data_dict)
    
    # Fetch additional indicator data
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                INDICATOR_DATA_URL,
                json={"tickerList": tickers, "ruleOfList": INDICATOR_RULES},
                headers={"X-API-KEY": STOCKNEAR_API_KEY},
                timeout=10.0
            )
            response.raise_for_status()
            overview = response.json()
    except Exception as e:
        print(f"Error fetching indicator data: {e}")
        overview = []
    
    # Prepare final output
    final_output = {'graph': merged, 'table': overview}
    
    # Cache and return response
    blob = orjson.dumps(final_output)
    compressed = gzip.compress(blob)
    redis_client.setex(cache_key, 3600, compressed)
    return StreamingResponse(
        io.BytesIO(compressed),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )




@app.post("/backtesting")
async def get_data(data: Backtesting, api_key: str = Security(get_api_key)):
    strategy_data = data.strategyData

    tickers = strategy_data['tickers']
    start_date = strategy_data['start_date']
    end_date = strategy_data['end_date']
    buy_conditions = strategy_data.get('buy_condition', [])
    sell_conditions = strategy_data.get('sell_condition', [])
    initial_capital = strategy_data['initial_capital']
    commission = strategy_data['commission']/100 #convert percent into decimal
    stop_loss = strategy_data.get('stop_loss', None)
    profit_taker = strategy_data.get('profit_taker', None)

    cache_key = f"backtesting-{','.join(tickers)}-{commission}-{initial_capital}-{start_date}-{end_date}-{stop_loss}-{profit_taker}-{hash(orjson.dumps([buy_conditions, sell_conditions]))}"

    # Check cache
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    engine = BacktestingEngine(initial_capital=initial_capital, commission=commission)

    try:
        res = await engine.run(
            tickers=tickers,
            buy_conditions=buy_conditions,
            sell_conditions=sell_conditions,
            start_date=start_date,
            end_date=end_date,
            stop_loss=stop_loss,
            profit_taker=profit_taker
        )
      
    except:
        res = {}


    compressed_data = await asyncio.to_thread(
        lambda: gzip.compress(orjson.dumps(res, option=orjson.OPT_SERIALIZE_NUMPY))
    )

    redis_client.setex(cache_key, 60 * 15, compressed_data)
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )




# Define fundamental tools once (DRY principle)
FUNDAMENTAL_TOOLS = [
    get_ticker_quote,
    get_ticker_bull_vs_bear,
    get_ticker_business_metrics,
    get_ticker_income_statement,
    get_ticker_balance_sheet_statement,
    get_ticker_cash_flow_statement,
    get_ticker_ratios_statement,
    get_ticker_key_metrics,
    get_ticker_owner_earnings,
    get_ticker_statistics,
]

TRIGGER_CONFIG = {
    "@optionsdata": [
        get_ticker_options_overview_data,
        get_ticker_open_interest_by_strike_and_expiry,
        get_ticker_max_pain,
        get_ticker_hottest_options_contracts,
        get_ticker_unusual_activity,
        get_ticker_quote,
    ],
    "@optionsflowfeed": [get_latest_options_flow_feed],
    "@analyst": [
        get_ticker_analyst_estimate,
        get_ticker_analyst_rating,
    ],
    "@darkpooldata": [
        get_latest_dark_pool_feed,
        get_ticker_dark_pool,
        get_ticker_quote,
    ],
    "@bullvsbear": [
        get_ticker_bull_vs_bear,
        get_why_priced_moved,
        get_ticker_analyst_estimate,
        get_ticker_analyst_rating,
        get_ticker_quote,
    ],
    "@comparestocks": [
        get_why_priced_moved,
        get_company_data,
        get_ticker_news,
        get_ticker_business_metrics,
        get_ticker_analyst_estimate,
        get_ticker_analyst_rating,
    ],
    "@fundamentaldata": FUNDAMENTAL_TOOLS,
    "@stockscreener": [],
    "@backtesting": [],
    "@warrenbuffet": FUNDAMENTAL_TOOLS,
    "@charliemunger": FUNDAMENTAL_TOOLS,
    "@billackman": FUNDAMENTAL_TOOLS,
    "@michaelburry": FUNDAMENTAL_TOOLS,
    "@peterlynch": FUNDAMENTAL_TOOLS,
    "@benjamingraham": FUNDAMENTAL_TOOLS,
    "@cathiewood": FUNDAMENTAL_TOOLS,
}


# Map triggers to instruction generators
TRIGGER_TO_INSTRUCTION = {
    "@warrenbuffet": generate_buffet_instruction,
    "@charliemunger": generate_munger_instruction,
    "@billackman": generate_ackman_instruction,
    "@michaelburry": generate_burry_instruction,
    "@peterlynch": generate_lynch_instruction,
    "@benjamingraham": generate_graham_instruction,
    "@cathiewood": generate_wood_instruction,
}



def normalize_query(query: str) -> str:
    """Normalize query for case-insensitive matching and handle spaces"""
    return query.strip().lower()

def get_tools_for_query(user_query: str) -> tuple[list, list[str]]:
    """Get tools and matched triggers for query with support for multiple triggers"""
    matched_triggers = []
    combined_tools = []
    seen_tool_names = set()  # Track tool names/ids to avoid duplicates
    
    # Check for all triggers in the query
    for trigger, tools in TRIGGER_CONFIG.items():
        if trigger in user_query:
            matched_triggers.append(trigger)
            # Add tools without duplicates (check by function name or id)
            for tool in tools:
                # Get a unique identifier for the tool (function name or string representation)
                tool_id = getattr(tool, '__name__', str(tool))
                if tool_id not in seen_tool_names:
                    combined_tools.append(tool)
                    seen_tool_names.add(tool_id)
    
    # If no triggers matched, return all tools
    if not matched_triggers:
        return all_tools, []
    
    return combined_tools, matched_triggers


async def create_backtesting_strategy(user_query: str, selected_model: str, model_settings) -> dict | None:
    """Create a backtesting strategy based on user query and return the parsed strategy."""
    try:
        agent = Agent(
            name="Stocknear AI Agent",
            instructions=BACKTESTING_INSTRUCTION,
            model=selected_model,
            tools=[],
            model_settings=model_settings
        )

        formatted_data = f"Create a backtesting strategy based on this information: {user_query}"
        
        messages = [
            {"role": "user", "content": formatted_data}
        ]
        
        today_date = datetime.now().strftime("%Y-%m-%d")
        date_context = {
            "role": "system",
            "content": f"Today's date is {today_date}. Use this for any date-related queries or when referring to current market conditions."
        }
        messages = [date_context] + messages

        # Run the agent and get the complete response
        result = await Runner.run(agent, input=messages)
        response = json.loads(result.final_output)
        if not response:
            print("No response content received")
            return None
        return response
        
    except Exception as e:
        print(f"Error in create_backtesting_strategy: {e}")
        return None
    

def strip_html(content):
    return BeautifulSoup(content, "html.parser").get_text()

@app.post("/chat")
async def get_data(data: ChatRequest, api_key: str = Security(get_api_key)):
    user_query = normalize_query(data.query)
    selected_model = os.getenv('CHAT_MODEL')

    model_settings = ModelSettings(
        tool_choice="auto",
        parallel_tool_calls=True,
        reasoning={"effort": "low"},
        text={ "verbosity": "low" }
    )

    history_messages = data.messages[-10:]
    cleaned_messages = []
    for item in history_messages:
        # Create a copy to avoid modifying the original
        cleaned_item = dict(item)
        if 'callComponent' in cleaned_item:
            del cleaned_item['callComponent']
        if 'sources' in cleaned_item:
            del cleaned_item['sources']  # Remove sources before sending to OpenAI API
        if 'relatedQuestions' in cleaned_item:
            del cleaned_item['relatedQuestions']  # Remove related questions before sending to OpenAI API
        if cleaned_item['role'] == 'system' and '<' in cleaned_item['content']:
            cleaned_item['content'] = strip_html(cleaned_item['content'])
        cleaned_messages.append(cleaned_item)

    history_messages = cleaned_messages

    # Get tools and matched triggers in single pass (now supports multiple)
    selected_tools, matched_triggers = get_tools_for_query(user_query)
    
    # Log if multiple triggers detected
    if len(matched_triggers) > 1:
        print(f"Multiple triggers detected: {matched_triggers}")

    # Handle special triggers
    if "@stockscreener" in matched_triggers:
        try:
            # Extract rules using complete frontend context
            extracted_rules = await extract_screener_rules(user_query)
            formatted_rules = await format_rules_for_screener(extracted_rules)
            # Screen stocks using the Python screener
            context = await python_screener.screen(formatted_rules, limit=20)  # Limit to 20 for token efficiency
            
            # Get only essential data for top results
            top_stocks = context.get('matched_stocks', [])[:10]  # Top 10 for display
            
            # Create minimal stock summaries with filter-relevant data
            stock_summaries = []
            for stock in top_stocks:
                summary = {
                    'symbol': stock.get('symbol'),
                    'name': stock.get('name', '')[:25],  # Truncate long names
                }
                
                # Add data based on the applied filters
                for rule in formatted_rules:
                    rule_name = rule.get('name')
                    if rule_name and rule_name in stock:
                        value = stock.get(rule_name)
                        if value is not None:
                            summary[rule_name] = value
                            
                stock_summaries.append(summary)
            
            # Format applied filters for display using ALL_RULES metadata
            from rule_extractor import ALL_RULES
            
            filter_summary = []
            for rule in formatted_rules:
                rule_name = rule.get('name', '')
                condition = rule.get('condition', '')
                value = rule.get('value', '')
                
                # Get rule metadata from ALL_RULES
                rule_meta = ALL_RULES.get(rule_name, {})
                label = rule_meta.get('label', rule_name)
                var_type = rule_meta.get('varType', None)
                
                # Format value based on variable type
                formatted_value = str(value)
                if var_type == 'percent' or var_type == 'percentSign':
                    formatted_value = f"{value}%"
                elif rule_name == 'price':
                    formatted_value = f"${value}"
                elif rule_name in ['marketCap', 'volume', 'avgVolume'] and isinstance(value, str):
                    # Keep string format for values like "10B", "1M"  
                    formatted_value = value
                
                # Create readable filter description
                filter_summary.append(f"{label} {condition} {formatted_value}")
            
            # Create concise system message
            total_found = context.get('total_matches', 0)
            filters_applied = " AND ".join(filter_summary)
            
            system_msg = {
                "role": "system",
                "content": (
                    f"Stock screener results: {total_found} total stocks found matching '{user_query}'.\n\n"
                    f"Applied Filters: {filters_applied}\n\n"
                    f"Top {len(stock_summaries)} results:\n{json.dumps(stock_summaries, indent=1)}\n\n"
                    f"Provide a clear response with:\n"
                    f"1. Brief explanation of the {len(formatted_rules)} screening criteria applied\n" 
                    f"2. Formatted table showing stocks with their filter-relevant metrics\n"
                    f"Focus on the filter criteria values rather than generic stock data."
                )
            }
            
        except Exception as e:
            print(f"Screener error: {e}")
            # Fallback message
            system_msg = {
                "role": "system", 
                "content": f"Unable to process stock screening query: '{user_query}'. Please try a simpler query like 'most shorted stocks below $10' or 'large cap tech stocks'."
            }
        
        history_messages = [system_msg] + history_messages

    elif "@backtesting" in matched_triggers:
        strategy_data = await create_backtesting_strategy(user_query, selected_model, model_settings)
        tickers = strategy_data['tickers']
        start_date = strategy_data.get('start_date',"2015-01-01")
        end_date = strategy_data.get('end_date', datetime.now().strftime("%Y-%m-%d"))
        buy_conditions = strategy_data.get('buy_condition', [])
        sell_conditions = strategy_data.get('sell_condition', [])
        initial_capital = strategy_data.get('initial_capital',100_000)
        commission = strategy_data.get('commission', 5)/100 #convert percent into decimal
        stop_loss = strategy_data.get('stop_loss', None)
        profit_taker = strategy_data.get('profit_taker', None)

        engine = BacktestingEngine(initial_capital=initial_capital, commission=commission)

        try:
            context = await engine.run(
                tickers=tickers,
                buy_conditions=buy_conditions,
                sell_conditions=sell_conditions,
                start_date=start_date,
                end_date=end_date,
                stop_loss=stop_loss,
                profit_taker=profit_taker
            )
            context.pop('trade_history', None)
            context.pop('plot_data', None)

        except:
            context = {}

        system_msg = {
            "role": "system",
            "content": (
                f"You are a trading assistant. Analyze the backtesting results carefully. "
                f"First, clearly list the strategy's buy and sell rules as provided: {json.dumps(strategy_data, indent=2)}. "
                f"Present the result data of the strategy: {json.dumps(context, indent=2)}. Do not add anything else aftewards."
            )
        }

        history_messages = [system_msg] + history_messages
        
    else:
        # Check if any matched triggers have instructions and combine them
        instruction_contents = []
        investor_names = []
        
        for trigger in matched_triggers:
            if trigger in TRIGGER_TO_INSTRUCTION:
                instruction_contents.append(TRIGGER_TO_INSTRUCTION[trigger]())
                # Extract investor name from trigger using a mapping
                investor_name_map = {
                    "@warrenbuffet": "Warren Buffett",
                    "@peterlynch": "Peter Lynch", 
                    "@charliemunger": "Charlie Munger",
                    "@billackman": "Bill Ackman",
                    "@michaelburry": "Michael Burry",
                    "@benjamingraham": "Benjamin Graham",
                    "@cathiewood": "Cathie Wood"
                }
                investor_name = investor_name_map.get(trigger, trigger.replace("@", "").title())
                investor_names.append(investor_name)
        
        if instruction_contents:
            # Combine multiple investor perspectives
            if len(instruction_contents) > 1:
                combined_content = f"You are analyzing this query from multiple investor perspectives:\n\n"
                for i, (name, content) in enumerate(zip(investor_names, instruction_contents), 1):
                    combined_content += f"--- {name}'s Perspective ---\n{content}\n\n"
                combined_content += "Provide insights from each investor's unique perspective, clearly labeling which analysis comes from which investor. Show how their different philosophies lead to different conclusions about the same investment."
            else:
                combined_content = instruction_contents[0]
            
            system_msg = {
                "role": "system",
                "content": combined_content
            }
            history_messages = [system_msg] + history_messages
    history_messages += [{'role': 'user', "content": user_query}]
    
    # Add today's date as context
    today_date = datetime.now().strftime("%B %d, %Y")
    date_context = {
        "role": "system",
        "content": f"Today's date is {today_date}. Use this for any date-related queries or when referring to current market conditions."
    }
    history_messages = [date_context] + history_messages
  
    # Agent setup
    agent = Agent(
        name="Stocknear AI Agent",
        instructions=CHAT_INSTRUCTION,
        model=selected_model,
        tools=selected_tools,
        model_settings=model_settings
    )

    async def event_generator():
            full_content = ""
            found_end_of_dicts = False
            sources_collected = []  # Track sources from function calls
            tools_called = set()  # Backup tracking of all tools called
            
            def add_source_from_tool(tool_name, tool_args=None):
                """Helper function to add a source from tool information"""
                if tool_name in tools_called:
                    return  # Already added this tool
                    
                tools_called.add(tool_name)
                
                # Get metadata from FUNCTION_SOURCE_METADATA
                metadata = FUNCTION_SOURCE_METADATA.get(tool_name, {})
                
                # Use metadata or fallback to generated name
                friendly_name = metadata.get("name", tool_name.replace("_", " ").title())
                description = metadata.get("description", f"Data from {friendly_name}")
                url_pattern = metadata.get("url_pattern", "")
                
                # Extract ticker from tool arguments  
                ticker = None
                ticker_type = "Stock"  # Default to Stock
                
                if tool_args:
                    # Try different parameter names for ticker
                    ticker = tool_args.get("ticker") or tool_args.get("symbol") or tool_args.get("stock")
                    
                    # Handle plural 'tickers' parameter (array) - create sources for all tickers
                    if not ticker and "tickers" in tool_args:
                        tickers_list = tool_args.get("tickers")
                        if isinstance(tickers_list, list) and len(tickers_list) > 0:
                            # For multiple tickers, create a source for each one
                            for ticker_symbol in tickers_list:
                                if ticker_symbol:
                                    individual_ticker_type = "ETF" if ticker_symbol in etf_symbols else "Stock"
                                    individual_source_url = ""
                                    if url_pattern and ticker_symbol:
                                        asset_type = "etf" if individual_ticker_type == "ETF" else "stocks"
                                        individual_source_url = url_pattern.format(asset_type=asset_type, ticker=ticker_symbol)
                                    
                                    individual_source_info = {
                                        "name": friendly_name,
                                        "description": description,
                                        "function": tool_name,
                                        "ticker": ticker_symbol,
                                        "type": individual_ticker_type,
                                        "url": individual_source_url,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                    sources_collected.append(individual_source_info)
                            return  # Exit early since we've handled multiple tickers
                    
                    # Check if it's an ETF
                    if ticker and ticker in etf_symbols:
                        ticker_type = "ETF"
                
                # Generate the URL based on pattern
                source_url = ""
                if url_pattern and ticker:
                    asset_type = "etf" if ticker_type == "ETF" else "stocks"
                    source_url = url_pattern.format(asset_type=asset_type, ticker=ticker)
                elif url_pattern and not ticker:
                    # For non-ticker specific URLs (like market news)
                    source_url = url_pattern

                source_info = {
                    "name": friendly_name,
                    "description": description,
                    "function": tool_name,
                    "ticker": ticker,
                    "type": ticker_type,
                    "url": source_url,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                sources_collected.append(source_info)
            
            try:
                result = Runner.run_streamed(agent, input=history_messages)
                async for event in result.stream_events():
                    try:
                        # Track function calls for source citations
                        if event.type == "run_item_stream_event" and hasattr(event, 'item'):
                            item = event.item
                            # Check if this is a tool call
                            if hasattr(item, 'type') and item.type == 'tool_call_item':
                                # Extract tool name from raw_item
                                tool_name = None
                                tool_args = {}
                                
                                if hasattr(item, 'raw_item'):
                                    raw_item = item.raw_item
                                    if hasattr(raw_item, 'name'):
                                        tool_name = raw_item.name
                                    
                                    if hasattr(raw_item, 'arguments'):
                                        try:
                                            # Arguments are JSON string
                                            tool_args = json.loads(raw_item.arguments)
                                        except:
                                            pass
                                
                                if tool_name:
                                    # Get metadata from FUNCTION_SOURCE_METADATA
                                    metadata = FUNCTION_SOURCE_METADATA.get(tool_name, {})
                                    
                                    # Use metadata or fallback to generated name
                                    friendly_name = metadata.get("name", tool_name.replace("_", " ").title())
                                    description = metadata.get("description", f"Data from {friendly_name}")
                                    url_pattern = metadata.get("url_pattern", "")
                                    
                                    # Extract ticker from tool arguments  
                                    ticker = None
                                    ticker_type = "Stock"  # Default to Stock
                                    
                                    if tool_args:
                                        # Try different parameter names for ticker
                                        ticker = tool_args.get("ticker") or tool_args.get("symbol") or tool_args.get("stock")
                                        
                                        # Handle plural 'tickers' parameter (array) - create sources for ALL tickers
                                        if not ticker and "tickers" in tool_args:
                                            tickers_list = tool_args.get("tickers")
                                            if isinstance(tickers_list, list) and len(tickers_list) > 0:
                                                # For multiple tickers, create a source for each one
                                                for ticker_symbol in tickers_list:
                                                    if ticker_symbol:
                                                        individual_ticker_type = "ETF" if ticker_symbol in etf_symbols else "Stock"
                                                        individual_source_url = ""
                                                        if url_pattern and ticker_symbol:
                                                            asset_type = "etf" if individual_ticker_type == "ETF" else "stocks"
                                                            individual_source_url = url_pattern.format(asset_type=asset_type, ticker=ticker_symbol)
                                                        elif url_pattern and not ticker_symbol:
                                                            individual_source_url = url_pattern
                                                        
                                                        individual_source_info = {
                                                            "name": friendly_name,
                                                            "description": description,
                                                            "function": tool_name,
                                                            "ticker": ticker_symbol,
                                                            "type": individual_ticker_type,
                                                            "url": individual_source_url,
                                                            "timestamp": datetime.utcnow().isoformat()
                                                        }
                                                        
                                                        # Avoid duplicate sources
                                                        if not any(s["function"] == tool_name and s.get("ticker") == ticker_symbol for s in sources_collected):
                                                            sources_collected.append(individual_source_info)
                                                # After processing multiple tickers, continue to next event
                                                continue
                                        
                                        # Check if it's an ETF
                                        if ticker and ticker in etf_symbols:
                                            ticker_type = "ETF"
                                    
                                    # Generate the URL based on pattern (for single ticker or non-ticker functions)
                                    source_url = ""
                                    if url_pattern and ticker:
                                        asset_type = "etf" if ticker_type == "ETF" else "stocks"
                                        source_url = url_pattern.format(asset_type=asset_type, ticker=ticker)
                                    elif url_pattern and not ticker:
                                        # For non-ticker specific URLs (like market news)
                                        source_url = url_pattern
                                
                                    source_info = {
                                        "name": friendly_name,
                                        "description": description,
                                        "function": tool_name,
                                        "ticker": ticker,
                                        "type": ticker_type,
                                        "url": source_url,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                    
                                    # Avoid duplicate sources (only add if we have a ticker or it's a non-ticker function)
                                    if ticker or (not ticker and url_pattern):
                                        if not any(s["function"] == tool_name and s.get("ticker") == ticker for s in sources_collected):
                                            sources_collected.append(source_info)
                        
                        # Process only raw_response_event events
                        if event.type == "raw_response_event":
                            delta = getattr(event.data, "delta", "")
                            if not delta:
                                continue
                            stripped_delta = delta.strip()
                            # Skip if it's echoing back the question
                            if stripped_delta.lower() == user_query or stripped_delta.lower().startswith(user_query):
                                continue
                            
                            # Check if we've passed all the JSON dictionaries
                            if not found_end_of_dicts:
                                full_content += delta
                                
                                # No need to start here anymore - already running in parallel
                                
                                # First check if content starts with JSON objects
                                temp_content = full_content.strip()
                                if not temp_content:
                                    continue  # No content yet
                                
                                # If content doesn't start with '{', it's regular text - don't filter
                                if not temp_content.startswith('{'):
                                    found_end_of_dicts = True
                                    yield orjson.dumps({
                                        "event": "response",
                                        "content": full_content
                                    }) + b"\n"
                                    continue
                                
                                # Content starts with JSON, so apply filtering logic
                                last_json_end = -1
                                
                                i = 0
                                while i < len(temp_content):
                                    if temp_content[i] == '{':
                                        # Try to find the matching closing brace
                                        brace_count = 1
                                        j = i + 1
                                        in_string = False
                                        escape_next = False
                                        
                                        while j < len(temp_content) and brace_count > 0:
                                            char = temp_content[j]
                                            
                                            if escape_next:
                                                escape_next = False
                                            elif char == '\\':
                                                escape_next = True
                                            elif char == '"' and not escape_next:
                                                in_string = not in_string
                                            elif not in_string:
                                                if char == '{':
                                                    brace_count += 1
                                                elif char == '}':
                                                    brace_count -= 1
                                            
                                            j += 1
                                        
                                        if brace_count == 0:
                                            # Found complete JSON object
                                            last_json_end = j - 1
                                            i = j
                                        else:
                                            # Incomplete JSON object
                                            break
                                    else:
                                        # Found non-JSON content
                                        if last_json_end >= 0:
                                            found_end_of_dicts = True
                                            # Keep only content after all JSON objects
                                            full_content = temp_content[last_json_end + 1:].strip()
                                            if full_content:  # Only yield if there's actual content
                                                yield orjson.dumps({
                                                    "event": "response",
                                                    "content": full_content
                                                }) + b"\n"
                                            break
                                        else:
                                            # No JSON objects found yet, might be at the start
                                            break
                                
                                # If no content after dictionaries yet, don't yield anything
                            else:
                                # We've already found the end of dictionaries, just append new deltas
                                full_content += delta
                                
                                # No need to start here anymore - already running in parallel
                                
                                yield orjson.dumps({
                                    "event": "response",
                                    "content": full_content
                                }) + b"\n"
                                
                    except Exception as e:
                        print(f"Error processing event: {e}")
                        yield orjson.dumps({
                            "event": "error",
                            "message": f"Event processing error: {str(e)}"
                        }) + b"\n"
                
                # Send sources after content is complete
                if sources_collected:
                    yield orjson.dumps({
                        "event": "sources",
                        "sources": sources_collected
                    }) + b"\n"
                
            except Exception as e:
                print(f"Streaming error: {e}")
                yield orjson.dumps({
                    "event": "error",
                    "message": f"Streaming failed: {str(e)}"
                }) + b"\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/newsletter")
async def get_newsletter():
    try:
        with open(f"json/newsletter/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []
    return res
