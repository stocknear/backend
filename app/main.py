# Standard library imports
import random
import io
import gzip
import csv
import re
import os
import secrets
from benzinga import financial_data
from typing import List, Dict, Set
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

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from functools import partial
from datetime import datetime
from utils.helper import load_latest_json, json_to_string, process_request

from openai import OpenAI, AsyncOpenAI
from llm_functions import * # Your function implementations
from contextlib import asynccontextmanager
from functools import lru_cache
from hashlib import md5





# Connection pooling for API client
@asynccontextmanager
async def get_client_session():
    # Create a session with connection pooling
    session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(
            limit=100,  # Max connections
            limit_per_host=100,  # Max connections per host
            keepalive_timeout=60  # Keep connections alive
        )
    )
    try:
        yield session
    finally:
        await session.close()
@lru_cache(maxsize=100)
def get_cached_response(query_hash: str) -> Dict[str, Any]:
    return RESPONSE_CACHE.get(query_hash)



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
CHAT_MODEL = os.getenv("CHAT_MODEL")
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
with open("json/llm/instructions.json","rb") as file:
    INSTRUCTIONS = json_to_string(orjson.loads(file.read()))

function_definitions = get_function_definitions()
function_map = {fn["name"]: globals()[fn["name"]] for fn in function_definitions}
tools_payload = ([{"type": "function", "function": fn} for fn in function_definitions] if function_definitions else None)

# Keep the system instruction separate
system_message = {"role": "system", "content": INSTRUCTIONS}
RESPONSE_CACHE = {}
MAX_CONCURRENT_REQUESTS = 25  # Limit concurrent requests
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
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
  } for row in raw_data if row[3] is not None]
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
fin = financial_data.Benzinga(Benzinga_API_KEY)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url = None)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

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
USER_API_KEY = os.getenv('USER_API_KEY')
VALID_API_KEYS = [STOCKNEAR_API_KEY, USER_API_KEY]
api_key_header = APIKeyHeader(name="X-API-KEY")


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"}
    )

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key not in VALID_API_KEYS:
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
        with open(file_path, 'r') as f:
            data = orjson.loads(f.read())
            # Cache the data in Redis for 10 minutes
            redis_client.set(file_path, orjson.dumps(data), ex=600)
            return data
    except Exception:
        return None


@app.get("/")
async def hello_world():
    return {"stocknear api"}



@app.post("/correlation-ticker")
async def rating_stock(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"correlation-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/correlation/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*24) # Set cache expiration time to 12 hour

    return res



@app.post("/stock-rating")
async def rating_stock(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"stock-rating-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/ta-rating/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*24)  # Set cache expiration time to 1 day
    return res

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
    
@app.post("/export-price-data")
async def get_stock(data: HistoricalPrice, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    time_period = data.timePeriod
    cache_key = f"export-price-data-{ticker}-{time_period}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    if time_period == 'max':
        try:
            with open(f"json/historical-price/max/{ticker}.json", 'rb') as file:
                res = orjson.loads(file.read())
        except:
            res = []
    else:
        try:
            with open(f"json/export/price/{time_period}/{ticker}.json", 'rb') as file:
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


@app.post("/hover-stock-chart")
async def get_hover_stock_chart(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()
    cache_key = f"hover-stock-chart-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/one-day-price/{ticker}.json", 'rb') as file:
            price_data = orjson.loads(file.read())
        with open(f"json/quote/{ticker}.json", 'rb') as file:
            quote_data = orjson.loads(file.read())
        res = {**quote_data, 'history': price_data}
    except:
        res = {}
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


@app.post("/similar-etfs")
async def get_similar_etfs(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"similar-etfs-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        query = """
            SELECT symbol, name, totalAssets, numberOfHoldings
            FROM etfs
            WHERE symbol <> ? AND ABS(totalAssets - (
                SELECT totalAssets FROM etfs WHERE symbol = ?
            )) >= 0.2 * (
                SELECT totalAssets FROM etfs WHERE symbol = ?
            )
            ORDER BY totalAssets DESC
            LIMIT 15
        """

        etf_cursor = etf_con.cursor()
        etf_cursor.execute(query, (ticker, ticker, ticker))
        raw_data = etf_cursor.fetchall()

        result = [
            {"symbol": row[0], "name": row[1], "totalAssets": row[2], "numberOfHoldings": row[3]}
            for row in raw_data
        ]

        if len(result) >= 5:
            result = random.sample(result, k=5)

        result.sort(key=lambda x: x["totalAssets"], reverse=True)  # Sort the list in-place

        etf_cursor.close()  # Close the cursor after executing the query
    except:
        result = []

    redis_client.set(cache_key, orjson.dumps(result))
    redis_client.expire(cache_key, 3600*3600)
    return result


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

@app.get("/mini-plots-index")
async def get_market_movers(api_key: str = Security(get_api_key)):
    cache_key = f"get-mini-plots-index"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/mini-plots-index/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 5*60)

    return res



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

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60)
    return res

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
    if 'symbol' not in rule_of_list:
        rule_of_list.append('symbol')
    if 'name' not in rule_of_list:
        rule_of_list.append('name')
    
    ticker_list = [t.upper() for t in data.tickerList if t is not None]

    combined_results = []
    
    # Load quote data in parallel
    quote_data = await asyncio.gather(*[load_json_async(f"json/quote/{ticker}.json") for ticker in ticker_list])
    quote_dict = {ticker: data for ticker, data in zip(ticker_list, quote_data) if data}

    # Categorize tickers and extract data
    for ticker, quote in quote_dict.items():
        # Determine the ticker type based on the sets
        ticker_type = (
            'etf' if ticker in etf_set else 
            'stock'
        )

        # Filter the quote based on keys in rule_of_list (use data only from quote.json for these)
        filtered_quote = {key: quote.get(key) for key in rule_of_list if key in quote}
        filtered_quote['type'] = ticker_type
        # Add the result to combined_results
        combined_results.append(filtered_quote)

    # Fetch and merge data from stock_screener_data, but exclude price, volume, and changesPercentage
    screener_keys = [key for key in rule_of_list if key not in ['volume', 'marketCap', 'changesPercentage', 'price', 'symbol', 'name']]
    if screener_keys:
        screener_dict = {item['symbol']: {k: v for k, v in item.items() if k in screener_keys} for item in stock_screener_data}
        for result in combined_results:
            symbol = result.get('symbol')
            if symbol in screener_dict:
                # Only merge screener data for keys that are not price, volume, or changesPercentage
                result.update(screener_dict[symbol])

    # Ensure all keys in rule_of_list are present, setting missing ones to None
    for result in combined_results:
        for key in rule_of_list:
            result.setdefault(key, None)
            
    # Serialize and compress the response
    res = orjson.dumps(combined_results)
    compressed_data = gzip.compress(res)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


async def process_watchlist_ticker(ticker, rule_of_list, quote_keys_to_include, screener_dict, etf_set):
    """Optimized single ticker processing with guaranteed rule_of_list keys."""
    ticker = ticker.upper()
    ticker_type = 'stocks'
    if ticker in etf_set:
        ticker_type = 'etf'

    try:
        # Combine I/O operations into single async read
        quote_dict, news_dict, earnings_dict = await asyncio.gather(
            load_json_async(f"json/quote/{ticker}.json"),
            load_json_async(f"json/market-news/companies/{ticker}.json"),
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

    # Process supplemental data
    news = [
        {k: v for k, v in item.items() if k not in ['image', 'text']}
        for item in (news_dict or [])[:5]
    ]
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
            etf_set
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
    
    async def fetch_ticker_data(ticker):
        try:
            news_task = load_json_async(f"json/market-news/companies/{ticker}.json")
            earnings_task = load_json_async(f"json/earnings/next/{ticker}.json")
            
            news_dict, earnings_dict = await asyncio.gather(news_task, earnings_task)
            
            # Process news
            news = []
            if news_dict:
                news = [
                    {key: value for key, value in item.items() if key not in ['image', 'text']}
                    for item in news_dict[:5]
                ]
            
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
        ticker_tasks = [fetch_ticker_data(ticker) for ticker in unique_tickers]
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
async def stock_finder(data:StockScreenerData, api_key: str = Security(get_api_key)):
    rule_of_list = sorted(data.ruleOfList)

    cache_key = f"stock-screener-data-{rule_of_list}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    #For now consider only US Stocks
    us_data_only = [item for item in stock_screener_data if item.get('exchange') != 'PNK']


    always_include = ['symbol', 'marketCap', 'price', 'changesPercentage', 'name','volume','priceToEarningsRatio']

    try:
        filtered_data = [
            {key: item.get(key) for key in set(always_include + rule_of_list) if key in item}
            for item in us_data_only
        ]
    except:
        filtered_data = []


    # Compress the JSON data
    res = orjson.dumps(filtered_data)
    compressed_data = gzip.compress(res)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

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
        return orjson.loads(cached_result)

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

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return res


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

    try:
        with open(f"json/hedge-funds/all-hedge-funds.json", 'rb') as file:
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
        if pattern.search(item.get("name", "")) or pattern.search(item.get("symbol", ""))
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

    query_template = """
        SELECT 
            profile, etfProvider
        FROM 
            etfs
        WHERE
            symbol = ?
    """
    query = query_template.format(ticker=ticker)
    cur = etf_con.cursor()
    cur.execute(query, (ticker,))
    result = cur.fetchone()  # Get the first row
    res = []

    try:
        if result is not None:
            res = orjson.loads(result[0])
            for item in res:
                item['etfProvider'] = result[1]
            #Show only hedge funds that are in the institute.db
    except:
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
    limit = 100
    if cached_result:
        return orjson.loads(cached_result)

    # Check if data is cached; if not, fetch and cache it
    cursor = etf_con.cursor()
    query = "SELECT symbol, name, expenseRatio, totalAssets, numberOfHoldings, inceptionDate FROM etfs ORDER BY inceptionDate DESC LIMIT ?"
    cursor.execute(query, (limit,))
    raw_data = cursor.fetchall()
    cursor.close()

    # Extract only relevant data and sort it
    res = [{'symbol': row[0], 'name': row[1], 'expenseRatio': row[2], 'totalAssets': row[3], 'numberOfHoldings': row[4], 'inceptionDate': row[5]} for row in raw_data]
    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return res

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
    cache_key = f"heatmap-{time_period}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    
    try:
        with open(f"json/heatmap/{time_period}.html", 'r', encoding='utf-8') as file:
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
async def get_data(data:ParamsData, api_key: str = Security(get_api_key)):
    ticker = data.params.upper()
    category = data.category.lower()

    cache_key = f"options-oi-{ticker}-{category}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    try:
        with open(f"json/oi/{category}/{ticker}.json", 'rb') as file:
            data = orjson.loads(file.read())
            if category == 'strike':
                val_sums = [item[f"call_oi"] + item[f"put_oi"] for item in data]
                threshold = np.percentile(val_sums, 85)
                data = [item for item in data if (item[f"call_oi"] + item[f"put_oi"]) >= threshold]     
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

@app.post("/options-stats-ticker")
async def get_options_stats_ticker(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"options-stats-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/options-stats/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*1)
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/raw-options-flow-ticker")
@limiter.limit("500/minute")
async def get_raw_options_flow_ticker(data:OptionsFlowData, request: Request, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    start_date = data.start_date
    end_date = data.end_date
    pagesize = data.pagesize
    page = data.page
    cache_key = f"raw-options-flow-{ticker}-{start_date}-{end_date}-{pagesize}-{page}"
    #print(ticker, start_date, end_date, pagesize, page)
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    try:
        data = fin.options_activity(company_tickers=ticker, date_from=start_date, date_to = end_date, page=page, pagesize=pagesize)
        data = orjson.loads(fin.output(data))['option_activity']
    except Exception as e:
        print(e)
        data = []

    data = orjson.dumps(data)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60)  # Set cache expiration time to 5 min

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/options-flow-ticker")
async def get_options_flow_ticker(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"options-flow-{ticker}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    try:
        data = fin.options_activity(company_tickers=ticker, pagesize=500)
        data = orjson.loads(fin.output(data))['option_activity']
        res_list = []
        keys_to_keep = {'time', 'sentiment','option_activity_type', 'price', 'underlying_price', 'cost_basis', 'strike_price', 'date', 'date_expiration', 'open_interest', 'put_call', 'volume'}
        for item in data:
            filtered_item = {key: value for key, value in item.items() if key in keys_to_keep}
            filtered_item['type'] = filtered_item['option_activity_type'].capitalize()
            filtered_item['sentiment'] = filtered_item['sentiment'].capitalize()
            filtered_item['underlying_price'] = round(float(filtered_item['underlying_price']),2)
            #filtered_item['time'] = (datetime.strptime(filtered_item['time'], '%H:%M:%S')-timedelta(hours=0)).strftime('%H:%M:%S')
            filtered_item['put_call'] = 'Calls' if filtered_item['put_call'] == 'CALL' else 'Puts'
            res_list.append(filtered_item)
    except:
        res_list = []

    data = orjson.dumps(res_list)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*5)  # Set cache expiration time to 5 min

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

@app.post("/options-historical-data-ticker")
async def get_options_chain(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"options-historical-data-{ticker}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    try:
        with open(f"json/options-historical-data/companies/{ticker}.json", 'rb') as file:
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
@app.post("/options-flow-feed")
async def get_options_flow_feed(data: OptionsFlowFeed, api_key: str = Security(get_api_key)):
    order_list = data.orderList

    try:
        with open(f"json/options-flow/feed/data.json", 'rb') as file:
            data = orjson.loads(file.read())
            if len(order_list) > 0:
                data = [item for item in data if item['id'] not in order_list]

            res_list = [{k: v for k, v in item.items() if k not in fields_to_remove} for item in data]
    except:
        res_list = []

    data = orjson.dumps(res_list)
    compressed_data = gzip.compress(data)
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
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
    redis_client.expire(cache_key,60)

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

@app.post("/dashboard-info")
async def get_dashboard_info(data: CustomSettings, api_key: str = Security(get_api_key)):
    # Extract user-specified sections
    custom_settings = data.customSettings + ['marketStatus']

    # Build cache key based on settings
    cache_key = f"dashboard-info-{','.join(custom_settings)}"
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
    filtered_res = { key: full_res.get(key) for key in custom_settings if key in full_res }

    # Serialize and compress
    raw = orjson.dumps(filtered_res)
    compressed = gzip.compress(raw)

    # Cache for 2 minutes
    redis_client.set(cache_key, compressed)
    redis_client.expire(cache_key, 120)

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
        with open(f"json/reddit-tracker/wallstreetbets/data.json", 'rb') as file:
            latest_post = orjson.loads(file.read())[0:25]
    except:
        latest_post = []

    try:
        with open(f"json/reddit-tracker/wallstreetbets/stats.json", 'rb') as file:
            stats = orjson.loads(file.read())
    except:
        stats = []

    try:
        with open(f"json/reddit-tracker/wallstreetbets/trending.json", 'rb') as file:
            trending = orjson.loads(file.read())
    except:
        trending = {}

    res = {'posts': latest_post, 'stats': stats, 'trending': trending}

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*15)

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

@app.post("/business-metrics")
async def get_fomc_impact(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"business-metrics-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/business-metrics/{ticker}.json", 'rb') as file:
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
    elif filter_list in ['highest-call-volume','highest-put-volume','monthly-dividend-stocks','most-buybacks','online-gambling','metaverse','sports-betting','virtual-reality','online-dating','pharmaceutical-stocks','gaming-stocks','augmented-reality','electric-vehicles','car-company-stocks','esports','clean-energy','mobile-games','social-media-stocks','ai-stocks','highest-option-premium','highest-option-iv-rank','highest-open-interest','highest-open-interest-change','most-shorted-stocks','most-ftd-shares','highest-income-tax','most-employees','highest-revenue','top-rated-dividend-stocks','penny-stocks','overbought-stocks','oversold-stocks','faang','magnificent-seven','ca','cn','de','gb','il','in','jp','nyse','nasdaq','amex','dowjones','sp500','nasdaq100','all-etf-tickers','all-stock-tickers']:
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
        with open(f"json/market-flow/overview.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
        
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,2*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/sector-flow")
async def get_data(api_key: str = Security(get_api_key)):
    cache_key = f"sector-flow"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/market-flow/sector.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}
        
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,2*60)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/ticker-flow")
async def get_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"ticker-flow-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/market-flow/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []
        
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key,2*60)

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

@app.get("/egg-price")
async def get_data(api_key: str = Security(get_api_key)):
    cache_key = f"egg-price"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/tracker/potus/egg_price.json", 'rb') as file:
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
            [{"date": point["date"], "value": round(point.get(value_key, 0), 2)}
             for point in raw if point["date"] >= "2000-01-01"],
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
        value_key = category.get("value")
        processed_data = config["processor"](raw_data, value_key=value_key)
        return processed_data
    except:
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
    
    # Use category value for the cache key since that identifies the specific data
    cache_key = f"compare-data-{','.join(tickers)}-{category['value']}"
    
    # Try to return cached response
    if cached := redis_client.get(cache_key):
        return StreamingResponse(
            io.BytesIO(cached),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"},
        )
    
    # Load data for all tickers in parallel
    loaders = [load_ticker_data(ticker, category) for ticker in tickers]
    histories = await asyncio.gather(*loaders, return_exceptions=False)
    
    # Create base response structure
    merged = create_merged_structure(tickers, histories, stock_screener_data_dict)
    
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
        # Log error here if needed
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




# Function to create a hash of the request for caching
def create_query_hash(messages: List) -> str:
    # Create a deterministic hash of the messages
    # Use a more direct approach to extract message data
    message_dicts = []
    for msg in messages:
        if hasattr(msg, 'model_dump'):  # Handle Pydantic models like ChatCompletionMessage
            msg_dict = msg.model_dump()
        elif isinstance(msg, dict):  # Handle plain dictionaries
            msg_dict = msg.copy()  # Use copy instead of dict comprehension
        else:  # Handle any other object by converting to a dict of its attributes
            msg_dict = {k: getattr(msg, k) for k in dir(msg)
                      if not k.startswith('_') and not callable(getattr(msg, k))}
        
        # Remove 'id' if it exists - do this after initial conversion to avoid redundant operations
        msg_dict.pop('id', None)  # More efficient than checking and then deleting
        message_dicts.append(msg_dict)
    
    serialized = orjson.dumps(message_dicts)  # orjson is already fast
    return md5(serialized).hexdigest()


async def generate_stream(messages: List):
    try:
        # Only calculate hash for non-tool messages to avoid unnecessary computation
        has_tool_messages = any(
            (isinstance(msg, dict) and msg.get('role') == "tool") or
            (hasattr(msg, 'role') and msg.role == "tool")
            for msg in messages
        )
        
        if not has_tool_messages:
            query_hash = create_query_hash(messages)
            # Try to use cache first
            cached = get_cached_response(query_hash)
            if cached:
                yield orjson.dumps(cached) + b"\n"
                return
        
        # Use semaphore to limit concurrent requests
        async with request_semaphore:
            # Use the session with your client
            stream = await async_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                stream=True
            )
            
            full_content = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    payload = {
                        "content": full_content
                    }
                    yield orjson.dumps(payload) + b"\n"
            
            # Update cache with full response if it's valuable to cache
            if full_content and not has_tool_messages:
                RESPONSE_CACHE[query_hash] = {"content": full_content}
                
    except Exception as e:
        print(f"Error in generate_stream: {str(e)}")
        yield orjson.dumps({"error": str(e)}) + b"\n"


# Background task to process tool calls and generate final response with parallel execution                


@app.post("/chat")
async def get_data(data: ChatRequest, api_key: str = Security(get_api_key)):
    # Process the request and get messages for streaming
    try:
        messages = await process_request(data, async_client, function_map, request_semaphore, system_message, CHAT_MODEL, MAX_TOKENS, tools_payload)
        
        # Stream the final content
        return StreamingResponse(
            generate_stream(messages),
            media_type="application/json"
        )
    except Exception as e:
        print(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/newsletter")
async def get_newsletter():
    try:
        with open(f"json/newsletter/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []
    return res

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
