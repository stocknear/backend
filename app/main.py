# Standard library imports
import random
import io
import gzip
import re
import os
import secrets
from typing import List, Dict, Set
# Third-party library imports
import numpy as np
import pandas as pd
import orjson
import aiohttp
import redis
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from benzinga import financial_data

# Database related imports
import sqlite3
from contextlib import contextmanager
from pocketbase import PocketBase

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, Security, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse, JSONResponse

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# DB constants & context manager

STOCK_DB = 'stocks'
ETF_DB = 'etf'
CRYPTO_DB = 'crypto'
INSTITUTE_DB = 'institute'

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

  cursor.execute("SELECT symbol, name, type FROM stocks")
  raw_data = cursor.fetchall()
  stock_list_data = [{
    'symbol': row[0],
    'name': row[1],
    'type': row[2].capitalize(),
  } for row in raw_data]
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

#------Start Crypto DB------------#
with db_connection(CRYPTO_DB) as cursor:
  cursor.execute("SELECT DISTINCT symbol FROM cryptos")
  crypto_symbols = [row[0] for row in cursor.fetchall()]

  cursor.execute("SELECT symbol, name, type FROM cryptos")
  raw_data = cursor.fetchall()
  crypto_list_data = [{
    'symbol': row[0],
    'name': row[1],
    'type': row[2].capitalize(),
  } for row in raw_data]
#------End Crypto DB------------#

#------Init Searchbar Data------------#
searchbar_data = stock_list_data + etf_list_data + crypto_list_data

#------Start Institute DB------------#
with db_connection(INSTITUTE_DB) as cursor:
  cursor.execute("SELECT cik FROM institutes")
  cik_list = [row[0] for row in cursor.fetchall()]
#------End Institute DB------------#

### TECH DEBT ###
con = sqlite3.connect('stocks.db')
etf_con = sqlite3.connect('etf.db')
crypto_con = sqlite3.connect('crypto.db')
con_inst = sqlite3.connect('institute.db')

load_dotenv()

pb = PocketBase('http://127.0.0.1:8090')

FMP_API_KEY = os.getenv('FMP_API_KEY')
Benzinga_API_KEY = os.getenv('BENZINGA_API_KEY')
fin = financial_data.Benzinga(Benzinga_API_KEY)


app = FastAPI(docs_url=None, redoc_url=None, openapi_url = None)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

origins = ["http://www.stocknear.com","https://www.stocknear.com","http://stocknear.com","https://stocknear.com","http://localhost:5173","http://localhost:4173"]

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


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"}
    )

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

class MarketNews(BaseModel):
    newsType: str

class OptionsFlowData(BaseModel):
    ticker: str = ''
    start_date: str = ''
    end_date: str = ''
    pagesize: int = Field(default=1000)
    page: int = Field(default=0)

class HistoricalPrice(BaseModel):
    ticker: str
    timePeriod: str

class AnalystId(BaseModel):
    analystId: str

class PoliticianId(BaseModel):
    politicianId: str

class TranscriptData(BaseModel):
    ticker: str
    year: str
    quarter: str

class GetWatchList(BaseModel):
    watchListId: str

class EditWatchList(BaseModel):
    watchListId: str
    title: str

class GetOnePost(BaseModel):
    postId: str

class UserPost(BaseModel):
    postId: str
    userId: str

class CreateWatchList(BaseModel):
    title: str
    user: str
    ticker: str

class UserId(BaseModel):
    userId: str

class GetLeaderBoard(BaseModel):
    startDate: str
    endDate: str


class GetStrategy(BaseModel):
    strategyId: str

class GetCIKData(BaseModel):
    cik: str

class CreateStrategy(BaseModel):
    title: str
    user: str
    rules: str

class SaveStrategy(BaseModel):
    strategyId: str
    rules: list

class GetFeedback(BaseModel):
    user: str
    rating: str
    description: str

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


def clean_financial_data(list1, list2):
    #combine income_statement with income_growth_statement
    combined_list = []
    for item1 in list1:
        for item2 in list2:
            if item1["date"] == item2["date"]:
                combined_item = {**item1, **item2}  # Combine the dictionaries
                combined_list.append(combined_item)
                break
    return combined_list



@app.get("/")
async def hello_world(api_key: str = Security(get_api_key)):
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
    redis_client.expire(cache_key, 60*5)

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


@app.get("/market-movers")
async def get_market_movers(api_key: str = Security(get_api_key)):
    cache_key = f"get-market-movers"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/market-movers/data.json", 'rb') as file:
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



@app.post("/stock-dividend")
async def stock_dividend(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"stock-dividend-{ticker}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    if ticker in etf_symbols:
        table_name = 'etfs'
        column_name = 'etf_dividend'
    else:
        table_name = 'stocks'
        column_name = 'stock_dividend'

    query_template = f"""
    SELECT 
        {column_name}, quote
    FROM 
        {table_name}
    WHERE
        symbol = ?
    """

    df = pd.read_sql_query(query_template, etf_con if table_name == 'etfs' else con, params=(ticker,))
    
    try:
        dividend_data = orjson.loads(df[column_name].iloc[0])
        if column_name == 'stock_dividend':
            res = dividend_data.get('historical', [])
        else:
            res = dividend_data.get('historical', [])
    except:
        res = []

    try:
        quote_data = orjson.loads(df['quote'].iloc[0])[0]
        eps = quote_data.get('eps')
        current_price = quote_data.get('price')
    except:
        eps = None
        current_price = None

    final_res = [res, eps, current_price]
    redis_client.set(cache_key, orjson.dumps(final_res), 3600*3600)  # Set cache expiration time to 1 hour
    return final_res



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

@app.post("/history-employees")
async def history_employees(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"history-employees-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    query_template = """
    SELECT 
        history_employee_count
    FROM 
        stocks 
    WHERE
        symbol = ?
    """

    df = pd.read_sql_query(query_template,con, params=(ticker,))
    try:
        history_employee_count = orjson.loads(df['history_employee_count'].iloc[0])
        res = sorted([entry for entry in history_employee_count if entry["employeeCount"] != 0], key=lambda x: x["filingDate"])
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600) # Set cache expiration time to 1 hour
    return res

@app.post("/stock-income")
async def stock_income(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"stock-income-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    query_template = """
    SELECT 
        income, income_growth
    FROM 
        stocks 
    WHERE
        symbol = ?
    """
    try:
        with db_connection(STOCK_DB) as cursor:
            cursor.execute(query_template, (ticker,))
            result = cursor.fetchone()
            if result:
                income_statement = orjson.loads(result[0])
                income_statement_growth = orjson.loads(result[1])
                res = clean_financial_data(income_statement, income_statement_growth)
            else:
                res = []
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600) # Set cache expiration time to 1 hour
    return res

@app.post("/stock-balance-sheet")
async def stock_balance_sheet(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"stock-balance-sheet-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    query_template = """
    SELECT 
        balance, balance_growth
    FROM 
        stocks 
    WHERE
        symbol = ?
    """
    try:
        with db_connection(STOCK_DB) as cursor:
            cursor.execute(query_template, (ticker,))
            result = cursor.fetchone()
            if result:
                balance_statement = orjson.loads(result[0])
                balance_statement_growth = orjson.loads(result[1])
                res = clean_financial_data(balance_statement, balance_statement_growth)
            else:
                res = []
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)
    return res    

@app.post("/stock-ratios")
async def stock_ratios(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"stock-ratios-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    query_template = """
    SELECT 
        ratios
    FROM 
        stocks 
    WHERE
        symbol = ?
    """

    try:
        df = pd.read_sql_query(query_template,con, params=(ticker,))
        res =  orjson.loads(df['ratios'].iloc[0])
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600) # Set cache expiration time to 1 hour
    return res


@app.post("/stock-cash-flow")
async def stock_cash_flow(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"stock-cash-flow-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    query_template = """
    SELECT 
        cashflow, cashflow_growth
    FROM 
        stocks 
    WHERE
        symbol = ?
    """

    try:
        df = pd.read_sql_query(query_template,con, params=(ticker,))
        cash_flow_statement =  orjson.loads(df['cashflow'].iloc[0])
        cash_flow_statement_growth =  orjson.loads(df['cashflow_growth'].iloc[0])
        res = clean_financial_data(cash_flow_statement,cash_flow_statement_growth)
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600) # Set cache expiration time to 1 hour
    return res



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
        with open(f"json/economic-calendar/calendar.json", 'rb') as file:
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
        with open(f"json/earnings-calendar/calendar.json", 'rb') as file:
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
        with open(f"json/dividends-calendar/calendar.json", 'rb') as file:
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

@app.get("/stock-splits-calendar")
async def stock_splits_calendar(api_key: str = Security(get_api_key)):
    cache_key = f"stock-splits-calendar"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/stock-splits-calendar/calendar.json", 'rb') as file:
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
        with open(f"json/analyst/summary/{ticker}.json", 'rb') as file:
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


@app.post("/get-watchlist")
async def get_watchlist(data: GetWatchList, api_key: str = Security(get_api_key)):
    data = data.dict()
    watchlist_id = data['watchListId']
    result = pb.collection("watchlist").get_one(watchlist_id)
    ticker_list = result.ticker

    query_stocks = """
    SELECT 
        stock_news
    FROM 
        stocks 
    WHERE
        symbol = ?
    """
    query_etf = """
    SELECT 
        etf_news
    FROM 
        etfs 
    WHERE
        symbol = ?
    """

    query_crypto = """
    SELECT 
        crypto_news
    FROM 
        cryptos
    WHERE
        symbol = ?
    """

    combined_results = []  # List to store the combined results
    combined_news = []
    
    for ticker in ticker_list:
        ticker = ticker.upper()
        if ticker in etf_symbols:
            try:
                with open(f"json/quote/{ticker}.json", 'rb') as file:
                    quote_dict = orjson.loads(file.read())
                    quote_dict['type'] = 'etf'
                    combined_results.append(quote_dict)
            except:
                pass
            df = pd.read_sql_query(query_etf, etf_con, params=(ticker,))
            if not df.empty:
                df_dict = df.to_dict()
                try:
                    combined_news.append(orjson.loads(df_dict['etf_news'][0])[0])
                except:
                    pass
        elif ticker in crypto_symbols:
            try:
                with open(f"json/quote/{ticker}.json", 'rb') as file:
                    quote_dict = orjson.loads(file.read())
                    quote_dict['type'] = 'crypto'
                    combined_results.append(quote_dict)
            except:
                pass
            df = pd.read_sql_query(query_crypto, crypto_con, params=(ticker,))
            if not df.empty:
                df_dict = df.to_dict()
                try:
                    combined_news.append(orjson.loads(df_dict['crypto_news'][0])[0])
                except:
                    pass
        else:
            try:
                with open(f"json/quote/{ticker}.json", 'rb') as file:
                    quote_dict = orjson.loads(file.read())
                    quote_dict['type'] = 'stock'
                    combined_results.append(quote_dict)
            except:
                pass
            df = pd.read_sql_query(query_stocks, con, params=(ticker,))
            if not df.empty:
                df_dict = df.to_dict()
                try:
                    combined_news.append(orjson.loads(df_dict['stock_news'][0])[0])
                except:
                    pass
    res = [combined_results, combined_news]
    return res


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

    always_include = ['symbol', 'marketCap', 'price', 'changesPercentage', 'name']

    try:
        with open(f"json/stock-screener/data.json", 'rb') as file:
            data = orjson.loads(file.read())
            filtered_data = [
                {key: item.get(key) for key in set(always_include + rule_of_list) if key in item}
                for item in data
            ]
    except Exception as e:
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


@app.post("/get-quant-stats")
async def get_quant_stats(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()
    cache_key = f"get-quant-stats-{ticker}"
    
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    if ticker in etf_symbols:
        table_name = 'etfs'
        query_con = etf_con
    elif ticker in crypto_symbols:
        table_name = 'cryptos'
        query_con = crypto_con
    else:
        table_name = 'stocks'
        query_con = con
    # If the hash doesn't exist or doesn't match, fetch data from the database
    query_metrics_template = f"""
        SELECT
            quantStats
        FROM
            {table_name}
        WHERE
            symbol = ?
    """

    metrics_data = pd.read_sql_query(query_metrics_template, query_con, params=(ticker,))
    
    try:
        #metrics_data = orjson.loads(metrics_data.to_dict()['quantStats'][0])
        metrics_data = metrics_data.to_dict()['quantStats'][0]
        metrics_data = eval(metrics_data)
    except:
        metrics_data = {}
    # Store the data and hash in the cache
    redis_client.set(cache_key, orjson.dumps(metrics_data))
    redis_client.expire(cache_key, 3600 *24) # Set cache expiration time to 1 hour

    return metrics_data



@app.post("/trading-signals")
async def get_trading_signals(data: TickerData, api_key: str = Security(get_api_key)):

    data = data.dict()
    ticker = data['ticker'].upper()
    cache_key = f"get-trading-signals-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    if ticker in etf_symbols:
        table_name = 'etfs'
    else:
        table_name = 'stocks'
    # If the hash doesn't exist or doesn't match, fetch data from the database
    query = f"""
        SELECT
            tradingSignals
        FROM
            {table_name}
        WHERE
            symbol = ?
    """
    try:
        # Execute the query and read the result into a DataFrame
        query_result = pd.read_sql_query(query, etf_con if table_name == 'etfs' else con, params=(ticker,))
        
        # Convert the DataFrame to a JSON object
        if not query_result.empty:
            res = query_result['tradingSignals'][0]
            res = orjson.loads(res)[0]  # Assuming 'tradingSignals' column contains JSON strings        
            res = replace_nan_inf_with_none(res)
            #res = {'Start': '2021-07-14', 'End': '2023-11-10', 'Return [%]': 59.99997000000007, 'Buy & Hold Return [%]': -90.58823529411765, 'Return (Ann.) [%]': 77.21227166877182, 'Duration': '849', 'Volatility (Ann.) [%]': 34.82411687248909, 'Sharpe Ratio': 2.21720688428338, 'Sortino Ratio': None, 'Calmar Ratio': None, 'Max. Drawdown [%]': -0.0, 'Avg. Drawdown [%]': None, 'Max. Drawdown Duration': None, 'Avg. Drawdown Duration': None, '# Trades': 7, 'Win Rate [%]': 85.71428571428571, 'Best Trade [%]': 10.000000000000009, 'Worst Trade [%]': 0.0, 'Avg. Trade [%]': 6.944880005339327, 'Max. Trade Duration': '189', 'Avg. Trade Duration': '53', 'Profit Factor': None, 'Expectancy [%]': 6.989439132296275, 'SQN': 5.999999999999999, 'nextSignal': 'Hold'}

        else:
            res = []  # Set a default value if query_result is empty
    except Exception as e:
        print("Error fetching data from the database:", e)
        res = []  # Set a default value in case of an error

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24) # Set cache expiration time to Infinity
    return res


@app.post("/fair-price")
async def get_fair_price(data: TickerData, api_key: str = Security(get_api_key)):
    data = data.dict()
    ticker = data['ticker'].upper()
    cache_key = f"get-fair-price-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    
    query_template = """
        SELECT 
            discounted_cash_flow
        FROM 
            stocks 
        WHERE
            symbol = ?
    """
    
    try:
        df = pd.read_sql_query(query_template, con, params=(ticker,))
        dcf_value = float(df['discounted_cash_flow'].iloc[0])
    except:
        dcf_value = None


    redis_client.set(cache_key, orjson.dumps(dcf_value))
    redis_client.expire(cache_key, 3600 * 24) # Set cache expiration time to Infinity
    return dcf_value


@app.post("/congress-trading-ticker")
async def get_fair_price(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"get-congress-trading-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/congress-trading/company/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24) # Set cache expiration time to Infinity

    return res



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
    
    cursor = con_inst.cursor()

    # Execute a SQL query to select the top 10 best performing cik entries by winRate
    cursor.execute("SELECT cik, name, numberOfStocks, performancePercentage3year, performancePercentage5year, performanceSinceInceptionPercentage, averageHoldingPeriod, turnover, marketValue, winRate, holdings, summary FROM institutes WHERE cik = ?", (cik,))
    cik_data = cursor.fetchall()
    res = [{
        'cik': row[0],
        'name': row[1],
        'numberOfStocks': row[2],
        'performancePercentage3year': row[3],
        'performancePercentage5year': row[4],
        'performanceSinceInceptionPercentage': row[5],
        'averageHoldingPeriod': row[6],
        'turnover': row[7],
        'marketValue': row[8],
        'winRate': row[9],
        'holdings': orjson.loads(row[10]),
        'summary': orjson.loads(row[11]),
    } for row in cik_data]

    res_json = orjson.dumps(res[0])
    compressed_data = gzip.compress(res_json)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 3600) # Set cache expiration time to Infinity

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/searchbar-data")
async def get_stock(api_key: str = Security(get_api_key)):
    cache_key = f"searchbar-data"
    cached_result = redis_client.get(cache_key)

    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    # Compress the JSON data
    searchbar_data_json = orjson.dumps(searchbar_data)
    compressed_data = gzip.compress(searchbar_data_json)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

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



@app.post("/crypto-profile")
async def get_crypto_profile(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"crypto-profile-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    query_template = """
        SELECT 
            profile
        FROM 
            cryptos
        WHERE
            symbol = ?
    """
    query = query_template.format(ticker=ticker)
    cur = crypto_con.cursor()
    cur.execute(query, (ticker,))
    result = cur.fetchone()  # Get the first row
    profile_list = []

    try:
        if result is not None:
            profile_list = orjson.loads(result[0])
    except:
        profile_list = []

    redis_client.set(cache_key, orjson.dumps(profile_list))
    redis_client.expire(cache_key, 3600 * 24) # Set cache expiration time to Infinity

    return profile_list


@app.post("/etf-profile")
async def get_fair_price(data: TickerData, api_key: str = Security(get_api_key)):

    data = data.dict()
    ticker = data['ticker'].upper()

    cache_key = f"etf-profile-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

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
    profile_list = []

    try:
        if result is not None:
            profile_list = orjson.loads(result[0])
            for item in profile_list:
                item['etfProvider'] = result[1]
            #Show only hedge funds that are in the institute.db
    except:
        profile_list = []

    redis_client.set(cache_key, orjson.dumps(profile_list))
    redis_client.expire(cache_key, 3600 * 24) # Set cache expiration time to Infinity

    return profile_list


@app.post("/etf-holdings")
async def etf_holdings(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"etf-holdings-{ticker}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)


    query_template = f"SELECT holding from etfs WHERE symbol = ?"
    df = pd.read_sql_query(query_template, etf_con, params=(ticker,))

    try:
        res = orjson.loads(df['holding'].iloc[0])
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res), 3600*3600)  # Set cache expiration time to 1 hour
    return res

@app.post("/etf-country-weighting")
async def etf_holdings(data: TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"etf-country-weighting-{ticker}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)


    query_template = f"SELECT country_weightings from etfs WHERE symbol = ?"
    df = pd.read_sql_query(query_template, etf_con, params=(ticker,))
    try:
        res = orjson.loads(df['country_weightings'].iloc[0])
        for item in res:
            if item["weightPercentage"] != 'NaN%':
                item["weightPercentage"] = float(item["weightPercentage"].rstrip('%'))
            else:
                item["weightPercentage"] = 0

        # Sort the list by weightPercentage in descending order
        res = sorted(res, key=lambda x: x["weightPercentage"], reverse=True)
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res), 3600*3600)  # Set cache expiration time to 1 hour
    return res






@app.get("/ai-signals")
async def top_ai_signals(api_key: str = Security(get_api_key)):
    cache_key = f"ai-signals"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/ai-signals/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

    return res



@app.post("/exchange-constituents")
async def top_ai_signals(data:FilterStockList, api_key: str = Security(get_api_key)):
    data = data.dict()
    filter_list = data['filterList']

    cache_key = f"filter-list-{filter_list}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    if filter_list == 'nasdaqConstituent':
        path = f"nasdaq_constituent.json"
    elif filter_list == 'dowjonesConstituent':
        path = f"dowjones_constituent.json"
    elif filter_list == 'sp500Constituent':
        path = f"sp500_constituent.json"

    try:
        with open(f"json/stocks-list/{path}", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

    return res


@app.get("/all-stock-tickers")
async def get_all_stock_tickers(api_key: str = Security(get_api_key)):
    cache_key = f"all_stock_tickers"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )
    try:
        with open(f"json/all-symbols/stocks.json", 'rb') as file:
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

@app.get("/all-crypto-tickers")
async def get_all_crypto_tickers(api_key: str = Security(get_api_key)):
    cache_key = f"all-crypto-tickers"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/all-symbols/cryptos.json", 'rb') as file:
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
        return orjson.loads(cached_result)
    try:
        with open(f"json/congress-trading/rss-feed/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60*60)  # Set cache expiration time to 1 day
    return res

@app.get("/analysts-price-targets-rss-feed")
async def get_analysts_price_targets_rss_feed(api_key: str = Security(get_api_key)):
    cache_key = f"analysts-price-targets-rss-feed"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/analysts-rss-feed/price-targets.json", 'rb') as file:
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


@app.get("/analysts-upgrades-downgrades-rss-feed")
async def get_analysts_upgrades_downgrades_rss_feed(api_key: str = Security(get_api_key)):
    cache_key = f"analysts-upgrades-downgrades-rss-feed"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/analysts-rss-feed/upgrades-downgrades.json", 'rb') as file:
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

@app.get("/delisted-companies")
async def get_delisted_companies(api_key: str = Security(get_api_key)):

    cache_key = f"delisted-companies"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/delisted-companies/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return res


@app.post("/filter-stock-list")
async def filter_stock_list(data:FilterStockList, api_key: str = Security(get_api_key)):
    data = data.dict()
    filter_list = data['filterList']
    cache_key = f"filter-list-{filter_list}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return orjson.loads(cached_result)

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")

    base_query = """
        SELECT symbol, name, price, changesPercentage, marketCap, revenue, netIncome
        FROM stocks 
        WHERE symbol != ? 
        AND (price IS NOT NULL OR changesPercentage IS NOT NULL) 
        AND {}
    """

    conditions = {
        'megaCap': "marketCap >= 200e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ')",
        'largeCap': "marketCap < 200e9 AND marketCap >= 10e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ')",
        'midCap': "marketCap < 10e9 AND marketCap >= 2e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ')",
        'smallCap': "marketCap < 2e9 AND marketCap >= 300e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ')",
        'microCap': "marketCap < 300e6 AND marketCap >= 50e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ')",
        'nanoCap': "marketCap < 50e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ')",
        'nasdaq': "exchangeShortName = 'NASDAQ'",
        'nyse': "exchangeShortName = 'NYSE'",
        'xetra': "exchangeShortName = 'XETRA'",
        'amex': "exchangeShortName = 'AMEX'",
        'DE': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND country = 'DE'",
        'CA': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND country = 'CA'",
        'CN': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND country = 'CN'",
        'IN': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND country = 'IN'",
        'IL': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND country = 'IL'",
        'UK': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND country = 'UK'",
        'JP': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND country = 'JP'",
        'financial': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND (sector = 'Financials' OR sector = 'Financial Services')",
        'healthcare': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND (sector = 'Healthcare')",
        'technology': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND (sector = 'Technology')",
        'industrials': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND (sector = 'Industrials')",
        'consumer-cyclical': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND (sector = 'Consumer Cyclical')",
        'real-estate': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND (sector = 'Real Estate')",
        'basic-materials': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND (sector = 'Basic Materials')",
        'communication-services': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND (sector = 'Communication Services')",
        'energy': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND (sector = 'Energy')",
        'consumer-defensive': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND (sector = 'Consumer Defensive')",
        'utilities': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ') AND (sector = 'Utilities')",
    }

    # Execute the query with the relevant country
    if filter_list in conditions:
        full_query = base_query.format(conditions[filter_list])
        cursor.execute(full_query, ('%5EGSPC',))

    # Fetch the results
    raw_data = cursor.fetchall()

    res_list = [{
            'symbol': symbol,
            'name': name,
            'price': price,
            'changesPercentage': changesPercentage,
            'marketCap': marketCap,
            'revenue': revenue,
            'netIncome': netIncome
        } for (symbol, name, price, changesPercentage, marketCap, revenue, netIncome) in raw_data]

    sorted_res_list = sorted(res_list, key=lambda x: x['marketCap'], reverse=True)

    # Cache the result
    redis_client.set(cache_key, orjson.dumps(sorted_res_list))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

    return sorted_res_list


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
        return orjson.loads(cached_result)
    
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

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return res

@app.get("/ticker-mentioning")
async def get_ticker_mentioning(api_key: str = Security(get_api_key)):

    cache_key = f"get-ticker-mentioning"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/ticker-mentioning/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return res


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
async def etf_provider(data: ETFProviderData, api_key: str = Security(get_api_key)):
    data = data.dict()
    etf_provider = data['etfProvider'].lower()

    cache_key = f"etf-provider-{etf_provider}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return orjson.loads(cached_result)

    # Check if data is cached; if not, fetch and cache it
    cursor = etf_con.cursor()
    query = "SELECT symbol, name, expenseRatio, totalAssets, numberOfHoldings FROM etfs WHERE etfProvider = ?"
    cursor.execute(query, (etf_provider,))
    raw_data = cursor.fetchall()
    cursor.close()

    # Extract only relevant data and sort it
    # Extract only relevant data and filter only integer totalAssets
    res = [
        {'symbol': row[0], 'name': row[1], 'expenseRatio': row[2], 'totalAssets': row[3], 'numberOfHoldings': row[4]}
        for row in raw_data if isinstance(row[3], int)
    ]
    sorted_res = sorted(res, key=lambda x: x['totalAssets'], reverse=True)
    redis_client.set(cache_key, orjson.dumps(sorted_res))
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day
    return sorted_res

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


@app.get("/magnificent-seven")
async def get_magnificent_seven(api_key: str = Security(get_api_key)):
    cache_key = f"all_magnificent_seven"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )
    try:
        with open(f"json/magnificent-seven/data.json", 'rb') as file:
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
    redis_client.expire(cache_key, 3600 * 24)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/trending")
async def get_trending(api_key: str = Security(get_api_key)):
    cache_key = f"get-trending"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/trending/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*15)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/heatmaps")
async def get_trending(data: HeatMapData, api_key: str = Security(get_api_key)):
    index = data.index
    cache_key = f"get-heatmaps-{index}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})

    try:
        with open(f"json/heatmaps/{index}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*5)  # Set cache expiration time to 5 min

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
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

@app.post("/bull-bear-say")
async def get_bull_bear_say(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"bull-bear-say-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/bull_bear_say/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res


@app.post("/options-net-flow-ticker")
async def get_options_net_flow(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"options-net-flow-ticker-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/options-net-flow/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, caching_time)
    return res

@app.post("/options-plot-ticker")
async def get_options_plot_ticker(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"options-plot-ticker-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/options-flow/company/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60)  # Set cache expiration time to 1 day
    return res


#api endpoint not for website but for user
@app.post("/raw-options-flow-ticker")
@limiter.limit("500/minute")
async def get_raw_options_flow_ticker(data:OptionsFlowData, request: Request, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    start_date = data.start_date
    end_date = data.end_date
    pagesize = data.pagesize
    page = data.page
    cache_key = f"raw-options-flow-{ticker}-{start_date}-{end_date}-{pagesize}-{page}"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    try:
        data = fin.options_activity(company_tickers=ticker, date_from=start_date, date_to = end_date, page=page, pagesize=pagesize)
        data = orjson.loads(fin.output(data))['option_activity']
    except:
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


'''
@app.post("/options-flow-feed")
async def get_options_flow_feed(data: LastOptionId, api_key: str = Security(get_api_key)):
    last_option_id = data.lastId

    try:
        with open(f"json/options-flow/feed/data.json", 'rb') as file:
            all_data = orjson.loads(file.read())

        if len(last_option_id) == 0:
            res_list = all_data[0:100]
        else:
            # Find the index of the element with the last known ID
            start_index = next((i for i, item in enumerate(all_data) if item["id"] == last_option_id), -1)
            if start_index == -1:
                raise ValueError("Last known ID not found in data")

            # Get the next 100 elements
            res_list = all_data[start_index + 1:start_index + 101]

        # Compress the data
        compressed_data = gzip.compress(orjson.dumps(res_list))

        return StreamingResponse(
            io.BytesIO(compressed_data),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    except Exception as e:
        # Log the error for debugging
        print(f"Error: {str(e)}")
        return StreamingResponse(
            io.BytesIO(gzip.compress(orjson.dumps([]))),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
'''
@app.get("/options-flow-feed")
async def get_options_flow_feed(api_key: str = Security(get_api_key)):
    try:
        with open(f"json/options-flow/feed/data.json", 'rb') as file:
            res_list = orjson.loads(file.read())
    except:
        res_list = []
    data = orjson.dumps(res_list)
    compressed_data = gzip.compress(data)
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.get("/options-zero-dte")
async def get_options_flow_feed(api_key: str = Security(get_api_key)):
    try:
        with open(f"json/options-flow/zero-dte/data.json", 'rb') as file:
            res_list = orjson.loads(file.read())
    except:
        res_list = []
    data = orjson.dumps(res_list)
    compressed_data = gzip.compress(data)
    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/options-bubble")
async def get_options_bubble(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"options-bubble-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/options-bubble/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*24)  # Set cache expiration time to 1 day
    return res


@app.get("/top-analysts")
async def get_all_analysts(api_key: str = Security(get_api_key)):
    cache_key = f"top-analysts"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/analyst/top-analysts.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60*60*2)  # Set cache expiration time to 1 day
    return res

@app.get("/top-analysts-stocks")
async def get_all_analysts(api_key: str = Security(get_api_key)):
    cache_key = f"top-analysts-stocks"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/analyst/top-stocks.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60*60*2)  # Set cache expiration time to 1 day
    return res

@app.post("/analyst-stats")
async def get_all_analysts(data:AnalystId, api_key: str = Security(get_api_key)):
    analyst_id = data.analystId

    cache_key = f"analyst-stats-{analyst_id}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/analyst/analyst-db/{analyst_id}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60*60*2)  # Set cache expiration time to 1 day
    return res

@app.post("/wiim")
async def get_wiim(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()

    cache_key = f"wiim-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)

    try:
        with open(f"json/wiim/company/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())[:5]
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60*60*2)
    return res

@app.get("/dashboard-info")
async def get_dashboard_info(api_key: str = Security(get_api_key)):

    cache_key = f"dashboard-info"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/dashboard/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 60*5)
    return res

@app.post("/sentiment-analysis")
async def get_sentiment_analysis(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"sentiment-analysis-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/sentiment-analysis/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res

@app.post("/trend-analysis")
async def get_trend_analysis(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"trend-analysis-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/trend-analysis/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res

@app.post("/price-analysis")
async def get_price_analysis(data:TickerData, api_key: str = Security(get_api_key)):
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
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )



@app.post("/fundamental-predictor-analysis")
async def get_fundamental_predictor_analysis(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"fundamental-predictor-analysis-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/fundamental-predictor-analysis/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res


@app.post("/value-at-risk")
async def get_trend_analysis(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"value-at-risk-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/var/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res

@app.post("/government-contract")
async def get_government_contract(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"government-contract-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/government-contract/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res

@app.post("/corporate-lobbying")
async def get_lobbying(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"corporate-lobbying-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/corporate-lobbying/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res

@app.post("/enterprise-values")
async def get_enterprise_values(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"enterprise-values-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/enterprise-values/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res


@app.post("/share-statistics")
async def get_enterprise_values(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"share-statistics-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/share-statistics/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res


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



@app.get("/most-shorted-stocks")
async def get_most_shorted_stocks(api_key: str = Security(get_api_key)):
    cache_key = f"most-shorted-stocks"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/most-shorted-stocks/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res

@app.get("/most-retail-volume")
async def get_most_retail_volume(api_key: str = Security(get_api_key)):
    cache_key = f"most-retail-volume"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/retail-volume/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res


@app.post("/retail-volume")
async def get_retail_volume(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"retail-volume-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/retail-volume/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res

@app.post("/dark-pool")
async def get_dark_pool(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"dark-pool-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/dark-pool/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res


@app.get("/dark-pool-flow")
async def get_dark_pool_flow(api_key: str = Security(get_api_key)):
    cache_key = f"dark-flow-flow"

    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
        io.BytesIO(cached_result),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"})
    try:
        with open(f"json/dark-pool/flow/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)
    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*15)  # Set cache expiration time to 15 min

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )


@app.post("/market-maker")
async def get_market_maker(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"market-maker-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/market-maker/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res

@app.post("/clinical-trial")
async def get_clinical_trial(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"clinical-trial-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/clinical-trial/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day

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
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res


@app.post("/fail-to-deliver")
async def get_fail_to_deliver(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"fail-to-deliver-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/fail-to-deliver/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res

@app.post("/borrowed-share")
async def get_borrowed_share(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"borrowed-share-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/borrowed-share/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res


@app.post("/analyst-insight")
async def get_analyst_insight(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"analyst-insight-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return orjson.loads(cached_result)
    try:
        with open(f"json/analyst/insight/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = {}

    redis_client.set(cache_key, orjson.dumps(res))
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day
    return res


@app.post("/implied-volatility")
async def get_clinical_trial(data:TickerData, api_key: str = Security(get_api_key)):
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
        with open(f"json/implied-volatility/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*3600)  # Set cache expiration time to 1 day

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.post("/swap-ticker")
async def get_swap_data(data:TickerData, api_key: str = Security(get_api_key)):
    ticker = data.ticker.upper()
    cache_key = f"swap-{ticker}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/swap/companies/{ticker}.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*3600)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/cramer-tracker")
async def get_cramer_tracker(api_key: str = Security(get_api_key)):
    cache_key = f"cramer-tracker"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    try:
        with open(f"json/cramer-tracker/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 3600*3600)

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
            trending = orjson.loads(file.read())[0:5]
    except:
        trending = []

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

@app.get("/dividend-kings")
async def get_dividend_kings():
    cache_key = f"dividend-kings"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/stocks-list/dividend-kings.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []

    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*20)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )

@app.get("/dividend-aristocrats")
async def get_dividend_kings():
    cache_key = f"dividend-aristocrats"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return StreamingResponse(
            io.BytesIO(cached_result),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )
    try:
        with open(f"json/stocks-list/dividend-aristocrats.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []
    data = orjson.dumps(res)
    compressed_data = gzip.compress(data)

    redis_client.set(cache_key, compressed_data)
    redis_client.expire(cache_key, 60*20)

    return StreamingResponse(
        io.BytesIO(compressed_data),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )
    return res

@app.get("/newsletter")
async def get_newsletter():
    try:
        with open(f"json/newsletter/data.json", 'rb') as file:
            res = orjson.loads(file.read())
    except:
        res = []
    return res