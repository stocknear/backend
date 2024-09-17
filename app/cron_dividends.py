import aiohttp
import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import pytz
import orjson
import os
from dotenv import load_dotenv

headers = {"accept": "application/json"}
url = "https://api.benzinga.com/api/v2.1/calendar/dividends"
load_dotenv()
api_key = os.getenv('BENZINGA_API_KEY')

ny_tz = pytz.timezone('America/New_York')
today = datetime.now(ny_tz).replace(hour=0, minute=0, second=0, microsecond=0)
N_days_ago = today - timedelta(days=10)


async def save_as_json(symbol, data,file_name):
    with open(f"{file_name}/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

async def get_data(ticker, con, etf_con, stock_symbols, etf_symbols):
    try:
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
    
        dividend_data = orjson.loads(df[column_name].iloc[0])
        
        res = dividend_data.get('historical', [])

        filtered_res = [item for item in res if item['recordDate'] != '' and item['paymentDate'] != '']

        # Calculate payout frequency based on dividends recorded in 2023
        payout_frequency = sum(1 for item in filtered_res if '2023' in item['recordDate'])
        quote_data = orjson.loads(df['quote'].iloc[0])[0]
        eps = quote_data.get('eps')
        current_price = quote_data.get('price')

        amount = filtered_res[0]['adjDividend'] if filtered_res else 0
        annual_dividend = round(amount * payout_frequency, 2)
        dividend_yield = round((annual_dividend / current_price) * 100, 2) if current_price else None

        payout_ratio = round((1 - (eps - annual_dividend) / eps) * 100, 2) if eps else None

        previous_index = next((i for i, item in enumerate(filtered_res) if '2023' in item['recordDate']), None)

        # Calculate previousAnnualDividend and dividendGrowth
        previous_annual_dividend = (filtered_res[previous_index]['adjDividend'] * payout_frequency) if previous_index is not None else 0
        dividend_growth = round(((annual_dividend - previous_annual_dividend) / previous_annual_dividend) * 100, 2) if previous_annual_dividend else None


        return {
            'payoutFrequency': payout_frequency,
            'annualDividend': annual_dividend,
            'dividendYield': dividend_yield,
            'payoutRatio': payout_ratio,
            'dividendGrowth': dividend_growth,
            'history': filtered_res,
        }
    
    except:
        res = {}

    return res



async def get_dividends_announcement(session, ticker, stock_symbols):
    querystring = {"token": api_key, "parameters[tickers]": ticker}
    ny_tz = pytz.timezone('America/New_York')
    today = ny_tz.localize(datetime.now())
    N_days_ago = today - timedelta(days=30)  # Example, adjust as needed

    try:
        async with session.get(url, params=querystring, headers=headers) as response:
            if response.status == 200:
                data = ujson.loads(await response.text())['dividends']
                recent_dates = [item for item in data if N_days_ago <= ny_tz.localize(datetime.strptime(item["date"], "%Y-%m-%d")) <= today]
                if recent_dates:
                    nearest_recent = min(recent_dates, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
                    try:
                        symbol = nearest_recent['ticker']
                        dividend = float(nearest_recent['dividend']) if nearest_recent['dividend'] != '' else 0
                        dividend_prior = float(nearest_recent['dividend_prior']) if nearest_recent['dividend_prior'] != '' else 0
                        dividend_yield = round(float(nearest_recent['dividend_yield']) * 100, 2) if nearest_recent['dividend_yield'] != '' else 0
                        ex_dividend_date = nearest_recent['ex_dividend_date'] if nearest_recent['ex_dividend_date'] != '' else 0
                        payable_date = nearest_recent['payable_date'] if nearest_recent['payable_date'] != '' else 0
                        record_date = nearest_recent['record_date'] if nearest_recent['record_date'] != '' else 0
                        if symbol in stock_symbols and dividend != 0 and payable_date != 0 and dividend_prior != 0 and ex_dividend_date != 0 and record_date != 0 and dividend_yield != 0:
                            res_dict = {
                                'symbol': symbol,
                                'date': nearest_recent['date'],
                                'dividend': dividend,
                                'dividendPrior': dividend_prior,
                                'dividendYield': dividend_yield,
                                'exDividendDate': ex_dividend_date,
                                'payableDate': payable_date,
                                'recordDate': record_date,
                            }
                            await save_as_json(symbol, res_dict,'json/dividends/announcement')
                    except Exception as e:
                        # Log or handle the exception
                        print(e)
    except:
        pass


async def run():

    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    total_symbols = stock_symbols + etf_symbols
    
    for ticker in tqdm(total_symbols):
        res = await get_data(ticker, con, etf_con, stock_symbols, etf_symbols)
        try:
            if len(res.get('history')) > 0 and res.get('dividendGrowth') != None:
                await save_as_json(ticker, res, 'json/dividends/companies')
        except:
            pass
    

    con.close()
    etf_con.close()

    delete_files_in_directory("json/dividends/announcement")

    async with aiohttp.ClientSession() as session:
        tasks = [get_dividends_announcement(session, symbol, stock_symbols) for symbol in stock_symbols]
        for f in tqdm(asyncio.as_completed(tasks), total=len(stock_symbols)):
            await f


try:
    asyncio.run(run())
except Exception as e:
    print(e)
