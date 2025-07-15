import orjson
import asyncio
import sqlite3
import pandas as pd
from tqdm import tqdm
from datetime import datetime

async def save_stockdeck(symbol, data):
    with open(f"json/stockdeck/{symbol}.json", 'w') as file:
        file.write(orjson.dumps(data).decode('utf-8'))

query_template = """
    SELECT 
        profile,
        stock_split
    FROM 
        stocks 
    WHERE
        symbol = ?
"""

with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

screener_columns = ['forwardPE','revenueTTM',"netIncomeTTM","profitPerEmployee","revenuePerEmployee"]
#profit and revenue per employee needed for profile/employee page

async def get_data(ticker):
    try: 
        df = pd.read_sql_query(query_template, con, params=(ticker,))
        if df.empty:
            res_list =[{}]
            return res_list
        else:
            data= df.to_dict(orient='records')
            data =data[0]

            company_profile =  orjson.loads(data['profile'])
            try:
                with open(f"json/quote/{ticker}.json", 'r') as file:
                    company_quote = orjson.loads(file.read())
            except:
                company_quote = {}
            
            try:
                with open(f"json/financial-statements/income-statement/annual/{ticker}.json", 'r') as file:
                    history = orjson.loads(file.read())
                    history = sorted(history, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"), reverse=True)
                    history = history[:5]
                    history = [{"date": item["date"], "revenue": item["revenue"], "netIncome": item["netIncome"]} for item in history]
                    history = sorted(history, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"), reverse=False)
                    change_percentage_revenue = round((history[-1]['revenue']/history[-2]['revenue']-1)*100,2)
                    change_percentage_net_income = round((history[-1]['netIncome']/history[-2]['netIncome']-1)*100,2)
                    financial_dict = {'changePercentageRevenue': change_percentage_revenue, 'changePercentageNetIncome': change_percentage_net_income, 'history': history}
            except:
                financial_dict = {}
            

            try:
                screener_result = {column: stock_screener_data_dict.get(ticker, {}).get(column, None) for column in screener_columns}
            except:
                screener_result = {column: None for column in screener_columns}

        
            if data['stock_split'] == None:
                company_stock_split = []
            else:
                company_stock_split = orjson.loads(data['stock_split'])
            

            res_list = {
                    **screener_result,
                    "ipoDate": company_profile[0]['ipoDate'],
                    'companyName': company_profile[0]['companyName'],
                    'industry': company_profile[0]['industry'],
                    'sector': company_profile[0]['sector'],
                    'beta': company_profile[0]['beta'],
                    'marketCap': company_profile[0]['mktCap'],
                    'avgVolume': company_profile[0]['volAvg'],
                    'country': company_profile[0]['country'],
                    'exchange': company_profile[0]['exchangeShortName'],
                    #'earning': company_quote['earningsAnnouncement'],
                    'pe': company_quote['pe'],
                    'eps': company_quote['eps'],
                    'sharesOutstanding': company_quote['sharesOutstanding'],
                    'previousClose': company_quote['price'], #This is true because I update my db before the market opens hence the price will be the previousClose price.
                    'website': company_profile[0]['website'],
                    'description': company_profile[0]['description'],
                    'fullTimeEmployees': company_profile[0]['fullTimeEmployees'],
                    'financialPerformance': financial_dict,
                    'stockSplits': company_stock_split,
                }

            return res_list
    except Exception as e:
        print(e)
        res_list ={}
        return res_list

async def run():
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stocks_symbols = [row[0] for row in cursor.fetchall()]
    for ticker in tqdm(stocks_symbols):
        res = await get_data(ticker)  
        await save_stockdeck(ticker, res)

try:
    con = sqlite3.connect('stocks.db')
    asyncio.run(run())
    con.close()
except Exception as e:
    print(e)