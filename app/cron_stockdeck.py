import ujson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
from rating import rating_model
import pandas as pd
from tqdm import tqdm

async def save_stockdeck(symbol, data):
    with open(f"json/stockdeck/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

def clean_financial_data(self, list1, list2):
    #combine income_statement with income_growth_statement
    combined_list = []
    for item1 in list1:
        for item2 in list2:
            if item1["date"] == item2["date"]:
                combined_item = {**item1, **item2}  # Combine the dictionaries
                combined_list.append(combined_item)
                break
    return combined_list

query_template = """
    SELECT 
        profile, quote,
        esg_ratings,esg_data,stock_split
    FROM 
        stocks 
    WHERE
        symbol = ?
"""

async def get_data(ticker):
    try: 
        df = pd.read_sql_query(query_template, con, params=(ticker,))
        if df.empty:
            final_res =[{}]
            return final_res
        else:
            data= df.to_dict(orient='records')
            data =data[0]

            company_profile =  ujson.loads(data['profile'])
            #company_quote =  ujson.loads(data['quote'])
            try:
                with open(f"json/quote/{ticker}.json", 'r') as file:
                    company_quote = ujson.load(file)
            except:
                company_quote = {}

            if data['esg_data'] == None:
                company_esg_score = {'ESGScore': 'n/a', 'socialScore': 'n/a', 'environmentalScore': 'n/a', 'governanceScore': 'n/a'}
                company_esg_rating = {'ESGRiskRating': 'n/a', 'industry': 'n/a'}
            else:
                company_esg_score =  ujson.loads(data['esg_data'])
            if data['esg_ratings'] == None:
                company_esg_rating = {'ESGRiskRating': 'n/a', 'industry': 'n/a'}
            else:
                company_esg_rating =  ujson.loads(data['esg_ratings'])

            if data['stock_split'] == None:
                company_stock_split = []
            else:
                company_stock_split = ujson.loads(data['stock_split'])

            res_profile = [
                {
                    'ceoName': company_profile[0]['ceo'],
                    'companyName': company_profile[0]['companyName'],
                    'industry': company_profile[0]['industry'],
                    'image': company_profile[0]['image'],
                    'sector': company_profile[0]['sector'],
                    'beta': company_profile[0]['beta'],
                    'marketCap': company_profile[0]['mktCap'],
                    'avgVolume': company_profile[0]['volAvg'],
                    'country': company_profile[0]['country'],
                    'exchange': company_profile[0]['exchangeShortName'],
                    'earning': company_quote['earningsAnnouncement'],
                    'previousClose': company_quote['price'], #This is true because I update my db before the market opens hence the price will be the previousClose price.
                    'website': company_profile[0]['website'],
                    'description': company_profile[0]['description'],
                    'esgScore': company_esg_score['ESGScore'],
                    'socialScore': company_esg_score['socialScore'],
                    'environmentalScore': company_esg_score['environmentalScore'],
                    'governanceScore': company_esg_score['governanceScore'],
                    'esgRiskRating': company_esg_rating['ESGRiskRating'],
                    'fullTimeEmployees': company_profile[0]['fullTimeEmployees'],
                    'stockSplits': company_stock_split,
                }
            ]


            if data['esg_data'] == None:
                company_esg_score = {'ESGScore': 'n/a', 'socialScore': 'n/a', 'environmentalScore': 'n/a', 'governanceScore': 'n/a'}
                company_esg_rating = {'ESGRiskRating': 'n/a', 'industry': 'n/a'}
            else:
                company_esg_score =  ujson.loads(data['esg_data'])
            if data['esg_ratings'] == None:
                company_esg_rating = {'ESGRiskRating': 'n/a', 'industry': 'n/a'}
            else:
                company_esg_rating =  ujson.loads(data['esg_ratings'])


            final_res = {k: v for d in [res_profile] for dict in d for k, v in dict.items()}
            
            return final_res
    except Exception as e:
        print(e)
        final_res =[{}]
        return final_res

async def run():
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stocks_symbols = [row[0] for row in cursor.fetchall()]
    for ticker in stocks_symbols:
        res = await get_data(ticker)  
        await save_stockdeck(ticker, [res])

try:
    con = sqlite3.connect('stocks.db')
    asyncio.run(run())
    con.close()
except Exception as e:
    print(e)