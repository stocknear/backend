import pandas as pd
import time
import ujson
import aiohttp
import asyncio


query_template = """
    SELECT 
        profile, quote,
        esg_ratings,esg_data,stock_split
    FROM 
        stocks 
    WHERE
        symbol = ?
"""

class FinancialModelingPrep:
    def __init__(self, ticker, con):
        #self.url = url
        self.ticker = ticker
        self.con = con


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

    async def company_info(self):
        df = pd.read_sql_query(query_template, self.con, params=(self.ticker,))
        
        #con.close()
        if df.empty:
            final_res =[{}]
            return final_res
        else:
            data= df.to_dict(orient='records')
            data =data[0]

            company_profile =  ujson.loads(data['profile'])
            company_quote =  ujson.loads(data['quote'])
            company_tier_list =  data['rating']

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
                    'earning': company_quote[0]['earningsAnnouncement'],
                    'previousClose': company_quote[0]['price'], #This is true because I update my db before the market opens hence the price will be the previousClose price.
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
