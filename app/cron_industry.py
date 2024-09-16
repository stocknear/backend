import aiohttp
import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
import orjson
from collections import defaultdict

with open(f"json/stock-screener/data.json", 'rb') as file:
        stock_screener_data = orjson.loads(file.read())

# Convert stock_screener_data into a dictionary keyed by symbol
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}


def save_as_json(data):
    with open(f"json/industry/overview.json", 'w') as file:
        ujson.dump(data, file)


#async def get_data():


def run():
    # Initialize a dictionary to store stock count, market cap, and other totals for each industry
    sector_industry_data = defaultdict(lambda: defaultdict(lambda: {
        'numStocks': 0, 
        'totalMarketCap': 0.0, 
        'totalPE': 0.0, 
        'totalDividendYield': 0.0, 
        'totalNetIncome': 0.0,
        'totalRevenue': 0.0,
        'totalChange1M': 0.0, 
        'totalChange1Y': 0.0,
        'peCount': 0, 
        'dividendCount': 0, 
        'change1MCount': 0, 
        'change1YCount': 0
    }))

    # Iterate through stock_screener_data to accumulate values
    for stock in stock_screener_data:
        sector = stock.get('sector')
        industry = stock.get('industry')
        market_cap = stock.get('marketCap')
        pe = stock.get('pe')
        dividend_yield = stock.get('dividendYield')
        net_income = stock.get('netIncome')
        revenue = stock.get('revenue')
        change_1_month = stock.get('change1M')
        change_1_year = stock.get('change1Y')
        
        # Ensure both sector and industry are valid and that market cap is a valid number
        if sector and industry and market_cap is not None:
            # Update stock count and accumulate market cap
            sector_industry_data[sector][industry]['numStocks'] += 1
            sector_industry_data[sector][industry]['totalMarketCap'] += float(market_cap)
            
            # Accumulate PE ratio if available
            if pe is not None:
                sector_industry_data[sector][industry]['totalPE'] += float(pe)
                sector_industry_data[sector][industry]['peCount'] += 1
            
            # Accumulate dividend yield if available
            if dividend_yield is not None:
                sector_industry_data[sector][industry]['totalDividendYield'] += float(dividend_yield)
                sector_industry_data[sector][industry]['dividendCount'] += 1
            
            # Accumulate net income and revenue for profit margin calculation
            if net_income is not None and revenue is not None:
                sector_industry_data[sector][industry]['totalNetIncome'] += float(net_income)
                sector_industry_data[sector][industry]['totalRevenue'] += float(revenue)
            
            # Accumulate 1-month change if available
            if change_1_month is not None:
                sector_industry_data[sector][industry]['totalChange1M'] += float(change_1_month)
                sector_industry_data[sector][industry]['change1MCount'] += 1
            
            # Accumulate 1-year change if available
            if change_1_year is not None:
                sector_industry_data[sector][industry]['totalChange1Y'] += float(change_1_year)
                sector_industry_data[sector][industry]['change1YCount'] += 1

    # Prepare the final data in the requested format
    result = {}

    for sector, industries in sector_industry_data.items():
        # Sort industries by stock count in descending order
        sorted_industries = sorted(industries.items(), key=lambda x: x[1]['numStocks'], reverse=True)
        
        # Add sorted industries with averages to the result for each sector
        result[sector] = [
            {
                'industry': industry,
                'numStocks': data['numStocks'],
                'totalMarketCap': data['totalMarketCap'],
                'pe': round((data['totalMarketCap'] / data['totalNetIncome']),2) if data['totalNetIncome'] > 0 else None,
                'avgDividendYield': round((data['totalDividendYield'] / data['dividendCount']),2) if data['dividendCount'] > 0 else None,
                'profitMargin': round((data['totalNetIncome'] / data['totalRevenue'])*100,2) if data['totalRevenue'] > 0 else None,
                'avgChange1M': round((data['totalChange1M'] / data['change1MCount']),2) if data['change1MCount'] > 0 else None,
                'avgChange1Y': round((data['totalChange1Y'] / data['change1YCount']),2) if data['change1YCount'] > 0 else None
            } for industry, data in sorted_industries
        ]

    print(result)

    save_as_json(result)


run()