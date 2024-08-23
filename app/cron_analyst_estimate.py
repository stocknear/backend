import aiohttp
import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


query_template = """
    SELECT 
        analyst_estimates, income
    FROM 
        stocks
    WHERE
        symbol = ?
"""

async def save_as_json(symbol, data):
    with open(f"json/analyst-estimate/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


async def get_data(ticker, con):
    try:
        data = pd.read_sql_query(query_template, con, params=(ticker,))
        analyst_estimates = ujson.loads(data['analyst_estimates'].iloc[0])
        income = ujson.loads(data['income'].iloc[0])
        combined_data = defaultdict(dict)

        #Sum up quarter results
        eps_sums = {}
        revenue_sums = {}
        ebitda_sums = {}
        net_income_sums = {}

        # Iterate over the income data
        for item_income in income:
            calendar_year = int(item_income['calendarYear'])
            if calendar_year >= 2015:
                if calendar_year not in eps_sums:
                    eps_sums[calendar_year] = 0  # Initialize the annual sum to 0
                if calendar_year not in revenue_sums:
                    revenue_sums[calendar_year] = 0  # Initialize the annual sum to 0
                if calendar_year not in ebitda_sums:
                    ebitda_sums[calendar_year] = 0  # Initialize the annual sum to 0
                if calendar_year not in net_income_sums:
                    net_income_sums[calendar_year] = 0  # Initialize the annual sum to 0

                eps_sums[calendar_year] += item_income.get('eps', 0)  # Add the EPS value to the annual sum
                revenue_sums[calendar_year] += item_income.get('revenue', 0)  # Add the Revenue value to the annual sum
                ebitda_sums[calendar_year] += item_income.get('ebitda', 0)  # Add the EBITDA value to the annual sum
                net_income_sums[calendar_year] += item_income.get('netIncome', 0)  # Add the Net Income value to the annual sum

        for item_estimate in analyst_estimates:
            for item_income in income:
                year = item_estimate['date'][:4]
                if year == item_income['calendarYear']:
                    try:
                        revenue = revenue_sums[int(year)]
                    except:
                        revenue = None
                    try:
                        net_income = net_income_sums[int(year)]
                    except:
                        net_income = None
                    try:
                        ebitda = ebitda_sums[int(year)]
                    except:
                        ebitda = None
                    try:
                        eps = round(eps_sums[int(year)],2)
                    except:
                        eps = None
                    try:
                        estimated_ebitda_avg = item_estimate['estimatedEbitdaAvg']
                    except:
                        estimated_ebitda_avg = None
                    try:
                        estimated_net_income_avg = item_estimate['estimatedNetIncomeAvg']
                    except:
                        estimated_net_income_avg = None
                    try:
                        estimated_revenue_avg = item_estimate['estimatedRevenueAvg']
                    except:
                        estimated_revenue_avg = None
                    try:
                        estimated_eps_avg = round(item_estimate['estimatedEpsAvg'],2)
                    except:
                        estimated_eps_avg = None

                    try:
                        numOfAnalysts = int((item_estimate['numberAnalystEstimatedRevenue']+item_estimate['numberAnalystsEstimatedEps'])/2)
                    except:
                        numOfAnalysts = None

                    combined_data[year].update({
                    'symbol': item_estimate['symbol'],
                    'date': int(item_estimate['date'][:4]),
                    'estimatedRevenueAvg': estimated_revenue_avg,
                    'estimatedEbitdaAvg': estimated_ebitda_avg,
                    'estimatedNetIncomeAvg': estimated_net_income_avg,
                    'estimatedEpsAvg': estimated_eps_avg,
                    'revenue': revenue,
                    'netIncome': net_income,
                    'ebitda': ebitda,
                    'eps': eps,
                    'numOfAnalysts': numOfAnalysts,
                })

        for item_estimate in analyst_estimates:
            year = item_estimate['date'][:4]
            if year == income[0]['calendarYear']:
                break
            else:
                try:
                    estimated_revenue_avg = item_estimate['estimatedRevenueAvg']
                except:
                    estimated_revenue_avg = None
                try:
                    estimated_ebitda_avg = item_estimate['estimatedEbitdaAvg']
                except:
                    estimated_ebitda_avg = None
                try:
                    estimated_net_income_avg = item_estimate['estimatedNetIncomeAvg']
                except:
                    estimated_net_income_avg = None
                try:
                    estimated_eps_avg = item_estimate['estimatedEpsAvg']
                except:
                    estimated_eps_avg = None

                try:
                    numOfAnalysts = int((item_estimate['numberAnalystEstimatedRevenue']+item_estimate['numberAnalystsEstimatedEps'])/2)
                except:
                    numOfAnalysts = None

                combined_data[year].update({
                'symbol': item_estimate['symbol'],
                'date': int(item_estimate['date'][:4]),
                'estimatedRevenueAvg': estimated_revenue_avg,
                'estimatedEbitdaAvg': estimated_ebitda_avg,
                'estimatedNetIncomeAvg': estimated_net_income_avg,
                'estimatedEpsAvg': estimated_eps_avg,
                'numOfAnalysts': numOfAnalysts,
                'revenue': None,
                'netIncome': None,
                'ebitda': None,
                'eps': None
                })

        # Convert combined_data to a list
        res = list(combined_data.values())
        res = sorted(res, key=lambda x: x['date'], reverse=False)
    
    except Exception as e:
        print(e)
        res = []

    return res


async def run():

    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    for ticker in tqdm(stock_symbols):
        res = await get_data(ticker, con)
        if len(res) > 0:
            await save_as_json(ticker, res)
    con.close()


try:
    asyncio.run(run())
except Exception as e:
    print(e)
