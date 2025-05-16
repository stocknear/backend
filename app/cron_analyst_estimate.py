import aiohttp
import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


query_template = """
    SELECT 
        analyst_estimates
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
        with open(f"json/financial-statements/income-statement/annual/{ticker}.json", "r") as file:
            income = ujson.load(file)

        combined_data = defaultdict(dict)
        #Sum up quarter results
        eps_sums = {}
        revenue_sums = {}
        ebitda_sums = {}
        net_income_sums = {}

        # Iterate over the income data
        for item_income in income:
            calendar_year = int(item_income['fiscalYear'])
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
                if year == item_income['fiscalYear']:
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
                        estimated_ebitda_high = item_estimate['estimatedEbitdaHigh']
                        estimated_ebitda_low = item_estimate['estimatedEbitdaLow']
                    except:
                        estimated_ebitda_avg = None
                        estimated_ebitda_high = None
                        estimated_ebitda_low = None
                    try:
                        estimated_net_income_avg = item_estimate['estimatedNetIncomeAvg']
                        estimated_net_income_high = item_estimate['estimatedNetIncomeHigh']
                        estimated_net_income_low = item_estimate['estimatedNetIncomeLow']
                    except:
                        estimated_net_income_avg = None
                        estimated_net_income_high = None
                        estimated_net_income_low = None
                    try:
                        estimated_revenue_avg = item_estimate['estimatedRevenueAvg']
                        estimated_revenue_high = item_estimate['estimatedRevenueHigh']
                        estimated_revenue_low = item_estimate['estimatedRevenueLow']
                    except:
                        estimated_revenue_avg = None
                        estimated_revenue_high = None
                        estimated_revenue_low = None
                    try:
                        estimated_eps_avg = round(item_estimate['estimatedEpsAvg'],2)
                        estimated_eps_high = round(item_estimate['estimatedEpsHigh'],2)
                        estimated_eps_low = round(item_estimate['estimatedEpsLow'],2)
                    except:
                        estimated_eps_avg = None
                        estimated_eps_high = None
                        estimated_eps_low = None

                    try:
                        numOfAnalysts = int((item_estimate['numberAnalystEstimatedRevenue']+item_estimate['numberAnalystsEstimatedEps'])/2)
                    except:
                        numOfAnalysts = None

                    combined_data[year].update({
                    'symbol': item_estimate['symbol'],
                    'date': int(item_estimate['date'][:4]),
                    'estimatedRevenueAvg': estimated_revenue_avg,
                    'estimatedRevenueHigh': estimated_revenue_high,
                    'estimatedRevenueLow': estimated_revenue_low,
                    'estimatedEbitdaAvg': estimated_ebitda_avg,
                    'estimatedEbitdaHigh': estimated_ebitda_high,
                    'estimatedEbitdaLow': estimated_ebitda_low,
                    'estimatedNetIncomeAvg': estimated_net_income_avg,
                    'estimatedNetIncomeHigh': estimated_net_income_high,
                    'estimatedNetIncomeLow': estimated_net_income_low,
                    'estimatedEpsAvg': estimated_eps_avg,
                    'estimatedEpsHigh': estimated_eps_high,
                    'estimatedEpsLow': estimated_eps_low,
                    'revenue': revenue,
                    'netIncome': net_income,
                    'ebitda': ebitda,
                    'eps': eps,
                    'numOfAnalysts': numOfAnalysts,
                })

        for item_estimate in analyst_estimates:
            year = item_estimate['date'][:4]
            if year == income[0]['fiscalYear']:
                break
            else:
                try:
                    estimated_revenue_avg = item_estimate['estimatedRevenueAvg']
                    estimated_revenue_high = item_estimate['estimatedRevenueHigh']
                    estimated_revenue_low = item_estimate['estimatedRevenueLow']
                except:
                    estimated_revenue_avg = None
                    estimated_revenue_high = None
                    estimated_revenue_low = None
                try:
                    estimated_ebitda_avg = item_estimate['estimatedEbitdaAvg']
                    estimated_ebitda_high = item_estimate['estimatedEbitdaHigh']
                    estimated_ebitda_low = item_estimate['estimatedEbitdaLow']
                except:
                    estimated_ebitda_avg = None
                    estimated_ebitda_high = None
                    estimated_ebitda_low = None
                try:
                    estimated_net_income_avg = item_estimate['estimatedNetIncomeAvg']
                    estimated_net_income_high = item_estimate['estimatedNetIncomeHigh']
                    estimated_net_income_low = item_estimate['estimatedNetIncomeLow']
                except:
                    estimated_net_income_avg = None
                    estimated_net_income_high = None
                    estimated_net_income_low = None
                try:
                    estimated_eps_avg = round(item_estimate['estimatedEpsAvg'],2)
                    estimated_eps_high = round(item_estimate['estimatedEpsHigh'],2)
                    estimated_eps_low = round(item_estimate['estimatedEpsLow'],2)
                except:
                    estimated_eps_avg = None
                    estimated_eps_high = None
                    estimated_eps_low = None

                try:
                    numOfAnalysts = int((item_estimate['numberAnalystEstimatedRevenue']+item_estimate['numberAnalystsEstimatedEps'])/2)
                except:
                    numOfAnalysts = None

                combined_data[year].update({
                'symbol': item_estimate['symbol'],
                'date': int(item_estimate['date'][:4]),
                'estimatedRevenueAvg': estimated_revenue_avg,
                'estimatedRevenueHigh': estimated_revenue_high,
                'estimatedRevenueLow': estimated_revenue_low,
                'estimatedEbitdaAvg': estimated_ebitda_avg,
                'estimatedEbitdaHigh': estimated_ebitda_high,
                'estimatedEbitdaLow': estimated_ebitda_low,
                'estimatedNetIncomeAvg': estimated_net_income_avg,
                'estimatedNetIncomeHigh': estimated_net_income_high,
                'estimatedNetIncomeLow': estimated_net_income_low,
                'estimatedEpsAvg': estimated_eps_avg,
                'estimatedEpsHigh': estimated_eps_high,
                'estimatedEpsLow': estimated_eps_low,
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
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
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
