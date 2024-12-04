import pytz
from datetime import datetime, timedelta
import json
import ujson
import orjson
import asyncio
import aiohttp
import aiofiles
import sqlite3
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from collections import Counter
import re
import hashlib
import glob
from tqdm import tqdm
from utils.country_list import country_list

from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv('FMP_API_KEY')
benzinga_api_key = os.getenv('benzinga_api_key')

berlin_tz = pytz.timezone('Europe/Berlin')


query_price = """
    SELECT 
        close
    FROM 
        "{symbol}"
    WHERE
        date <= ?
    ORDER BY 
        date DESC
    LIMIT 1
"""

query_shares = f"""
    SELECT 
        historicalShares
    FROM 
        stocks
    WHERE
        symbol = ?
"""


time_frames = {
    'change1W': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
    'change1M': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
    'change3M': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
    'change6M': (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
    'change1Y': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
    'change3Y': (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d'),
}

def calculate_price_changes(symbol, item, con):
    try:
        # Loop through each time frame to calculate the change
        for name, date in time_frames.items():
            item[name] = None  # Initialize to None

            query = query_price.format(symbol=symbol)
            data = pd.read_sql_query(query, con, params=(date,))
            
            # Check if data was retrieved and calculate the percentage change
            if not data.empty:
                past_price = data.iloc[0]['close']
                current_price = item['price']
                change = round(((current_price - past_price) / past_price) * 100, 2)
                
                # Set item[name] to None if the change is -100
                item[name] = None if change == -100 else change
                
    except:
        # Handle exceptions by setting all fields to None
        for name in time_frames.keys():
            item[name] = None


def filter_data_quarterly(data):
    # Generate a range of quarter-end dates from the start to the end date
    start_date = data[0]['date']
    end_date = datetime.today().strftime('%Y-%m-%d')
    quarter_ends = pd.date_range(start=start_date, end=end_date, freq='QE').strftime('%Y-%m-%d').tolist()

    # Filter data to keep only entries with dates matching quarter-end dates
    filtered_data = [entry for entry in data if entry['date'] in quarter_ends]
    
    return filtered_data

def calculate_share_changes(symbol, item, con):
    item['sharesQoQ'] = None
    item['sharesYoY'] = None
    item['floatShares'] = None
    try: 
        # Execute query and load data
        df = pd.read_sql_query(query_shares, con, params=(symbol,))
        shareholder_statistics = orjson.loads(df.to_dict()['historicalShares'][0])
        
        # Keys to keep
        keys_to_keep = ["date", "floatShares", "outstandingShares"]

        # Create new list with only the specified keys and convert floatShares and outstandingShares to integers
        shareholder_statistics = [
            {key: int(d[key]) if key in ["floatShares", "outstandingShares"] else d[key] 
             for key in keys_to_keep}
            for d in shareholder_statistics
        ]

        shareholder_statistics = sorted(shareholder_statistics, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=False)
        #Add latest float shares for statistics page
        item['floatShares'] = shareholder_statistics[-1]['floatShares']
        historical_shares = filter_data_quarterly(shareholder_statistics)
        
        latest_data = historical_shares[-1]['outstandingShares']
        previous_quarter = historical_shares[-2]['outstandingShares']
        previous_year = historical_shares[-4]['outstandingShares']

        item['sharesQoQ'] = round((latest_data/previous_quarter-1)*100,2)
        item['sharesYoY'] = round((latest_data/previous_year-1)*100,2)
    except:
        item['sharesQoQ'] = None
        item['sharesYoY'] = None
        item['floatShares'] = None


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

def custom_symbol_sort(item):
    symbol = item['symbol']
    # Use regular expression to check if the symbol matches the typical stock ticker format (e.g., AAPL)
    if re.match(r'^[A-Z]+$', symbol):
        return symbol  # Sort uppercase symbols first
    else:
        return 'ZZZZZZZZZZZZZZ'  # Place non-standard symbols at the bottom

def generate_id(name):
    hashed = hashlib.sha256(name.encode()).hexdigest()
    return hashed[:10]



def count_consecutive_growth_years(financial_data, key_element):
    # Sort the financial data by date
    financial_data = sorted(financial_data, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
    
    consecutive_years = 0
    prev_val = None

    for data in financial_data:
        current_val = data[key_element] #e.g. revenue
        
        if current_val is not None:
            if prev_val is not None:
                if current_val > prev_val:
                    consecutive_years += 1
                else:
                    consecutive_years = 0
            prev_val = current_val

    # Check one last time in case the streak continues to the end
    
    return consecutive_years



def process_financial_data(file_path, key_list):
    """
    Read JSON data from file and extract specified keys with rounding.
    """
    data = defaultdict(lambda: None)  # Initialize with default value of None
    try:
        with open(file_path, 'r') as file:
            res = orjson.loads(file.read())[0]
            for key in key_list:
                if key in res:
                    try:
                        value = float(res[key])
                        if 'growth' in file_path or key in ['grossProfitMargin','netProfitMargin','pretaxProfitMargin','operatingProfitMargin','longTermDebtToCapitalization','totalDebtToCapitalization']:
                            value *= 100  # Multiply by 100 for percentage
                        data[key] = round(value, 2) if value is not None else None
                    except (ValueError, TypeError):
                        # If there's an issue converting the value, leave it as None
                        data[key] = None
    except (FileNotFoundError, KeyError, IndexError):
        # If the file doesn't exist or there's an issue reading the data,
        # data will retain None as the default value for all keys
        pass

    return data

def check_and_process(file_path, key_list):
    """
    Check if the file exists, and then process the financial data if it does.
    If the file doesn't exist, return a dictionary with all keys set to None.
    """
    if os.path.exists(file_path):
        return process_financial_data(file_path, key_list)
    else:
        return {key: None for key in key_list}

def get_financial_statements(item, symbol):
    """
    Update item with financial data from various JSON files.
    """
    # Define the keys to be extracted for each type of statement
    key_ratios = [
        "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
        "daysOfInventoryOutstanding", "operatingCycle", "daysOfPayablesOutstanding",
        "cashConversionCycle", "grossProfitMargin", "operatingProfitMargin",
        "pretaxProfitMargin", "netProfitMargin", "effectiveTaxRate", "returnOnAssets",
        "returnOnEquity", "returnOnCapitalEmployed", "netIncomePerEBT", "ebtPerEbit",
        "ebitPerRevenue", "debtRatio", "debtEquityRatio", "longTermDebtToCapitalization",
        "totalDebtToCapitalization", "interestCoverage", "cashFlowToDebtRatio",
        "companyEquityMultiplier", "receivablesTurnover", "payablesTurnover",
        "inventoryTurnover", "fixedAssetTurnover", "assetTurnover",
        "operatingCashFlowPerShare", "freeCashFlowPerShare", "cashPerShare", "payoutRatio",
        "operatingCashFlowSalesRatio", "freeCashFlowOperatingCashFlowRatio",
        "cashFlowCoverageRatios", "shortTermCoverageRatios", "capitalExpenditureCoverageRatio",
        "dividendPaidAndCapexCoverageRatio", "dividendPayoutRatio", "priceBookValueRatio",
        "priceToBookRatio", "priceToSalesRatio", "priceEarningsRatio", "priceToFreeCashFlowsRatio",
        "priceToOperatingCashFlowsRatio", "priceCashFlowRatio", "priceEarningsToGrowthRatio",
        "priceSalesRatio", "dividendYield", "enterpriseValueMultiple", "priceFairValue"
    ]
    
    key_cash_flow = [
        "netIncome", "depreciationAndAmortization", "deferredIncomeTax", "stockBasedCompensation",
        "changeInWorkingCapital", "accountsReceivables", "inventory", "accountsPayables",
        "otherWorkingCapital", "otherNonCashItems", "netCashProvidedByOperatingActivities",
        "investmentsInPropertyPlantAndEquipment", "acquisitionsNet", "purchasesOfInvestments",
        "salesMaturitiesOfInvestments", "otherInvestingActivites", "netCashUsedForInvestingActivites",
        "debtRepayment", "commonStockIssued", "commonStockRepurchased", "dividendsPaid",
        "otherFinancingActivites", "netCashUsedProvidedByFinancingActivities", "effectOfForexChangesOnCash",
        "netChangeInCash", "cashAtEndOfPeriod", "cashAtBeginningOfPeriod", "operatingCashFlow",
        "capitalExpenditure", "freeCashFlow"
    ]

    key_income = [
        "revenue", "costOfRevenue", "grossProfit", "grossProfitRatio",
        "researchAndDevelopmentExpenses", "generalAndAdministrativeExpenses", "sellingAndMarketingExpenses",
        "sellingGeneralAndAdministrativeExpenses", "otherExpenses", "operatingExpenses",
        "costAndExpenses", "interestIncome", "interestExpense", "depreciationAndAmortization",
        "ebitda", "ebitdaratio", "operatingIncome", "operatingIncomeRatio",
        "totalOtherIncomeExpensesNet", "incomeBeforeTax", "incomeBeforeTaxRatio", "incomeTaxExpense",
        "netIncome", "netIncomeRatio", "eps", "epsdiluted", "weightedAverageShsOut",
        "weightedAverageShsOutDil"
    ]

    key_balance_sheet = [
        "cashAndCashEquivalents", "shortTermInvestments", "cashAndShortTermInvestments",
        "netReceivables", "inventory", "otherCurrentAssets", "totalCurrentAssets",
        "propertyPlantEquipmentNet", "goodwill", "intangibleAssets", "goodwillAndIntangibleAssets",
        "longTermInvestments", "taxAssets", "otherNonCurrentAssets", "totalNonCurrentAssets",
        "otherAssets", "totalAssets", "accountPayables", "shortTermDebt", "taxPayables",
        "deferredRevenue", "otherCurrentLiabilities", "totalCurrentLiabilities", "longTermDebt",
        "deferredRevenueNonCurrent", "deferredTaxLiabilitiesNonCurrent", "otherNonCurrentLiabilities",
        "totalNonCurrentLiabilities", "otherLiabilities", "capitalLeaseObligations", "totalLiabilities",
        "preferredStock", "commonStock", "retainedEarnings", "accumulatedOtherComprehensiveIncomeLoss",
        "othertotalStockholdersEquity", "totalStockholdersEquity", "totalEquity",
        "totalLiabilitiesAndStockholdersEquity", "minorityInterest", "totalLiabilitiesAndTotalEquity",
        "totalInvestments", "totalDebt", "netDebt"
    ]

    key_income_growth = [
        "growthRevenue",
        "growthCostOfRevenue",
        "growthGrossProfit",
        "growthGrossProfitRatio",
        "growthResearchAndDevelopmentExpenses",
        "growthGeneralAndAdministrativeExpenses",
        "growthSellingAndMarketingExpenses",
        "growthOtherExpenses",
        "growthOperatingExpenses",
        "growthCostAndExpenses",
        "growthInterestExpense",
        "growthDepreciationAndAmortization",
        "growthEBITDA",
        "growthEBITDARatio",
        "growthOperatingIncome",
        "growthOperatingIncomeRatio",
        "growthTotalOtherIncomeExpensesNet",
        "growthIncomeBeforeTax",
        "growthIncomeBeforeTaxRatio",
        "growthIncomeTaxExpense",
        "growthNetIncome",
        "growthNetIncomeRatio",
        "growthEPS",
        "growthEPSDiluted",
        "growthWeightedAverageShsOut",
        "growthWeightedAverageShsOutDil"
    ]
    key_cash_flow_growth = [
    "growthNetIncome",
    "growthDepreciationAndAmortization",
    "growthDeferredIncomeTax",
    "growthStockBasedCompensation",
    "growthChangeInWorkingCapital",
    "growthAccountsReceivables",
    "growthInventory",
    "growthAccountsPayables",
    "growthOtherWorkingCapital",
    "growthOtherNonCashItems",
    "growthNetCashProvidedByOperatingActivites",
    "growthInvestmentsInPropertyPlantAndEquipment",
    "growthAcquisitionsNet",
    "growthPurchasesOfInvestments",
    "growthSalesMaturitiesOfInvestments",
    "growthOtherInvestingActivites",
    "growthNetCashUsedForInvestingActivites",
    "growthDebtRepayment",
    "growthCommonStockIssued",
    "growthCommonStockRepurchased",
    "growthDividendsPaid",
    "growthOtherFinancingActivites",
    "growthNetCashUsedProvidedByFinancingActivities",
    "growthEffectOfForexChangesOnCash",
    "growthNetChangeInCash",
    "growthCashAtEndOfPeriod",
    "growthCashAtBeginningOfPeriod",
    "growthOperatingCashFlow",
    "growthCapitalExpenditure",
    "growthFreeCashFlow"
]
    key_balance_sheet_growth = [
        "growthCashAndCashEquivalents",
        "growthShortTermInvestments",
        "growthCashAndShortTermInvestments",
        "growthNetReceivables",
        "growthInventory",
        "growthOtherCurrentAssets",
        "growthTotalCurrentAssets",
        "growthPropertyPlantEquipmentNet",
        "growthGoodwill",
        "growthIntangibleAssets",
        "growthGoodwillAndIntangibleAssets",
        "growthLongTermInvestments",
        "growthTaxAssets",
        "growthOtherNonCurrentAssets",
        "growthTotalNonCurrentAssets",
        "growthOtherAssets",
        "growthTotalAssets",
        "growthAccountPayables",
        "growthShortTermDebt",
        "growthTaxPayables",
        "growthDeferredRevenue",
        "growthOtherCurrentLiabilities",
        "growthTotalCurrentLiabilities",
        "growthLongTermDebt",
        "growthDeferredRevenueNonCurrent",
        "growthDeferredTaxLiabilitiesNonCurrent",
        "growthOtherNonCurrentLiabilities",
        "growthTotalNonCurrentLiabilities",
        "growthOtherLiabilities",
        "growthTotalLiabilities",
        "growthCommonStock",
        "growthRetainedEarnings",
        "growthAccumulatedOtherComprehensiveIncomeLoss",
        "growthOtherTotalStockholdersEquity",
        "growthTotalStockholdersEquity",
        "growthTotalLiabilitiesAndStockholdersEquity",
        "growthTotalInvestments",
        "growthTotalDebt",
        "growthNetDebt"
    ]

    # Process each financial statement
    statements = [
        (f"json/financial-statements/ratios/annual/{symbol}.json", key_ratios),
        (f"json/financial-statements/cash-flow-statement/annual/{symbol}.json", key_cash_flow),
        (f"json/financial-statements/income-statement/annual/{symbol}.json", key_income),
        (f"json/financial-statements/balance-sheet-statement/annual/{symbol}.json", key_balance_sheet),
        (f"json/financial-statements/income-statement-growth/annual/{symbol}.json", key_income_growth),
        (f"json/financial-statements/balance-sheet-statement-growth/annual/{symbol}.json", key_balance_sheet_growth),
        (f"json/financial-statements/cash-flow-statement-growth/annual/{symbol}.json", key_cash_flow_growth)
    ]

    # Process each financial statement
    for file_path, key_list in statements:
        item.update(check_and_process(file_path, key_list))
    
    try:
        item['freeCashFlowMargin'] = round((item['freeCashFlow'] / item['revenue']) * 100,2)
    except:
        item['freeCashFlowMargin'] = None
    try:
        item['earningsYield'] = round((item['eps'] / item['price']) * 100,2)
    except:
        item['earningsYield'] = None
    try:
        item['freeCashFlowYield'] = round((item['freeCashFlow'] / item['marketCap']) * 100,2)
    except:
        item['freeCashFlowYield'] = None
    try:
        item['ebitdaMargin'] = round((item['ebitda'] / item['revenue']) * 100,2)
    except:
        item['ebitdaMargin'] = None
    try:
        item['revenuePerEmployee'] = round((item['revenue'] / item['employees']),2)
    except:
        item['revenuePerEmployee'] = None
    try:
        item['profitPerEmployee'] = round((item['netIncome'] / item['employees']),2)
    except:
        item['profitPerEmployee'] = None
    try:
        tax_rate = item['incomeTaxExpense'] / item['incomeBeforeTax'] if item['incomeBeforeTax'] != 0 else 0
        nopat = item['operatingIncome'] * (1 - tax_rate)
        invested_capital = item['totalDebt'] + item['totalEquity']
        item['returnOnInvestedCapital'] = round((nopat / invested_capital)*100,2) if invested_capital != 0 else None
    except:
        item['returnOnInvestedCapital'] = None
    try:
        item['researchDevelopmentRevenueRatio'] = round((item['researchAndDevelopmentExpenses'] / item['revenue']) * 100,2)
    except:
        item['researchDevelopmentRevenueRatio'] = None
    try:
        item['shortTermDebtToCapitalization'] = round((item['shortTermDebt'] / item['marketCap']) * 100,1)
    except:
        item['shortTermDebtToCapitalization'] = None
    try:
        item['interestIncomeToCapitalization'] = round((item['interestIncome'] / item['marketCap']) * 100,1)
    except:
        item['interestIncomeToCapitalization'] = None

    try:
        item['ebit'] = item['operatingIncome']
        item['operatingMargin'] = round((item['operatingIncome'] / item['revenue']) * 100,2)
        item['ebitMargin'] = item['operatingMargin']
    except:
        item['ebit'] = None
        item['operatingMargin'] = None
        item['ebitMargin'] = None


    return item

def get_halal_compliant(item, debt_threshold=30, interest_threshold=30, revenue_threshold=5, liquidity_threshold=30, forbidden_industries=None):
    # Set default forbidden industries if not provided
    if forbidden_industries is None:
        forbidden_industries = {'Alcohol', 'Equity', 'Palantir','Holding', 'Acquisition','Tobacco', 'Gambling', 'Weapons', 'Pork', 'Aerospace', 'Defense', 'Asset', 'Banks'}

    # Ensure all required fields are present
    required_fields = [
        'longTermDebtToCapitalization', 
        'shortTermDebtToCapitalization', 
        'interestIncomeToCapitalization', 
        'cashAndCashEquivalents',  # Field for liquidity
        'totalAssets',  # Field for liquidity
        'name', 
        'industry',
        'country',
    ]
    for field in required_fields:
        if field not in item:
            halal_compliant = None  # In case of missing data
            return halal_compliant

    # Calculate liquidity ratio
    liquidity_ratio = (item['cashAndCashEquivalents'] / item['totalAssets']) * 100

    # Apply halal-compliance checks
    if (item['country'] == 'United States'
        and item['longTermDebtToCapitalization'] < debt_threshold
        and item['shortTermDebtToCapitalization'] < debt_threshold
        and item['interestIncomeToCapitalization'] < interest_threshold
        and liquidity_ratio < liquidity_threshold  # Liquidity ratio check
        and not any(sector in item['name'] for sector in forbidden_industries)
        and not any(industry in item['industry'] for industry in forbidden_industries)):

        halal_compliant = 'Compliant'
    else:
        halal_compliant = 'Non-Compliant'
    return halal_compliant

def get_country_name(country_code):
    for country in country_list:
        if country['short'] == country_code:
            return country['long']
    return None

def calculate_cagr(start_value, end_value, periods):
    try:
        return round(((end_value / start_value) ** (1 / periods) - 1) * 100, 2)
    except:
        return None


async def get_stock_screener(con):
    #Stock Screener Data
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    
    next_year = datetime.now().year+1

    #Stock Screener Data
    
    cursor.execute("SELECT symbol, name, sma_20, sma_50, sma_100, sma_200, ema_20, ema_50, ema_100, ema_200, rsi, atr, stoch_rsi, mfi, cci, beta FROM stocks WHERE symbol NOT LIKE '%.%' AND eps IS NOT NULL AND marketCap IS NOT NULL AND beta IS NOT NULL")
    raw_data = cursor.fetchall()
    stock_screener_data = [{
            'symbol': symbol,
            'name': name,
            'sma20': sma_20,
            'sma50': sma_50,
            'sma100': sma_100,
            'sma200': sma_200,
            'ema20': ema_20,
            'ema50': ema_50,
            'ema100': ema_100,
            'ema200': ema_200,
            'rsi': rsi,
            'atr': atr,
            'stochRSI': stoch_rsi,
            'mfi': mfi,
            'cci': cci,
            'beta': beta,
        } for (symbol, name, sma_20, sma_50, sma_100, sma_200, ema_20, ema_50, ema_100, ema_200, rsi, atr, stoch_rsi, mfi, cci, beta) in raw_data]

    stock_screener_data = [{k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in entry.items()} for entry in stock_screener_data]

   
    cursor.execute("SELECT symbol, name,  sma_50, sma_200, ema_50, ema_200, rsi, atr, stoch_rsi, mfi, cci, beta FROM stocks WHERE symbol NOT LIKE '%.%' AND eps IS NOT NULL AND marketCap IS NOT NULL AND beta IS NOT NULL")
    raw_data = cursor.fetchall()

    # Iterate through stock_screener_data and update 'price' and 'changesPercentage' if symbols match
    # Add VaR value to stock screener
    for item in tqdm(stock_screener_data):
        symbol = item['symbol']

        try:
            with open(f"json/quote/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                item['price'] = round(float(res['price']),2)
                item['changesPercentage'] = round(float(res['changesPercentage']),2)
                item['avgVolume'] = int(res['avgVolume'])
                item['volume'] = int(res['volume'])
                item['relativeVolume'] = round(( item['volume'] / item['avgVolume'] )*100,2)
                item['pe'] = round(float(res['pe']),2)
                item['marketCap'] = int(res['marketCap'])
        except:
            item['price'] = None
            item['changesPercentage'] = None
            item['avgVolume'] = None
            item['volume'] = None
            item['relativeVolume'] = None
            item['pe'] = None
            item['marketCap'] = None

        calculate_price_changes(symbol, item, con)
        calculate_share_changes(symbol, item, con)
        

        try:
            with open(f"json/stockdeck/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                item['employees'] = int(res['fullTimeEmployees'])
                item['sharesOutStanding'] = int(res['sharesOutstanding'])
                item['country'] = get_country_name(res['country'])
                item['sector'] = res['sector']
                item['industry'] = res['industry']
        except:
            item['employees'] = None
            item['sharesOutStanding'] = None
            item['country'] = None
            item['sector'] = None
            item['industry'] = None

        try:
            with open(f"json/stockdeck/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                data = res['stockSplits'][0]
                item['lastStockSplit'] = data['date']
                item['splitType'] = 'forward' if data['numerator'] > data['denominator'] else 'backward'
                item['splitRatio'] = f"{data['numerator']}"+":"+f"{data['denominator']}"
        except:
            item['lastStockSplit'] = None
            item['splitType'] = None
            item['splitRatio'] = None

        #Financial Statements
        item.update(get_financial_statements(item, symbol))

        try:
            with open(f"json/financial-statements/income-statement/annual/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
            
            # Ensure there are enough elements in the list
            if len(res) >= 5:
                latest_revenue = int(res[0].get('revenue', 0))
                revenue_3_years_ago = int(res[2].get('revenue', 0))
                revenue_5_years_ago = int(res[4].get('revenue', 0))

                latest_eps = int(res[0].get('eps', 0))
                eps_3_years_ago = int(res[2].get('eps', 0))  # eps 3 years ago
                eps_5_years_ago = int(res[4].get('eps', 0))  # eps 5 years ago
                
                item['cagr3YearRevenue'] = calculate_cagr(revenue_3_years_ago, latest_revenue, 3)
                item['cagr5YearRevenue'] = calculate_cagr(revenue_5_years_ago, latest_revenue, 5)
                item['cagr3YearEPS'] = calculate_cagr(eps_3_years_ago, latest_eps, 3)
                item['cagr5YearEPS'] = calculate_cagr(eps_5_years_ago, latest_eps, 5)
            else:
                item['cagr3YearRevenue'] = None
                item['cagr5YearRevenue'] = None
                item['cagr3YearEPS'] = None
                item['cagr3YearEPS'] = None

        except (FileNotFoundError, orjson.JSONDecodeError) as e:
            item['cagr3YearRevenue'] = None
            item['cagr5YearRevenue'] = None
            item['cagr3YearEPS'] = None
            item['cagr5YearEPS'] = None

        try:
            with open(f"json/var/{symbol}.json", 'r') as file:
                item['var'] = orjson.loads(file.read())['history'][-1]['var']
        except:
            item['var'] = None

        try:
            with open(f"json/enterprise-values/{symbol}.json", 'r') as file:
                ev = orjson.loads(file.read())[-1]['enterpriseValue']
                item['enterpriseValue'] = ev
                item['evSales'] = round(ev / item['revenue'],2)
                item['evEarnings'] = round(ev / item['netIncome'],2)
                item['evEBITDA'] = round(ev / item['ebitda'],2)
                item['evEBIT'] = round(ev / item['ebit'],2)
                item['evFCF'] = round(ev / item['freeCashFlow'],2)
        except:
            item['enterpriseValue'] = None
            item['evSales'] = None
            item['evEarnings'] = None
            item['evEBITDA'] = None
            item['evEBIT'] = None
            item['evFCF'] = None

        try:
            with open(f"json/analyst/summary/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                item['analystRating'] = res['consensusRating']
                item['analystCounter'] = res['numOfAnalyst']
                item['priceTarget'] = res['medianPriceTarget']
                item['upside'] = round((item['priceTarget']/item['price']-1)*100, 1) if item['price'] else None
        except Exception as e:
            item['analystRating'] = None
            item['analystCounter'] = None
            item['priceTarget'] = None
            item['upside'] = None

        try:
            with open(f"json/fail-to-deliver/companies/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())[-1]
                item['failToDeliver'] = res['failToDeliver']
                item['relativeFTD'] = round((item['failToDeliver']/item['avgVolume'] )*100,2)
        except Exception as e:
            item['failToDeliver'] = None
            item['relativeFTD'] = None

        try:
            with open(f"json/ownership-stats/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                if res['ownershipPercent'] > 100:
                    item['institutionalOwnership'] = 99.99
                else:
                    item['institutionalOwnership'] = round(res['ownershipPercent'],2)
        except Exception as e:
            item['institutionalOwnership'] = None

        try:
            with open(f"json/financial-statements/key-metrics/annual/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())[0]
                item['revenuePerShare'] = round(res['revenuePerShare'],2)
                item['netIncomePerShare'] = round(res['netIncomePerShare'],2)
                item['shareholdersEquityPerShare'] = round(res['shareholdersEquityPerShare'],2)
                item['interestDebtPerShare'] = round(res['interestDebtPerShare'],2)
                item['capexPerShare'] = round(res['capexPerShare'],2)
                item['tangibleAssetValue'] = round(res['tangibleAssetValue'],2)
                item['returnOnTangibleAssets'] = round(res['returnOnTangibleAssets'],2)
                item['grahamNumber'] = round(res['grahamNumber'],2)

        except:
            item['revenuePerShare'] = None
            item['netIncomePerShare'] = None
            item['shareholdersEquityPerShare'] = None
            item['interestDebtPerShare'] = None
            item['capexPerShare'] = None
            item['tangibleAssetValue'] = None
            item['returnOnTangibleAssets'] = None
            item['grahamNumber'] = None


        try:
            with open(f"json/financial-statements/key-metrics/ttm/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())[0]
                item['revenueTTM'] = round(res['revenuePerShareTTM']*item['sharesOutStanding'],2)
                item['netIncomeTTM'] = round(res['netIncomePerShareTTM']*item['sharesOutStanding'],2)
         
        except:
            item['revenueTTM'] = None
            item['netIncomeTTM'] = None
            

        try:
            with open(f"json/ai-score/companies/{symbol}.json", 'r') as file:
                score = orjson.loads(file.read())['score']
                
                if  score == 10:
                    item['score'] = 'Strong Buy'
                elif score in [7,8,9]:
                    item['score'] = 'Buy'
                elif score in [4,5,6]:
                    item['score'] = 'Hold'
                elif score in [2,3]:
                    item['score'] = 'Sell'
                elif score == 1:
                    item['score'] = 'Strong Sell'
                else:
                    item['score'] = None
        except:
            item['score'] = None

        try:
            with open(f"json/forward-pe/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                if res['forwardPE'] != 0:
                    item['forwardPE'] = round(res['forwardPE'],2)
        except:
            item['forwardPE'] = None

        try:
            with open(f"json/financial-score/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                item['altmanZScore'] = res['altmanZScore']
                item['piotroskiScore'] = res['piotroskiScore']
                item['workingCapital'] = res['workingCapital']
                item['totalAssets'] = res['totalAssets']

        except:
            item['altmanZScore'] = None
            item['piotroskiScore'] = None
            item['workingCapital'] = None
            item['totalAssets'] = None

        try:
            with open(f"json/dividends/companies/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                item['annualDividend'] = round(res['annualDividend'],2)
                item['dividendYield'] = round(res['dividendYield'],2)
                item['payoutRatio'] = round(res['payoutRatio'],2)
                item['dividendGrowth'] = round(res['dividendGrowth'],2)
        except:
            item['annualDividend'] = None
            item['dividendYield'] = None
            item['payoutRatio'] = None
            item['dividendGrowth'] = None



        try:
            with open(f"json/share-statistics/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                item['sharesShort'] = round(float(res['sharesShort']),2)
                item['shortRatio'] = round(float(res['shortRatio']),2)
                item['shortOutStandingPercent'] = round(float(res['shortOutStandingPercent']),2)
                item['shortFloatPercent'] = round(float(res['shortFloatPercent']),2)
        except:
            item['sharesShort'] = None
            item['shortRatio'] = None
            item['shortOutStandingPercent'] = None
            item['shortFloatPercent'] = None


        try:
            with open(f"json/analyst-estimate/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                item['forwardPS'] = None
                item['peg'] = None
                for analyst_item in res:
                    if analyst_item['date'] == next_year and item['marketCap'] > 0 and analyst_item['estimatedRevenueAvg'] > 0:
                        # Calculate forwardPS: marketCap / estimatedRevenueAvg
                        item['forwardPS'] = round(item['marketCap'] / analyst_item['estimatedRevenueAvg'], 1)
                        if item['eps'] > 0:
                            cagr = ((analyst_item['estimatedEpsHigh']/item['eps'] ) -1)*100
                            item['peg'] = round(item['priceEarningsRatio'] / cagr,2) if cagr > 0 else None
                        break  # Exit the loop once the desired item is found
        except:
            item['forwardPS'] = None
            item['peg'] = None
 
        try:
            item['halalStocks'] = get_halal_compliant(item)
        except:
            item['halalStocks'] = None

        try:
            with open(f"json/financial-statements/income-statement/annual/{symbol}.json", "r") as file:
                financial_data = orjson.loads(file.read())
                item['revenueGrowthYears'] = count_consecutive_growth_years(financial_data, "revenue")
                item['epsGrowthYears'] = count_consecutive_growth_years(financial_data, 'eps')
                item['netIncomeGrowthYears'] = count_consecutive_growth_years(financial_data, 'netIncome')
                item['grossProfitGrowthYears'] = count_consecutive_growth_years(financial_data, 'grossProfit')
        except:
            item['revenueGrowthYears'] = None
            item['epsGrowthYears'] = None
            item['netIncomeGrowthYears'] = None
            item['grossProfitGrowthYears'] = None

    for item in stock_screener_data:
        for key, value in item.items():
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    item[key] = None
                    print(key)

    return stock_screener_data


async def get_dividends_calendar(con,symbols):

    berlin_tz = pytz.timezone('Europe/Berlin')
    today = datetime.now(berlin_tz)

    # Calculate the start date (Monday) 4 weeks before
    start_date = today - timedelta(weeks=4)
    start_date = start_date - timedelta(days=(start_date.weekday() - 0) % 7)

    # Calculate the end date (Friday) 4 weeks after
    end_date = today + timedelta(weeks=4)
    end_date = end_date + timedelta(days=(4 - end_date.weekday()) % 7)

    # Format dates as strings in 'YYYY-MM-DD' format
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
        
    async with aiohttp.ClientSession() as session:

        #Database read 1y and 3y data
        query_template = """
            SELECT 
                name, marketCap
            FROM 
                stocks 
            WHERE
                symbol = ?
        """
        
        url = f"https://financialmodelingprep.com/api/v3/stock_dividend_calendar?from={start_date}&to={end_date}&apikey={api_key}"
        async with session.get(url) as response:
            data = await response.json()
            filtered_data = [{k: v for k, v in stock.items() if '.' not in stock['symbol'] and stock['symbol'] in symbols} for stock in data]
            filtered_data = [entry for entry in filtered_data if entry]

            for entry in filtered_data:
                try:
                    symbol = entry['symbol']
                    try:
                        with open(f"json/financial-statements/income-statement/annual/{symbol}.json", 'rb') as file:
                            entry['revenue'] = orjson.loads(file.read())[0]['revenue']
                    except:
                        entry['revenue'] = None

                    data = pd.read_sql_query(query_template, con, params=(symbol,))
                    entry['name'] = data['name'].iloc[0]
                    entry['marketCap'] = int(data['marketCap'].iloc[0])
                except:
                    entry['name'] = 'n/a'
                    entry['marketCap'] = None
                    entry['revenue'] = None

    filtered_data = [d for d in filtered_data if d['symbol'] in symbols]

    return filtered_data



async def get_earnings_calendar(con, stock_symbols):
    def remove_duplicates(elements):
        seen = set()
        unique_elements = []
        for element in elements:
            if element['symbol'] not in seen:
                seen.add(element['symbol'])
                unique_elements.append(element)
        return unique_elements

    BENZINGA_API_KEY = os.getenv('BENZINGA_API_KEY')
    headers = {"accept": "application/json"}
    url = "https://api.benzinga.com/api/v2.1/calendar/earnings"
    importance_list = ["0", "1", "2", "3", "4", "5"]
    berlin_tz = pytz.timezone('Europe/Berlin')

    today = datetime.now(berlin_tz)
    # Start date: 2 weeks ago, rounded to the previous Monday
    start_date = today - timedelta(weeks=2)
    start_date -= timedelta(days=start_date.weekday())  # Reset to Monday

    # End date: 2 weeks ahead, rounded to the following Friday
    end_date = today + timedelta(weeks=2)
    end_date += timedelta(days=(4 - end_date.weekday()))  # Set to Friday

    res_list = []
    async with aiohttp.ClientSession() as session:
        query_template = """
            SELECT name, marketCap FROM stocks WHERE symbol = ?
        """
        while start_date <= end_date:
            date_str = start_date.strftime('%Y-%m-%d')  # Same date for start and end in API call
            for importance in importance_list:
                querystring = {
                    "token": BENZINGA_API_KEY,
                    "parameters[importance]": importance,
                    "parameters[date_from]": date_str,
                    "parameters[date_to]": date_str
                }
                try:
                    async with session.get(url, params=querystring, headers=headers) as response:
                        data = ujson.loads(await response.text())['earnings']
                        res = [item for item in data if item['ticker'] in stock_symbols and '.' not in item['ticker'] and '-' not in item['ticker']]
                        
                        for item in res:
                            try:
                                symbol = item['ticker']
                                eps_prior = float(item['eps_prior']) if item['eps_prior'] else None
                                eps_est = float(item['eps_est']) if item['eps_est'] else None
                                revenue_est = float(item['revenue_est']) if item['revenue_est'] else None
                                revenue_prior = float(item['revenue_prior']) if item['revenue_prior'] else None
                                
                                # Time-based release type
                                time = datetime.strptime(item['time'], "%H:%M:%S").time()
                                if time < datetime.strptime("09:30:00", "%H:%M:%S").time():
                                    release = "bmo"
                                elif time > datetime.strptime("16:00:00", "%H:%M:%S").time():
                                    release = "amc"
                                else:
                                    release = "during"
                                
                                # Only include valid data
                                df = pd.read_sql_query(query_template, con, params=(symbol,))
                                market_cap = float(df['marketCap'].iloc[0]) if df['marketCap'].iloc[0] else 0
                                res_list.append({
                                    'symbol': symbol,
                                    'name': item['name'],
                                    'date': item['date'],
                                    'marketCap': market_cap,
                                    'epsPrior': eps_prior,
                                    'epsEst': eps_est,
                                    'revenuePrior': revenue_prior,
                                    'revenueEst': revenue_est,
                                    'release': release
                                })
                            except Exception as e:
                                print(f"Error processing item for symbol {symbol}: {e}")
                                continue

                    res_list = remove_duplicates(res_list)
                    res_list.sort(key=lambda x: x['marketCap'], reverse=True)

                except:
                    #print(f"Error fetching data from API: {e}")
                    continue

            start_date += timedelta(days=1)  # Increment date by one day

    seen_symbols = set()
    unique_data = []

    for item in res_list:
        symbol = item.get('symbol')
        try:
            with open(f"json/quote/{symbol}.json", 'r') as file:
                quote = ujson.load(file)
                try:
                    earnings_date = datetime.strptime(quote['earningsAnnouncement'].split('T')[0], '%Y-%m-%d').strftime('%Y-%m-%d')
                except:
                    earnings_date = '-'
        except Exception as e:
            earnings_date = '-'
            print(e)

        if symbol is None or symbol not in seen_symbols:
            #bug in fmp endpoint. Double check that earnings date is the same as in quote endpoint
            if item['date'] == earnings_date:
                #print(symbol, item['date'], earnings_date)
                unique_data.append(item)
            seen_symbols.add(symbol)

    return res_list


async def get_stock_splits_calendar(con,symbols):

    berlin_tz = pytz.timezone('Europe/Berlin')
    today = datetime.now(berlin_tz)

    # Calculate the start date (Monday) 4 weeks before
    start_date = today - timedelta(weeks=4)
    start_date = start_date - timedelta(days=(start_date.weekday() - 0) % 7)

    # Calculate the end date (Friday) 4 weeks after
    end_date = today + timedelta(weeks=4)
    end_date = end_date + timedelta(days=(4 - end_date.weekday()) % 7)

    # Format dates as strings in 'YYYY-MM-DD' format
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
        
    async with aiohttp.ClientSession() as session:

        #Database read 1y and 3y data
        query_template = """
            SELECT 
                name, marketCap,eps
            FROM 
                stocks 
            WHERE
                symbol = ?
        """
        
        url = f"https://financialmodelingprep.com/api/v3/stock_split_calendar?from={start_date}&to={end_date}&apikey={api_key}"
        async with session.get(url) as response:
            data = await response.json()
            filtered_data = [{k: v for k, v in stock.items() if stock['symbol'] in symbols} for stock in data]
            filtered_data = [entry for entry in filtered_data if entry]

            for entry in filtered_data:
                try:
                    symbol = entry['symbol']

                    data = pd.read_sql_query(query_template, con, params=(symbol,))
                    entry['name'] = data['name'].iloc[0]
                    entry['marketCap'] = int(data['marketCap'].iloc[0])
                    entry['eps'] = float(data['eps'].iloc[0])
                except:
                    entry['name'] = 'n/a'
                    entry['marketCap'] = None
                    entry['eps'] = None

    filtered_data = [d for d in filtered_data if d['symbol'] in symbols]

    return filtered_data


async def get_economic_calendar():

    country_list = [
    {'short': 'AW', 'long': 'ABW'},
    {'short': 'AF', 'long': 'AFG'},
    {'short': 'AO', 'long': 'AGO'},
    {'short': 'AI', 'long': 'AIA'},
    {'short': 'AX', 'long': 'ALA'},
    {'short': 'AL', 'long': 'ALB'},
    {'short': 'AD', 'long': 'AND'},
    {'short': 'AE', 'long': 'ARE'},
    {'short': 'AR', 'long': 'ARG'},
    {'short': 'AM', 'long': 'ARM'},
    {'short': 'AS', 'long': 'ASM'},
    {'short': 'AQ', 'long': 'ATA'},
    {'short': 'TF', 'long': 'ATF'},
    {'short': 'AG', 'long': 'ATG'},
    {'short': 'AU', 'long': 'AUS'},
    {'short': 'AT', 'long': 'AUT'},
    {'short': 'AZ', 'long': 'AZE'},
    {'short': 'BI', 'long': 'BDI'},
    {'short': 'BE', 'long': 'BEL'},
    {'short': 'BJ', 'long': 'BEN'},
    {'short': 'BQ', 'long': 'BES'},
    {'short': 'BF', 'long': 'BFA'},
    {'short': 'BD', 'long': 'BGD'},
    {'short': 'BG', 'long': 'BGR'},
    {'short': 'BH', 'long': 'BHR'},
    {'short': 'BS', 'long': 'BHS'},
    {'short': 'BA', 'long': 'BIH'},
    {'short': 'BL', 'long': 'BLM'},
    {'short': 'BY', 'long': 'BLR'},
    {'short': 'BZ', 'long': 'BLZ'},
    {'short': 'BM', 'long': 'BMU'},
    {'short': 'BO', 'long': 'BOL'},
    {'short': 'BR', 'long': 'BRA'},
    {'short': 'BB', 'long': 'BRB'},
    {'short': 'BN', 'long': 'BRN'},
    {'short': 'BT', 'long': 'BTN'},
    {'short': 'BV', 'long': 'BVT'},
    {'short': 'BW', 'long': 'BWA'},
    {'short': 'CF', 'long': 'CAF'},
    {'short': 'CA', 'long': 'CAN'},
    {'short': 'CC', 'long': 'CCK'},
    {'short': 'CH', 'long': 'CHE'},
    {'short': 'CL', 'long': 'CHL'},
    {'short': 'CN', 'long': 'CHN'},
    {'short': 'CI', 'long': 'CIV'},
    {'short': 'CM', 'long': 'CMR'},
    {'short': 'CD', 'long': 'COD'},
    {'short': 'CG', 'long': 'COG'},
    {'short': 'CK', 'long': 'COK'},
    {'short': 'CO', 'long': 'COL'},
    {'short': 'KM', 'long': 'COM'},
    {'short': 'CV', 'long': 'CPV'},
    {'short': 'CR', 'long': 'CRI'},
    {'short': 'CU', 'long': 'CUB'},
    {'short': 'CW', 'long': 'CUW'},
    {'short': 'CX', 'long': 'CXR'},
    {'short': 'KY', 'long': 'CYM'},
    {'short': 'CY', 'long': 'CYP'},
    {'short': 'CZ', 'long': 'CZE'},
    {'short': 'DE', 'long': 'DEU'},
    {'short': 'DJ', 'long': 'DJI'},
    {'short': 'DM', 'long': 'DMA'},
    {'short': 'DK', 'long': 'DNK'},
    {'short': 'DO', 'long': 'DOM'},
    {'short': 'DZ', 'long': 'DZA'},
    {'short': 'EC', 'long': 'ECU'},
    {'short': 'EG', 'long': 'EGY'},
    {'short': 'ER', 'long': 'ERI'},
    {'short': 'EH', 'long': 'ESH'},
    {'short': 'ES', 'long': 'ESP'},
    {'short': 'EE', 'long': 'EST'},
    {'short': 'ET', 'long': 'ETH'},
    {'short': 'FI', 'long': 'FIN'},
    {'short': 'FJ', 'long': 'FJI'},
    {'short': 'FK', 'long': 'FLK'},
    {'short': 'FR', 'long': 'FRA'},
    {'short': 'FO', 'long': 'FRO'},
    {'short': 'FM', 'long': 'FSM'},
    {'short': 'GA', 'long': 'GAB'},
    {'short': 'GB', 'long': 'GBR'},
    {'short': 'GE', 'long': 'GEO'},
    {'short': 'GG', 'long': 'GGY'},
    {'short': 'GH', 'long': 'GHA'},
    {'short': 'GI', 'long': 'GIB'},
    {'short': 'GN', 'long': 'GIN'},
    {'short': 'GP', 'long': 'GLP'},
    {'short': 'GM', 'long': 'GMB'},
    {'short': 'GW', 'long': 'GNB'},
    {'short': 'GQ', 'long': 'GNQ'},
    {'short': 'GR', 'long': 'GRC'},
    {'short': 'GD', 'long': 'GRD'},
    {'short': 'GL', 'long': 'GRL'},
    {'short': 'GT', 'long': 'GTM'},
    {'short': 'GF', 'long': 'GUF'},
    {'short': 'GU', 'long': 'GUM'},
    {'short': 'GY', 'long': 'GUY'},
    {'short': 'HK', 'long': 'HKG'},
    {'short': 'HM', 'long': 'HMD'},
    {'short': 'HN', 'long': 'HND'},
    {'short': 'HR', 'long': 'HRV'},
    {'short': 'HT', 'long': 'HTI'},
    {'short': 'HU', 'long': 'HUN'},
    {'short': 'ID', 'long': 'IDN'},
    {'short': 'IM', 'long': 'IMN'},
    {'short': 'IN', 'long': 'IND'},
    {'short': 'IO', 'long': 'IOT'},
    {'short': 'IE', 'long': 'IRL'},
    {'short': 'IR', 'long': 'IRN'},
    {'short': 'IQ', 'long': 'IRQ'},
    {'short': 'IS', 'long': 'ISL'},
    {'short': 'IL', 'long': 'ISR'},
    {'short': 'IT', 'long': 'ITA'},
    {'short': 'JM', 'long': 'JAM'},
    {'short': 'JE', 'long': 'JEY'},
    {'short': 'JO', 'long': 'JOR'},
    {'short': 'JP', 'long': 'JPN'},
    {'short': 'KZ', 'long': 'KAZ'},
    {'short': 'KE', 'long': 'KEN'},
    {'short': 'KG', 'long': 'KGZ'},
    {'short': 'KH', 'long': 'KHM'},
    {'short': 'KI', 'long': 'KIR'},
    {'short': 'KN', 'long': 'KNA'},
    {'short': 'KR', 'long': 'KOR'},
    {'short': 'KW', 'long': 'KWT'},
    {'short': 'LA', 'long': 'LAO'},
    {'short': 'LB', 'long': 'LBN'},
    {'short': 'LR', 'long': 'LBR'},
    {'short': 'LY', 'long': 'LBY'},
    {'short': 'LC', 'long': 'LCA'},
    {'short': 'LI', 'long': 'LIE'},
    {'short': 'LK', 'long': 'LKA'},
    {'short': 'LS', 'long': 'LSO'},
    {'short': 'LT', 'long': 'LTU'},
    {'short': 'LU', 'long': 'LUX'},
    {'short': 'LV', 'long': 'LVA'},
    {'short': 'MO', 'long': 'MAC'},
    {'short': 'MF', 'long': 'MAF'},
    {'short': 'MA', 'long': 'MAR'},
    {'short': 'MC', 'long': 'MCO'},
    {'short': 'MD', 'long': 'MDA'},
    {'short': 'MG', 'long': 'MDG'},
    {'short': 'MV', 'long': 'MDV'},
    {'short': 'MX', 'long': 'MEX'},
    {'short': 'MH', 'long': 'MHL'},
    {'short': 'MK', 'long': 'MKD'},
    {'short': 'ML', 'long': 'MLI'},
    {'short': 'MT', 'long': 'MLT'},
    {'short': 'MM', 'long': 'MMR'},
    {'short': 'ME', 'long': 'MNE'},
    {'short': 'MN', 'long': 'MNG'},
    {'short': 'MP', 'long': 'MNP'},
    {'short': 'MZ', 'long': 'MOZ'},
    {'short': 'MR', 'long': 'MRT'},
    {'short': 'MS', 'long': 'MSR'},
    {'short': 'MQ', 'long': 'MTQ'},
    {'short': 'MU', 'long': 'MUS'},
    {'short': 'MW', 'long': 'MWI'},
    {'short': 'MY', 'long': 'MYS'},
    {'short': 'YT', 'long': 'MYT'},
    {'short': 'NA', 'long': 'NAM'},
    {'short': 'NC', 'long': 'NCL'},
    {'short': 'NE', 'long': 'NER'},
    {'short': 'NF', 'long': 'NFK'},
    {'short': 'NG', 'long': 'NGA'},
    {'short': 'NI', 'long': 'NIC'},
    {'short': 'NU', 'long': 'NIU'},
    {'short': 'NL', 'long': 'NLD'},
    {'short': 'NO', 'long': 'NOR'},
    {'short': 'NP', 'long': 'NPL'},
    {'short': 'NR', 'long': 'NRU'},
    {'short': 'NZ', 'long': 'NZL'},
    {'short': 'OM', 'long': 'OMN'},
    {'short': 'PK', 'long': 'PAK'},
    {'short': 'PA', 'long': 'PAN'},
    {'short': 'PN', 'long': 'PCN'},
    {'short': 'PE', 'long': 'PER'},
    {'short': 'PH', 'long': 'PHL'},
    {'short': 'PW', 'long': 'PLW'},
    {'short': 'PG', 'long': 'PNG'},
    {'short': 'PL', 'long': 'POL'},
    {'short': 'PR', 'long': 'PRI'},
    {'short': 'KP', 'long': 'PRK'},
    {'short': 'PT', 'long': 'PRT'},
    {'short': 'PY', 'long': 'PRY'},
    {'short': 'PS', 'long': 'PSE'},
    {'short': 'PF', 'long': 'PYF'},
    {'short': 'QA', 'long': 'QAT'},
    {'short': 'RE', 'long': 'REU'},
    {'short': 'RO', 'long': 'ROU'},
    {'short': 'RU', 'long': 'RUS'},
    {'short': 'RW', 'long': 'RWA'},
    {'short': 'SA', 'long': 'SAU'},
    {'short': 'SD', 'long': 'SDN'},
    {'short': 'SN', 'long': 'SEN'},
    {'short': 'SG', 'long': 'SGP'},
    {'short': 'GS', 'long': 'SGS'},
    {'short': 'SH', 'long': 'SHN'},
    {'short': 'SJ', 'long': 'SJM'},
    {'short': 'SB', 'long': 'SLB'},
    {'short': 'SL', 'long': 'SLE'},
    {'short': 'SV', 'long': 'SLV'},
    {'short': 'SM', 'long': 'SMR'},
    {'short': 'SO', 'long': 'SOM'},
    {'short': 'PM', 'long': 'SPM'},
    {'short': 'RS', 'long': 'SRB'},
    {'short': 'SS', 'long': 'SSD'},
    {'short': 'ST', 'long': 'STP'},
    {'short': 'SR', 'long': 'SUR'},
    {'short': 'SZ', 'long': 'SWZ'},
    {'short': 'SE', 'long': 'SWE'},
    {'short': 'SY', 'long': 'SYR'},
    {'short': 'TH', 'long': 'THA'},
    {'short': 'TJ', 'long': 'TJK'},
    {'short': 'TK', 'long': 'TKL'},
    {'short': 'TM', 'long': 'TKM'},
    {'short': 'TL', 'long': 'TLS'},
    {'short': 'TO', 'long': 'TON'},
    {'short': 'TT', 'long': 'TTO'},
    {'short': 'TN', 'long': 'TUN'},
    {'short': 'TR', 'long': 'TUR'},
    {'short': 'TV', 'long': 'TUV'},
    {'short': 'TZ', 'long': 'TZA'},
    {'short': 'UG', 'long': 'UGA'},
    {'short': 'UA', 'long': 'UKR'},
    {'short': 'UM', 'long': 'UMI'},
    {'short': 'UY', 'long': 'URY'},
    {'short': 'US', 'long': 'USA'},
    {'short': 'UZ', 'long': 'UZB'},
    {'short': 'VA', 'long': 'VAT'},
    {'short': 'VC', 'long': 'VCT'},
    {'short': 'VE', 'long': 'VEN'},
    {'short': 'VG', 'long': 'VGB'},
    {'short': 'VI', 'long': 'VIR'},
    {'short': 'VN', 'long': 'VNM'},
    {'short': 'VU', 'long': 'VUT'},
    {'short': 'WF', 'long': 'WLF'},
    {'short': 'WS', 'long': 'WSM'},
    {'short': 'YE', 'long': 'YEM'},
    {'short': 'ZA', 'long': 'ZAF'},
    {'short': 'ZM', 'long': 'ZMB'},
    {'short': 'ZW', 'long': 'ZWE'}
    ]
    ny_tz = pytz.timezone('America/New_York')
    today = datetime.now(ny_tz)

    start_date = today - timedelta(weeks=6)
    start_date = start_date - timedelta(days=(start_date.weekday() - 0) % 7)

    end_date = today + timedelta(weeks=6)
    end_date = end_date + timedelta(days=(4 - end_date.weekday()) % 7)

    url = "https://api.benzinga.com/api/v2.1/calendar/economics"
    headers = {"accept": "application/json"}

    all_data = []
    current_date = start_date

    async with aiohttp.ClientSession() as session:
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            querystring = {
                "token": benzinga_api_key,
                "parameters[date_from]": date_str,
                "parameters[date_to]": date_str,
            }

            try:
                async with session.get(url, params=querystring, headers=headers) as response:
                    data = ujson.loads(await response.text())['economics']
                    if len(data) > 0:
                        all_data.extend(data)
                        print(f"Fetched data for {date_str}: {len(data)} events")
            except Exception as e:
                print(f'Error fetching data for {date_str}:', e)

            current_date += timedelta(days=1)

    filtered_data = []

    for item in all_data:
        try:
            matching_country = next((c['short'] for c in country_list if c['long'] == item['country']), None)

            if matching_country:
                country_code = matching_country.lower()

                filtered_data.append({
                    'countryCode': country_code,
                    'country': item['country'],
                    'time': item['time'][0:5],
                    'date': item['date'],
                    'prior': round(float(item['prior']), 2) if item['prior'] != '' else '',
                    'consensus': round(float(item['consensus']), 2) if item['consensus'] != '' else '',
                    'actual': round(float(item['actual']), 2) if item['actual'] != '' else '',
                    'importance': item['importance'],
                    'event': item['event_name'],
                })
        except Exception as e:
            pass


    return filtered_data


def replace_representative(office):
    replacements = {
        'Carper, Thomas R. (Senator)': 'Tom Carper',
        'Thomas R. Carper': 'Tom Carper',
        'Tuberville, Tommy (Senator)': 'Tommy Tuberville',
        'Ricketts, Pete (Senator)': 'John Ricketts',
        'Pete Ricketts': 'John Ricketts',
        'Moran, Jerry (Senator)': 'Jerry Moran',
        'Fischer, Deb (Senator)': 'Deb Fischer',
        'Mullin, Markwayne (Senator)': 'Markwayne Mullin',
        'Whitehouse, Sheldon (Senator)': 'Sheldon Whitehouse',
        'Toomey, Pat (Senator)': 'Pat Toomey',
        'Sullivan, Dan (Senator)': 'Dan Sullivan',
        'Capito, Shelley Moore (Senator)': 'Shelley Moore Capito',
        'Roberts, Pat (Senator)': 'Pat Roberts',
        'King, Angus (Senator)': 'Angus King',
        'Hoeven, John (Senator)': 'John Hoeven',
        'Duckworth, Tammy (Senator)': 'Tammy Duckworth',
        'Perdue, David (Senator)': 'David Perdue',
        'Inhofe, James M. (Senator)': 'James M. Inhofe',
        'Murray, Patty (Senator)': 'Patty Murray',
        'Boozman, John (Senator)': 'John Boozman',
        'Loeffler, Kelly (Senator)': 'Kelly Loeffler',
        'Reed, John F. (Senator)': 'John F. Reed',
        'Collins, Susan M. (Senator)': 'Susan M. Collins',
        'Cassidy, Bill (Senator)': 'Bill Cassidy',
        'Wyden, Ron (Senator)': 'Ron Wyden',
        'Hickenlooper, John (Senator)': 'John Hickenlooper',
        'Booker, Cory (Senator)': 'Cory Booker',
        'Donald Beyer, (Senator).': 'Donald Sternoff Beyer',
        'Peters, Gary (Senator)': 'Gary Peters',
        'Donald Sternoff Beyer, (Senator).': 'Donald Sternoff Beyer',
        'Donald S. Beyer, Jr.': 'Donald Sternoff Beyer',
        'Donald Sternoff Honorable Beyer': 'Donald Sternoff Beyer',
        'K. Michael Conaway': 'Michael Conaway',
        'C. Scott Franklin': 'Scott Franklin',
        'Robert C. "Bobby" Scott': 'Bobby Scott',
        'Madison Cawthorn': 'David Madison Cawthorn',
        'Cruz, Ted (Senator)': 'Ted Cruz',
        'Smith, Tina (Senator)': 'Tina Smith',
        'Graham, Lindsey (Senator)': 'Lindsey Graham',
        'Hagerty, Bill (Senator)': 'Bill Hagerty',
        'Scott, Rick (Senator)': 'Rick Scott',
        'Warner, Mark (Senator)': 'Mark Warner',
        'McConnell, A. Mitchell Jr. (Senator)': 'Mitch McConnell',
        'Mitchell McConnell': 'Mitch McConnell',
        'Charles J. "Chuck" Fleischmann': 'Chuck Fleischmann',
        'Vance, J.D. (Senator)': 'James Vance',
        'Neal Patrick MD, Facs Dunn': 'Neal Dunn',
        'Neal Patrick MD, Facs Dunn (Senator)': 'Neal Dunn',
        'Neal Patrick Dunn, MD, FACS': 'Neal Dunn',
        'Neal P. Dunn': 'Neal Dunn',
        'Tillis, Thom (Senator)': 'Thom Tillis',
        'W. Gregory Steube': 'Greg Steube',
        'W. Grego Steube': 'Greg Steube',
        'W. Greg Steube': 'Greg Steube',
        'David David Madison Cawthorn': 'David Madison Cawthorn',
        'Blunt, Roy (Senator)': 'Roy Blunt',
        'Thune, John (Senator)': 'John Thune',
        'Rosen, Jacky (Senator)': 'Jacky Rosen',
        'Britt, Katie (Senator)': 'Katie Britt',
        'James Costa': 'Jim Costa',
        'Lummis, Cynthia (Senator)': 'Cynthia Lummis',
        'Coons, Chris (Senator)': 'Chris Coons',
        'Udall, Tom (Senator)': 'Tom Udall',
        'Kennedy, John (Senator)': 'John Kennedy',
        'Bennet, Michael (Senator)': 'Michael Bennet',
        'Casey, Robert P. Jr. (Senator)': 'Robert Casey',
        'Van Hollen, Chris (Senator)': 'Chris Van Hollen',
        'Manchin, Joe (Senator)': 'Joe Manchin',
        'Cornyn, John (Senator)': 'John Cornyn',
        'Enzy, Michael (Senator)': 'Michael Enzy',
        'Cardin, Benjamin (Senator)': 'Benjamin Cardin',
        'Kaine, Tim (Senator)': 'Tim Kaine',
        'Joseph P. Kennedy III': 'Joe Kennedy',
        'James E Hon Banks': 'Jim Banks',
        'Michael F. Q. San Nicolas': 'Michael San Nicolas',
        'Barbara J Honorable Comstock': 'Barbara Comstock',
        'Darin McKay LaHood': 'Darin LaHood',
        'Mr ': '',
        'Mr. ': '',
        'Dr ': '',
        'Dr. ': '',
        'Mrs ': '',
        'Mrs. ': '',
        '(Senator)': '',
    }

    for old, new in replacements.items():
        office = office.replace(old, new)
        office = ' '.join(office.split())
    return office

async def get_congress_rss_feed(symbols, etf_symbols, crypto_symbols):

    amount_mapping = {
    '$1,001 -': '$1K-$15K',
    '$1,001 - $15,000': '$1K-$15K',
    '$15,001 - $50,000': '$15K-$50K',
    '$15,001 -': '$15K-$50K',
    '$50,001 - $100,000': '$50K-$100K',
    '$100,001 - $250,000': '$100K-$250K',
    '$100,001 - $500,000': '$100K-$500K',
    '$250,001 - $500,000': '$250K-$500K',
    '$500,001 - $1,000,000': '$500K-$1M',
    '$1,000,001 - $5,000,000': '$1M-$5M',
    'Spouse/DC Over $1,000,000': 'Over $1M'
    }

    urls = [f"https://financialmodelingprep.com/api/v4/senate-disclosure-rss-feed?page=0&apikey={api_key}",
            f"https://financialmodelingprep.com/api/v4/senate-disclosure-rss-feed?page=1&apikey={api_key}"]

    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        data = [await response.json() for response in responses]

    data = data[0] +data[1]
    congressional_districts = {"UT": "Utah","CA": "California","NY": "New York","TX": "Texas","FL": "Florida","IL": "Illinois","PA": "Pennsylvania","OH": "Ohio","GA": "Georgia","MI": "Michigan","NC": "North Carolina","AZ": "Arizona","WA": "Washington","CO": "Colorado","OR": "Oregon","VA": "Virginia","NJ": "New Jersey","TN": "Tennessee","MA": "Massachusetts","WI": "Wisconsin","SC": "South Carolina","KY": "Kentucky","LA": "Louisiana","AR": "Arkansas","AL": "Alabama","MS": "Mississippi","NDAL": "North Dakota","SDAL": "South Dakota","MN": "Minnesota","IA": "Iowa","OK": "Oklahoma","ID": "Idaho","NH": "New Hampshire","NE": "Nebraska","MTAL": "Montana","WYAL": "Wyoming","WV": "West Virginia","VTAL": "Vermont","DEAL": "Delaware","RI": "Rhode Island","ME": "Maine","HI": "Hawaii","AKAL": "Alaska","NM": "New Mexico","KS": "Kansas","MS": "Mississippi","CT": "Connecticut","MD": "Maryland","NV": "Nevada",}
    
    for item in data:
        ticker = item.get("ticker")
        ticker = ticker.replace('BRK.A','BRK-A')
        ticker = ticker.replace('BRK/A','BRK-A')
        ticker = ticker.replace('BRK.B','BRK-B')
        ticker = ticker.replace('BRK/B','BRK-B')
        
        if item['assetDescription'] == 'Bitcoin':
            item['ticker'] = 'BTCUSD'
            ticker = item.get("ticker")

        if item['assetDescription'] == 'Ethereum':
            item['ticker'] = 'ETHUSD'
            ticker = item.get("ticker")
        
        item['assetDescription'] = item['assetDescription'].replace('U.S','US')


        if 'Sale' in item['type']:
            item['type'] = 'Sold'
        if 'Purchase' in item['type']:
            item['type'] = 'Bought'

        item['amount'] = amount_mapping.get(item['amount'], item['amount'])


        item['ticker'] = ticker
        if ticker in symbols:
           item["assetType"] = "stock"
        elif ticker in etf_symbols:
            item["assetType"] = "etf"
        elif ticker in crypto_symbols:
            item['assetType'] = "crypto"
        else:
            item['assetType'] = ''

        if 'representative' in item:
            item['representative'] = replace_representative(item['representative'])

        item['id'] = generate_id(item['representative'])

        # Check if 'district' key exists in item
        if 'district' in item:
            # Extract state code from the 'district' value
            state_code = item['district'][:2]
            
            # Replace 'district' value with the corresponding value from congressional_districts
            item['district'] = f"{congressional_districts.get(state_code, state_code)}"

    return data



async def get_all_etf_tickers(etf_con):
    cursor = etf_con.cursor()
    cursor.execute("SELECT symbol, name, totalAssets, numberOfHoldings FROM etfs WHERE totalAssets IS NOT NULL")
    raw_data = cursor.fetchall()

    # Extract only relevant data and sort it
    etf_list_data = sorted([{'symbol': row[0], 'name': row[1], 'totalAssets': row[2], 'numberOfHoldings': row[3]} for row in raw_data], key=custom_symbol_sort)
    return etf_list_data

async def get_all_crypto_tickers(crypto_con):
    cursor = crypto_con.cursor()
    cursor.execute("SELECT symbol, name, marketCap, circulatingSupply, maxSupply FROM cryptos")
    raw_data = cursor.fetchall()

    # Extract only relevant data and sort it
    crypto_list_data = sorted([{'symbol': row[0], 'name': row[1], 'marketCap': row[2], 'circulatingSupply': row[3], 'maxSupply': row[4]} for row in raw_data], key=custom_symbol_sort)
    return crypto_list_data



async def etf_providers(etf_con, etf_symbols):

    etf_provider_list = []

    query_template = """
        SELECT 
            symbol, etfProvider, expenseRatio, totalAssets, numberOfHoldings
        FROM 
            etfs
        WHERE
            symbol = ?
    """
    
    for symbol in etf_symbols:
        try:
            data = pd.read_sql_query(query_template, etf_con, params=(symbol,))
            etf_provider = data['etfProvider'].iloc[0]
            expense_ratio = float(data['expenseRatio'].iloc[0])
            total_assets = int(data['totalAssets'].iloc[0])
            number_of_holdings = int(data['numberOfHoldings'].iloc[0])

            etf_provider_list.append(
                {'symbol': symbol,
                'etfProvider': etf_provider,
                'expenseRatio': expense_ratio,
                'totalAssets': total_assets,
                'numberOfHoldings': number_of_holdings
                }
            )

        except:
            pass
    # Dictionary to store counts and total expense ratios for each etfProvider
    etf_provider_stats = {}

    # Iterate through the list and update the dictionary
    for etf in etf_provider_list:
        etf_provider = etf['etfProvider']
        expense_ratio = etf['expenseRatio']
        number_of_holdings = etf['numberOfHoldings']
        total_assets = etf['totalAssets']

        if etf_provider in etf_provider_stats:
            etf_provider_stats[etf_provider]['count'] += 1
            etf_provider_stats[etf_provider]['totalExpenseRatio'] += expense_ratio
            etf_provider_stats[etf_provider]['totalNumberOfHoldings'] += number_of_holdings
            etf_provider_stats[etf_provider]['totalAssets'] += total_assets
        else:
            etf_provider_stats[etf_provider] = {'count': 1, 'totalExpenseRatio': expense_ratio, 'totalAssets': total_assets, 'totalNumberOfHoldings': number_of_holdings}

    # Create the new list with average expense ratio
    result_list = [
        {'etfProvider': provider, 'funds': stats['count'], 'totalAssets': stats['totalAssets'] ,'avgExpenseRatio': round(stats['totalExpenseRatio'] / stats['count'],2), 'avgHoldings': int(stats['totalNumberOfHoldings'] / stats['count'])}
        for provider, stats in etf_provider_stats.items()
    ]
    result_list = sorted(result_list, key=lambda x: x['totalAssets'], reverse=True)

    return result_list


async def get_ipo_calendar(con, symbols):
    # Define function to get end date of each quarter
    import datetime
    def get_end_of_quarter(year, quarter):
        month = quarter * 3
        return datetime.date(year, month, 1) + datetime.timedelta(days=30)

    start_date = datetime.date(2019, 1, 1)
    end_date = datetime.date.today()
    urls = []
    combined_data = []
    query_quote = """
        SELECT 
            quote
        FROM 
            stocks 
        WHERE
            symbol = ?
    """
    query_open_price = """
        SELECT open
        FROM "{ticker}"
        LIMIT 1
    """

    # Iterate through quarters
    current_date = start_date
    while current_date < end_date:
        # Get end date of current quarter
        end_of_quarter = get_end_of_quarter(current_date.year, (current_date.month - 1) // 3 + 1)
        
        # Ensure end date does not exceed end_date
        if end_of_quarter > end_date:
            end_of_quarter = end_date
        
        # Construct URL with current quarter's start and end dates
        url = f"https://financialmodelingprep.com/api/v3/ipo_calendar?from={current_date}&to={end_of_quarter}&apikey={api_key}"
        
        # Append URL to list
        urls.append(url)
        
        # Move to next quarter
        current_date = end_of_quarter + datetime.timedelta(days=1)

    #print(urls)
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        data = [await response.json() for response in responses]
    
    for sublist in data:
        for item in sublist:
            if item not in combined_data and item['symbol'] in symbols and item['exchange'] in ['NASDAQ Global','NASDAQ Capital','NASDAQ Global Select','NYSE','NASDAQ','Nasdaq','Nyse','Amex']:
                if item['priceRange'] != None:
                    item['priceRange'] = round(float(item['priceRange'].split('-')[0]),2)

                combined_data.append(item)

    res = []
    for entry in combined_data:
        df = pd.read_sql_query(query_quote, con, params=(entry['symbol'],))
        try:
            entry['currentPrice'] = round((ujson.loads(df['quote'].iloc[0])[0]).get('price'),2)
        except:
            entry['currentPrice'] = None
        try:
            entry['marketCap'] = (ujson.loads(df['quote'].iloc[0])[0]).get('marketCap')
        except:
            entry['marketCap'] = None
        try:
            df =  pd.read_sql_query(query_open_price.format(ticker = entry['symbol']), con)
            entry['ipoPrice'] = round(df['open'].iloc[0], 2) if df['open'].iloc[0] != 0 else None
        except:
            entry['ipoPrice'] = entry['priceRange']

        entry['return'] = None if (entry['ipoPrice'] in (0, None) or entry['currentPrice'] in (0, None)) else round(((entry['currentPrice'] / entry['ipoPrice'] - 1) * 100), 2)
        
        res.append({
            "symbol": entry["symbol"],
            "name": entry["company"],
            "date": entry["date"],
            "marketCap": entry["marketCap"],
            "ipoPrice": entry["ipoPrice"],
            "currentPrice": entry["currentPrice"],
            "return": entry["return"],
        })
    
    res_sorted = sorted(res, key=lambda x: x['date'], reverse=True)

    return res_sorted

async def get_most_shorted_stocks(con):
    directory_path = 'json/share-statistics/*.json'

    def filename_has_no_dot(file_path):
        filename = os.path.basename(file_path)
        if filename.endswith('.json'):
            base_name = filename[:-5]  # Remove the .json part
            # Return True only if there is no dot in the base name
            if '.' not in base_name:
                return True
        return False

    async def read_json_files(directory_path):
        for file_path in glob.glob(directory_path):
            if filename_has_no_dot(file_path):
                try:
                    async with aiofiles.open(file_path, 'r') as file:
                        data = await file.read()
                        json_data = json.loads(data)
                        yield file_path, json_data
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading {file_path}: {e}")

    def extract_elements(file_path, data):
        symbol = os.path.basename(file_path).rsplit('.', 1)[0]
        return {
            "symbol": symbol,
            "sharesShort": data.get("sharesShort"),
            "shortRatio": data.get("shortRatio"),
            "shortOutStandingPercent": data.get("shortOutStandingPercent"),
            "shortFloatPercent": data.get("shortFloatPercent"),
            #"sharesShortPriorMonth": data.get("sharesShortPriorMonth"),
            #"latestOutstandingShares": data.get("latestOutstandingShares"),
            #"latestFloatShares": data.get("latestFloatShares")
        }

    # Initialize a list to hold the extracted data
    extracted_data = []

    # Read and process JSON files
    async for file_path, json_data in read_json_files(directory_path):
        element = extract_elements(file_path, json_data)
        short_outstanding_percent = element.get("shortOutStandingPercent")

        # Check if shortOutStandingPercent is at least 20
        if short_outstanding_percent is not None and float(short_outstanding_percent) >= 20 and float(short_outstanding_percent) < 100:
            extracted_data.append(element)

    sorted_list = sorted(extracted_data, key=lambda x: x['shortOutStandingPercent'], reverse=True)

    for index, item in enumerate(sorted_list, start=1):
        try:
            symbol = item['symbol']
            with open(f"json/quote/{symbol}.json") as file:
                data = orjson.loads(file.read())
            item['name'] = data['name']
            item['rank'] = index
            #item['sector'] = data['sector'].iloc[0]
        except Exception as e:
            print(e)

    return sorted_list

async def save_json_files():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')
    crypto_con = sqlite3.connect('crypto.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    crypto_cursor = crypto_con.cursor()
    crypto_cursor.execute("PRAGMA journal_mode = wal")
    crypto_cursor.execute("SELECT DISTINCT symbol FROM cryptos")
    crypto_symbols = [row[0] for row in crypto_cursor.fetchall()]



    stock_screener_data = await get_stock_screener(con)
    with open(f"json/stock-screener/data.json", 'w') as file:
        ujson.dump(stock_screener_data, file)

    
    earnings_list = await get_earnings_calendar(con,symbols)
    with open(f"json/earnings-calendar/calendar.json", 'w') as file:
        ujson.dump(earnings_list, file)

    economic_list = await get_economic_calendar()
    if len(economic_list) > 0:
        with open(f"json/economic-calendar/calendar.json", 'w') as file:
            ujson.dump(economic_list, file)

    dividends_list = await get_dividends_calendar(con,symbols)
    with open(f"json/dividends-calendar/calendar.json", 'w') as file:
        ujson.dump(dividends_list, file)

    
    data = await get_most_shorted_stocks(con)
    with open(f"json/most-shorted-stocks/data.json", 'w') as file:
        ujson.dump(data, file)

    
    data = await get_congress_rss_feed(symbols, etf_symbols, crypto_symbols)
    with open(f"json/congress-trading/rss-feed/data.json", 'w') as file:
        ujson.dump(data, file)
    
    data = await get_ipo_calendar(con, symbols)
    with open(f"json/ipo-calendar/data.json", 'w') as file:
        ujson.dump(data, file)

    data = await get_all_etf_tickers(etf_con)
    with open(f"json/all-symbols/etfs.json", 'w') as file:
        ujson.dump(data, file)

    data = await get_all_crypto_tickers(crypto_con)
    with open(f"json/all-symbols/cryptos.json", 'w') as file:
        ujson.dump(data, file)

    
    data = await etf_providers(etf_con, etf_symbols)
    with open(f"json/all-etf-providers/data.json", 'w') as file:
        ujson.dump(data, file)
    

            
    stock_splits_data = await get_stock_splits_calendar(con,symbols)
    with open(f"json/stock-splits-calendar/calendar.json", 'w') as file:
        ujson.dump(stock_splits_data, file)

    
    
    

    con.close()
    etf_con.close()
    crypto_con.close()
    
        
try:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(save_json_files())
except Exception as e:
    print(e)