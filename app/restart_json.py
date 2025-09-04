import pytz
from datetime import datetime, timedelta, date
import time
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
from utils.helper import replace_representative


from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv('FMP_API_KEY')
BENZINGA_API_KEY = os.getenv('BENZINGA_API_KEY')

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

one_year_ago = datetime.now() - timedelta(days=365)
today = date.today()
today_str = datetime.today().strftime('%Y-%m-%d')
current_year = datetime.now().year

# YTD start (January 1st of the current year)
ytd_start = date(today.year, 1, 1)

def calculate_price_changes(symbol, item, con):
    try:
        # Loop through each time frame to calculate the change
        for name, date in time_frames.items():
            try:
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
                item[name] = None
                
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
        try:
            current_val = data[key_element] #e.g. revenue
            
            if current_val is not None:
                if prev_val is not None:
                    if current_val > prev_val:
                        consecutive_years += 1
                    else:
                        consecutive_years = 0
                prev_val = current_val
        except:
            pass

    # Check one last time in case the streak continues to the end
    
    return consecutive_years


def filter_latest_analyst_unique_rating(data):
    latest_entries = {}

    for entry in data:
        try:
            # Create a unique key by combining 'analyst' and 'name'
            key = f"{entry.get('analyst')}-{entry.get('name')}"

            # Convert date and time to a datetime object
            date_time_str = f"{entry.get('date')}"
            date_time = datetime.strptime(date_time_str, "%Y-%m-%d")

            # Check if this entry is the latest for the given key
            if key not in latest_entries or date_time > latest_entries[key]['dateTime']:
                latest_entries[key] = {'dateTime': date_time, 'entry': entry}
        except Exception as e:
            print(f"Error processing entry: {e}")

    # Extract and return the latest entries
    return [value['entry'] for value in latest_entries.values()]

def process_top_analyst_data(data, current_price):
    try:
        # Filter for top analysts with score >= 4
        filtered_data = []
        for item in data:
            try:
                if item['analystScore'] >= 4:
                    filtered_data.append(item)
            except:
                pass
        data = filtered_data

        data = filter_latest_analyst_unique_rating(data)
        
        # Filter recent data from last 12 months, limit to 30 most recent
        recent_data = []

        for item in data:
            try:
                if 'date' in item and datetime.strptime(item['date'], "%Y-%m-%d") >= one_year_ago:
                    recent_data.append(item)
            except:
                pass

        recent_data = recent_data[:30]

        if not recent_data:
            return {
                "topAnalystCounter": None,
                "topAnalystPriceTarget": None, 
                "topAnalystUpside": None,
                "topAnalystRating": None
            }

        filtered_analyst_count = len(recent_data)

        # Extract valid price targets
        price_targets = []
        for item in recent_data:
            try:
                pt = item.get('adjusted_pt_current')
                if pt and not math.isnan(float(pt)):
                    price_targets.append(float(pt))
            except (ValueError, TypeError):
                continue

        # Calculate median price target
        median_price_target = None
        if price_targets:
            price_targets.sort()
            mid = len(price_targets) // 2
            median_price_target = (
                price_targets[mid] if len(price_targets) % 2 
                else (price_targets[mid-1] + price_targets[mid]) / 2
            )
            if median_price_target <= 0:
                median_price_target = None

        # Calculate upside percentage
        upside = None
        if median_price_target and current_price and current_price > 0:
            upside = round(((median_price_target / current_price - 1) * 100), 2)

        # Rating scores mapping
        rating_scores = {
            "Strong Buy": 5,
            "Buy": 4, 
            "Hold": 3,
            "Sell": 2,
            "Strong Sell": 1
        }

        # Calculate average rating
        valid_ratings = [
            rating_scores.get(item.get('rating_current', ''), 0) 
            for item in recent_data
        ]
        
        average_rating_score = (
            round(sum(valid_ratings) / len(valid_ratings), 2)
            if valid_ratings else 0
        )

        # Map average score to consensus rating
        consensus_rating = None
        if average_rating_score >= 4.5:
            consensus_rating = "Strong Buy"
        elif average_rating_score >= 3.5:
            consensus_rating = "Buy"
        elif average_rating_score >= 2.5:
            consensus_rating = "Hold" 
        elif average_rating_score >= 1.5:
            consensus_rating = "Sell"
        elif average_rating_score >= 1.0:
            consensus_rating = "Strong Sell"

        return {
            "topAnalystCounter": filtered_analyst_count,
            "topAnalystPriceTarget": median_price_target,
            "topAnalystUpside": upside,
            "topAnalystRating": consensus_rating
        }

    except Exception as e:
        return {
            "topAnalystCounter": None,
            "topAnalystPriceTarget": None,
            "topAnalystUpside": None, 
            "topAnalystRating": None
        }


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
                        if 'growth' in file_path or key in ['effectiveTaxRate','grossProfitMargin','freeCashFlowMargin',"ebitMargin","ebitdaMargin","netProfitMargin","operatingProfitMargin","pretaxProfitMargin"]:
                            value = value*100  # Multiply by 100 for percentage

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
        "grossProfitMargin",
        "ebitMargin",
        "ebitdaMargin",
        "operatingProfitMargin",
        "pretaxProfitMargin",
        "continuousOperationsProfitMargin",
        "netProfitMargin",
        "bottomLineProfitMargin",
        "receivablesTurnover",
        "payablesTurnover",
        "inventoryTurnover",
        "fixedAssetTurnover",
        "assetTurnover",
        "currentRatio",
        "quickRatio",
        "solvencyRatio",
        "cashRatio",
        "priceToEarningsRatio",
        "priceToEarningsGrowthRatio",
        "forwardPriceToEarningsGrowthRatio",
        "priceToBookRatio",
        "priceToSalesRatio",
        "priceToFreeCashFlowRatio",
        "priceToOperatingCashFlowRatio",
        "debtToAssetsRatio",
        "debtToEquityRatio",
        "debtToCapitalRatio",
        "longTermDebtToCapitalRatio",
        "financialLeverageRatio",
        "workingCapitalTurnoverRatio",
        "operatingCashFlowRatio",
        "operatingCashFlowSalesRatio",
        "freeCashFlowOperatingCashFlowRatio",
        "debtServiceCoverageRatio",
        "interestCoverageRatio",
        "shortTermOperatingCashFlowCoverageRatio",
        "operatingCashFlowCoverageRatio",
        "capitalExpenditureCoverageRatio",
        "dividendPaidAndCapexCoverageRatio",
        "dividendPayoutRatio",
        "dividendYield",
        "dividendYieldPercentage",
        "revenuePerShare",
        "netIncomePerShare",
        "interestDebtPerShare",
        "cashPerShare",
        "bookValuePerShare",
        "tangibleBookValuePerShare",
        "shareholdersEquityPerShare",
        "operatingCashFlowPerShare",
        "capexPerShare",
        "freeCashFlowPerShare",
        "netIncomePerEBT",
        "ebtPerEbit",
        "priceToFairValue",
        "debtToMarketCap",
        "effectiveTaxRate",
        "enterpriseValueMultiple",
        "dividendPerShare"
    ]

    
    key_cash_flow = [
        "netIncome",
        "depreciationAndAmortization",
        "deferredIncomeTax",
        "stockBasedCompensation",
        "changeInWorkingCapital",
        "accountsReceivables",
        "inventory",
        "accountsPayables",
        "otherWorkingCapital",
        "otherNonCashItems",
        "netCashProvidedByOperatingActivities",
        "investmentsInPropertyPlantAndEquipment",
        "acquisitionsNet",
        "purchasesOfInvestments",
        "salesMaturitiesOfInvestments",
        "otherInvestingActivities",
        "netCashProvidedByInvestingActivities",
        "netDebtIssuance",
        "longTermNetDebtIssuance",
        "shortTermNetDebtIssuance",
        "netStockIssuance",
        "netCommonStockIssuance",
        "commonStockIssuance",
        "commonStockRepurchased",
        "netPreferredStockIssuance",
        "netDividendsPaid",
        "commonDividendsPaid",
        "preferredDividendsPaid",
        "otherFinancingActivities",
        "netCashProvidedByFinancingActivities",
        "effectOfForexChangesOnCash",
        "netChangeInCash",
        "cashAtEndOfPeriod",
        "cashAtBeginningOfPeriod",
        "operatingCashFlow",
        "capitalExpenditure",
        "freeCashFlow",
        "incomeTaxesPaid",
        "interestPaid"
    ]


    key_income = [
        "revenue",
        "costOfRevenue",
        "grossProfit",
        "researchAndDevelopmentExpenses",
        "generalAndAdministrativeExpenses",
        "sellingAndMarketingExpenses",
        "sellingGeneralAndAdministrativeExpenses",
        "otherExpenses",
        "operatingExpenses",
        "costAndExpenses",
        "netInterestIncome",
        "interestIncome",
        "interestExpense",
        "depreciationAndAmortization",
        "ebitda",
        "ebit",
        "nonOperatingIncomeExcludingInterest",
        "operatingIncome",
        "totalOtherIncomeExpensesNet",
        "incomeBeforeTax",
        "incomeTaxExpense",
        "netIncomeFromContinuingOperations",
        "netIncomeFromDiscontinuedOperations",
        "otherAdjustmentsToNetIncome",
        "netIncome",
        "netIncomeDeductions",
        "bottomLineNetIncome",
        "eps",
        "epsDiluted",
        "weightedAverageShsOut",
        "weightedAverageShsOutDil"
    ]


    key_balance_sheet = [
        "cashAndCashEquivalents",
        "shortTermInvestments",
        "cashAndShortTermInvestments",
        "netReceivables",
        "accountsReceivables",
        "otherReceivables",
        "inventory",
        "prepaids",
        "otherCurrentAssets",
        "totalCurrentAssets",
        "propertyPlantEquipmentNet",
        "goodwill",
        "intangibleAssets",
        "goodwillAndIntangibleAssets",
        "longTermInvestments",
        "taxAssets",
        "otherNonCurrentAssets",
        "totalNonCurrentAssets",
        "otherAssets",
        "totalAssets",
        "totalPayables",
        "accountPayables",
        "otherPayables",
        "accruedExpenses",
        "shortTermDebt",
        "capitalLeaseObligationsCurrent",
        "taxPayables",
        "deferredRevenue",
        "otherCurrentLiabilities",
        "totalCurrentLiabilities",
        "longTermDebt",
        "capitalLeaseObligationsNonCurrent",
        "deferredRevenueNonCurrent",
        "deferredTaxLiabilitiesNonCurrent",
        "otherNonCurrentLiabilities",
        "totalNonCurrentLiabilities",
        "otherLiabilities",
        "capitalLeaseObligations",
        "totalLiabilities",
        "treasuryStock",
        "preferredStock",
        "commonStock",
        "retainedEarnings",
        "additionalPaidInCapital",
        "accumulatedOtherComprehensiveIncomeLoss",
        "otherTotalStockholdersEquity",
        "totalStockholdersEquity",
        "totalEquity",
        "minorityInterest",
        "totalLiabilitiesAndTotalEquity",
        "totalInvestments",
        "totalDebt",
        "netDebt"
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
        try:
            item.update(check_and_process(file_path, key_list))
        except:
            pass
    
    try:
        item['freeCashFlowMargin'] = round((item['freeCashFlow'] / item['revenue']) * 100,2)
    except:
        item['freeCashFlowMargin'] = None

    try:
        item['debtToFreeCashFlowRatio'] = round((item['totalDebt'] / item['freeCashFlow']),2)
    except:
        item['debtToFreeCashFlowRatio'] = None
    try:
        item['debtToEBITDARatio'] = round((item['totalDebt'] / item['ebitda']),2)
    except:
        item['debtToEBITDARatio'] = None
    try:
        item['revenuePerEmployee'] = round((item['revenue'] / item['employees']),2)
    except:
        item['revenuePerEmployee'] = None
    try:
        item['profitPerEmployee'] = round((item['netIncome'] / item['employees']),2)
    except:
        item['profitPerEmployee'] = None
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
        item['operatingMargin'] = round((item['operatingIncome'] / item['revenue']) * 100,2)
        item['ebitMargin'] = item['operatingMargin']
    except:
        item['operatingMargin'] = None
        item['ebitMargin'] = None


    return item

def get_price_on_or_nearest(data, target_date):
    # assume data sorted newest-to-oldest
    # return the adjClose whose 'date' is closest to target_date
    return min(
        data,
        key=lambda x: abs(
            datetime.strptime(x["date"], "%Y-%m-%d").date() - target_date
        )
    )["adjClose"]

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
        res = round(((end_value / start_value) ** (1 / periods) - 1) * 100, 2)
        if res and res != 0:
            return res
        else:
            None
    except:
        return None


def clean_for_json(data):
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return round(data, 4)
    return data

        
async def get_stock_screener(con):
    #Stock Screener Data
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    
    next_year = datetime.now().year+1

    #Stock Screener Data
    cursor.execute("""
        SELECT symbol, name, sma_20, sma_50, sma_100, sma_200,
               ema_20, ema_50, ema_100, ema_200,
               rsi, atr, stoch_rsi, mfi, cci, beta
        FROM stocks
        WHERE symbol NOT LIKE '%.%'
          AND eps IS NOT NULL
          AND marketCap IS NOT NULL
          AND beta IS NOT NULL
          AND sma_20 IS NOT NULL
    """)

    raw_data = cursor.fetchall()
    if len(raw_data) < 4000:
        #make sure the stock screener has all the data
        print(f'Stock Screener Skipped because only {len(raw_data)} stocks found')
        return

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
    #test mode
    #filtered_data = [item for item in stock_screener_data if item['symbol'] == 'TSLA']

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
                item['exchange'] = res['exchange']
        except:
            item['employees'] = None
            item['sharesOutStanding'] = None
            item['country'] = None
            item['sector'] = None
            item['industry'] = None
            item['exchange'] = None

        try:
            with open(f"json/profile/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                item['isin'] = res['isin']
        except:
            item['isin'] = None

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
                latest_revenue = int(res[0].get('revenue', 0) or 0)
                revenue_3_years_ago = int(res[2].get('revenue', 0) or 0)
                revenue_5_years_ago = int(res[4].get('revenue', 0) or 0)

                latest_eps = float(res[0].get('eps', 0) or 0)
                eps_3_years_ago = float(res[2].get('eps', 0) or 0)  # eps 3 years ago
                eps_5_years_ago = float(res[4].get('eps', 0) or 0)  # eps 5 years ago


                item['cagr3YearRevenue'] = calculate_cagr(revenue_3_years_ago, latest_revenue, 3)
                item['cagr5YearRevenue'] = calculate_cagr(revenue_5_years_ago, latest_revenue, 5)
                item['cagr3YearEPS'] = calculate_cagr(eps_3_years_ago, latest_eps, 3)
                item['cagr5YearEPS'] = calculate_cagr(eps_5_years_ago, latest_eps, 5)
                if latest_eps >= 0:
                    #computing lynch fair value
                    growth_rate = min(max(item['cagr5YearEPS'], 5), 25)
                    item['lynchFairValue'] = round(growth_rate * latest_eps, 2)
                    item['lynchUpside'] = round((item['lynchFairValue'] / item['price'] - 1) * 100, 2)
                else:
                    item['lynchFairValue'] = None
                    item['lynchUpside'] = None
            else:
                item['cagr3YearRevenue'] = None
                item['cagr5YearRevenue'] = None
                item['cagr3YearEPS'] = None
                item['cagr3YearEPS'] = None
                item['lynchFairValue'] = None
                item['lynchUpside'] = None

        except:
            item['cagr3YearRevenue'] = None
            item['cagr5YearRevenue'] = None
            item['cagr3YearEPS'] = None
            item['cagr5YearEPS'] = None
            item['lynchFairValue'] = None
            item['lynchUpside'] = None


        
        try:
            with open(f"json/historical-price/adj/{symbol}.json", 'r') as file:
                hist = orjson.loads(file.read())
            
            latest_date = datetime.strptime(hist[0]["date"], "%Y-%m-%d").date()
            latest_p = hist[0]["adjClose"]

            # 1M           
            n1 = 30
            p1m =  hist[n1]["adjClose"]

            # 3) Annualize your month‐over‐month return:
            cagr1m = (latest_p / p1m) - 1.0

            # YTD
            ytd_start = datetime(latest_date.year, 1, 1).date()
            pYTD = get_price_on_or_nearest(hist, ytd_start)
            
            delta_days = (latest_date - ytd_start).days
            cagrYTD = (latest_p / pYTD) - 1.0

            # 1Y
            n252 = 252
            p1y = hist[n252]["adjClose"]
            cagr1y = (latest_p / p1y) ** (1.0) - 1.0  # since T ≈1 year

            # 5Y
            n5 = 252 * 5
            p5y = hist[n5]["adjClose"]
            cagr5y = (latest_p / p5y) ** (1/5) - 1.0

            # Max
            pmax = hist[-1]["adjClose"]
            years_max = (latest_date - datetime.strptime(hist[-1]["date"], "%Y-%m-%d").date()).days / 365.0
            cagr_max = (latest_p / pmax) ** (1/years_max) - 1.0

            # store them:
            item.update({
                'cagr1MReturn': round(cagr1m*100,2),
                'cagrYTDReturn': round(cagrYTD*100,2),
                'cagr1YReturn': round(cagr1y*100,2),
                'cagr5YReturn': round(cagr5y*100,2),
                'cagrMaxReturn': round(cagr_max*100,2),
            })

            #print(item['cagr1MReturn'], item['cagrYTDReturn'], item['cagr1YReturn'], item['cagr5YReturn'], item['cagrMaxReturn'])

        except:
            for key in ['cagr1M','cagrYTD','cagr1Y','cagr5Y','cagrMax']:
                item[key] = None

        try:
            with open(f"json/var/{symbol}.json", 'r') as file:
                item['var'] = orjson.loads(file.read())['history'][-1]['var']
        except:
            item['var'] = None



        try:
            with open(f"json/analyst/summary/all_analyst/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                item['analystRating'] = res['consensusRating']
                item['analystCounter'] = res['numOfAnalyst']
                item['priceTarget'] = res['medianPriceTarget']
                item['upside'] = round((item['priceTarget']/item['price']-1)*100, 1) if item['price'] else None
        except:
            item['analystRating'] = None
            item['analystCounter'] = None
            item['priceTarget'] = None
            item['upside'] = None

        #top analyst rating
        try:
            with open(f"json/analyst/history/{symbol}.json") as file:
                data = orjson.loads(file.read())
                res_dict = process_top_analyst_data(data, item['price'])

                item['topAnalystCounter'] = res_dict['topAnalystCounter']
                item['topAnalystPriceTarget'] = res_dict['topAnalystPriceTarget']
                item['topAnalystUpside'] = res_dict['topAnalystUpside']
                item['topAnalystRating'] = res_dict['topAnalystRating']
        except:
            item['topAnalystCounter'] = None
            item['topAnalystPriceTarget'] = None
            item['topAnalystUpside'] = None
            item['topAnalystRating'] = None


        try:
            with open(f"json/fail-to-deliver/companies/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())[-1]
                item['failToDeliver'] = res['failToDeliver']
                item['relativeFTD'] = round((item['failToDeliver']/item['avgVolume'] )*100,2)
        except:
            item['failToDeliver'] = None
            item['relativeFTD'] = None

        try:
            with open(f"json/ownership-stats/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                if res['ownershipPercent'] > 100:
                    item['institutionalOwnership'] = 99.99
                else:
                    item['institutionalOwnership'] = round(res['ownershipPercent'],2)
        except:
            item['institutionalOwnership'] = None

        try:
            with open(f"json/financial-statements/key-metrics/annual/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())[0]
                item['returnOnEquity'] = round(res['returnOnEquity']*100,2)
                item['returnOnInvestedCapital'] = round(res['returnOnInvestedCapital']*100,2)
                item['returnOnCapitalEmployed'] = round(res['returnOnCapitalEmployed']*100,2)
                item['returnOnAssets'] = round(res['returnOnAssets']*100,2)
                item['returnOnTangibleAssets'] = round(res['returnOnTangibleAssets']*100,2)
                item['stockBasedCompensationToRevenue'] = round(res['stockBasedCompensationToRevenue']*100,2)

                item['earningsYield'] = round(res['earningsYield']*100,2)
                item['freeCashFlowYield'] = round(res['freeCashFlowYield']*100,2)

                item['enterpriseValue'] = res['enterpriseValue']
                item['evToSales'] = round(res['evToSales'],2)
                item['evToOperatingCashFlow'] = round(res['evToOperatingCashFlow'],2)
                item['evToEBIT'] = round(res['evToOperatingCashFlow'],2)
                item['evToFreeCashFlow'] = round(res['evToFreeCashFlow'],2)
                item['evToEBITDA'] = round(res['evToEBITDA'],2)

                item['tangibleAssetValue'] = round(res['tangibleAssetValue'],2)
                item['grahamNumber'] = round(res['grahamNumber'],2)
                item['grahamUpside'] = round((item['grahamNumber'] / item['price'] - 1) * 100, 2)

        except:
            item['returnOnEquity'] = None
            item['returnOnInvestedCapital'] = None
            item['returnOnCapitalEmployed'] = None
            item['returnOnAssets'] = None
            item['earningsYield'] = None
            item['freeCashFlowYield'] = None
            item['enterpriseValue'] = None
            item['evToSales'] = None
            item['evToOperatingCashFlow'] = None
            item['evToFreeCashFlow'] = None
            item['evToEBITDA'] = None
            item['stockBasedCompensationToRevenue'] = None

            item['tangibleAssetValue'] = None
            item['returnOnTangibleAssets'] = None
            item['grahamNumber'] = None
            item['grahamUpside'] = None


        try:
            with open(f"json/financial-statements/income-statement/ttm/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())[0]
                item['revenueTTM'] = int(res['revenue'])
                item['netIncomeTTM'] = int(res['netIncome'])

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
                item['payoutFrequency'] = str(res['payoutFrequency']) if res['payoutFrequency'] != 'Special' else None
        except:
            item['annualDividend'] = None
            item['dividendYield'] = None
            item['payoutRatio'] = None
            item['dividendGrowth'] = None
            item['payoutFrequency'] = None

        try:
            with open(f"json/share-statistics/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                item['sharesShort'] = res['sharesShort']
                item['shortRatio'] = res['shortRatio']
                item['shortOutstandingPercent'] = res['shortOutstandingPercent']
                item['shortFloatPercent'] = res['shortFloatPercent']
        except:
            item['sharesShort'] = None
            item['shortRatio'] = None
            item['shortOutstandingPercent'] = None
            item['shortFloatPercent'] = None



        try:
            with open(f"json/options-historical-data/companies/{symbol}.json", "r") as file:
                res = orjson.loads(file.read())[0]
                item['ivRank'] = res['iv_rank']
                item['iv30d'] = res['iv']
                item['totalOI'] = res['total_open_interest']
                item['changeOI'] = res['changeOI']
                item['callVolume'] = res['call_volume']
                item['putVolume'] = res['put_volume']
                item['pcRatio'] = res['putCallRatio']
                item['totalPrem'] = res['total_premium']
        except:
            item['ivRank'] = None
            item['iv30d'] = None
            item['totalOI'] = None
            item['changeOI'] = None
            item['callVolume'] = None
            item['putVolume'] = None
            item['pcRatio'] = None
            item['totalPrem'] = None


        try:
            with open(f"json/earnings/raw/{symbol}.json", "r") as file:
                res = orjson.loads(file.read())
                res = sorted(res, key=lambda x: x['date'], reverse=True)
                today_str = datetime.today().strftime('%Y-%m-%d')

                # Try to find an entry with today's date
                today_entry = next((x for x in res if x['date'] == today_str), None)
                next_earning = today_entry if today_entry else res[0]

                item['earningsDate'] = next_earning['date'] if res else None
                if next_earning['eps_est'] != '' and next_earning['revenue_est'] != '':
                    item['earningsEPSEst'] = round(float(next_earning['eps_est']),2)
                    item['earningsRevenueEst'] = round(float(next_earning['revenue_est']),0)

                    eps_prior = float(next_earning['eps_prior'])
                    revenue_prior = round(float(next_earning['revenue_prior']),0)

                    item['earningsEPSGrowthEst'] = round((item['earningsEPSEst']/eps_prior-1)*100,2)
                    item['earningsRevenueGrowthEst'] = round((item['earningsRevenueEst']/revenue_prior-1)*100,2)

                else:
                    item['earningsEPSEst'] = None
                    item['earningsEPSGrowthEst'] = None
                    item['earningsRevenueEst'] = None
                    item['earningsRevenueGrowthEst'] = None

                time = datetime.strptime(next_earning['time'], "%H:%M:%S").time()
                if time < datetime.strptime("09:30:00", "%H:%M:%S").time():
                    item['earningsTime'] = "Before Market Open"
                else:
                    item['earningsTime'] = "After Market Close"
        except:
            item['earningsDate'] = None
            item['earningsTime'] = None
            item['earningsEPSEst'] = None
            item['earningsEPSGrowthEst'] = None
            item['earningsRevenueEst'] = None
            item['earningsRevenueGrowthEst'] = None

        #print(item['earningsEPSEst'],item['earningsEPSGrowthEst'],item['earningsRevenueEst'],item['earningsRevenueGrowthEst'])
        
        try:
            with open(f"json/analyst-estimate/{symbol}.json", 'r') as file:
                res = orjson.loads(file.read())
                
                # Initialize values
                item['forwardPS'] = None
                item['cagrNext3YearEPS'] = None
                item['cagrNext5YearEPS'] = None

                item['cagrNext3YearRevenue'] = None
                item['cagrNext5YearRevenue'] = None
                
                # Create lookup dict by year for easier access
                estimates_by_year = {est['date']: est for est in res}
                
                # Get next year estimate for forward P/S
                next_year = current_year + 1
                if next_year in estimates_by_year:
                    next_year_data = estimates_by_year[next_year]
                    if item['marketCap'] > 0 and next_year_data['estimatedRevenueAvg'] > 0:
                        item['forwardPS'] = round(item['marketCap'] / next_year_data['estimatedRevenueAvg'], 1)
                
                # Calculate Next N-year EPS & RevenueCAGR
                year_3 = current_year + 3
                year_5 = current_year + 5

                if year_3 in estimates_by_year and item['eps']:
                    eps_year_3 = estimates_by_year[year_3]['estimatedEpsAvg']
                    if eps_year_3:
                        item['cagrNext3YearEPS'] = calculate_cagr(item['eps'], eps_year_3, 3)

                if year_5 in estimates_by_year and item['eps']:
                    eps_year_5 = estimates_by_year[year_5]['estimatedEpsAvg']
                    if eps_year_5:
                        item['cagrNext5YearEPS'] = calculate_cagr(item['eps'], eps_year_5, 5)

                if year_3 in estimates_by_year and item['revenue']:
                    revenue_year_3 = estimates_by_year[year_3]['estimatedRevenueAvg']
                    if revenue_year_3:
                        item['cagrNext3YearRevenue'] = calculate_cagr(item['revenue'], revenue_year_3, 3)

                if year_5 in estimates_by_year and item['revenue']:
                    revenue_year_5 = estimates_by_year[year_5]['estimatedRevenueAvg']
                    if revenue_year_5:
                        item['cagrNext5YearRevenue'] = calculate_cagr(item['revenue'], revenue_year_5, 5)

                        
        except:
            item['forwardPS'] = None
            item['cagrNext3YearEPS'] = None
            item['cagrNext5YearEPS'] = None
            item['cagrNext3YearRevenue'] = None
            item['cagrNext5YearRevenue'] = None

        '''
        try:
            item['halalStocks'] = get_halal_compliant(item)
        except:
            item['halalStocks'] = None
        '''

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
        for key in list(item.keys()):
            value = item[key]
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    item[key] = None
            elif isinstance(value, (dict, list)):
                continue
            elif not isinstance(value, (str, int, bool, type(None))):
                try:
                    # Force convert unsupported types to string
                    item[key] = str(value)
                except:
                    item[key] = None

    stock_screener_data = clean_for_json(stock_screener_data)

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

    headers = {"accept": "application/json"}
    url = "https://api.benzinga.com/api/v2.1/calendar/earnings"
    importance_list = ["0", "1", "2", "3", "4", "5"]
    berlin_tz = pytz.timezone('Europe/Berlin')

    today = datetime.now(berlin_tz)
    # Start date: 2 weeks ago, rounded to the previous Monday
    start_date = today - timedelta(weeks=4)
    start_date -= timedelta(days=start_date.weekday())  # Reset to Monday

    # End date: 2 weeks ahead, rounded to the following Friday
    end_date = today + timedelta(weeks=4)
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


async def get_economic_calendar():
    ny_tz = pytz.timezone('America/New_York')
    today = datetime.now(ny_tz)

    start_date = (today - timedelta(weeks=2)).strftime("%Y-%m-%d")
    end_date = (today + timedelta(weeks=4)).strftime("%Y-%m-%d")

    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/stable/economic-calendar?from={start_date}&to={end_date}&apikey={api_key}"
        async with session.get(url) as response:
            data = await response.json()
            print(f"Fetched data: {len(data)} events")
      
    filtered_data = []
    for item in data:
        try:
            country = item['country']
            if country == 'USA' or country == 'US':
                country_code = 'us'
            elif len(country) in (2, 3):
                country_code = country.lower()
            else:
                matching_country = next((c['short'] for c in country_list if c['long'].lower() == country.lower()), None)
                if matching_country:
                    country_code = matching_country.lower()
                else:
                    continue

            impact = item.get('impact', None)
            importance = 3 if impact == 'High' else 2 if impact == 'Medium' else 1

            # Parse date as UTC (naive)
            dt_utc = datetime.strptime(item['date'], "%Y-%m-%d %H:%M:%S")

            # Convert to New York time
            dt_ny = dt_utc - timedelta(hours=5)

            filtered_data.append({
                'countryCode': country_code,
                'country': country,
                'time': dt_ny.strftime("%H:%M"),  
                'date': dt_ny.strftime("%Y-%m-%d"),  
                'prior': item['previous'],  
                'consensus': item['estimate'],  
                'actual': item['actual'],  
                'importance': importance,
                'event': item['event'],
            })
            
        except Exception as e:
            print(f"Error processing item: {e}")

    return filtered_data


async def get_congress_rss_feed(symbols, etf_symbols):

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
    
    res_list = []
    for item in data:
        try:
            ticker = item.get("ticker")
            ticker = ticker.replace('BRK.A','BRK-A')
            ticker = ticker.replace('BRK/A','BRK-A')
            ticker = ticker.replace('BRK.B','BRK-B')
            ticker = ticker.replace('BRK/B','BRK-B')
            
            if ticker in symbols or ticker in etf_symbols:
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

                if 'representative' in item:
                    item['representative'] = replace_representative(item['representative'])

                item['id'] = generate_id(item['representative'])

                # Check if 'district' key exists in item
                if 'district' in item:
                    # Extract state code from the 'district' value
                    state_code = item['district'][:2]
                    
                    # Replace 'district' value with the corresponding value from congressional_districts
                    item['district'] = f"{congressional_districts.get(state_code, state_code)}"

                res_list.append(item)
        except:
            pass

    return res_list




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
    combined_data = []
    query_open_price = """
        SELECT open
        FROM "{ticker}"
        LIMIT 1
    """
    
    # Ensure api_key is defined in your context
    url = f"https://financialmodelingprep.com/stable/ipos-calendar?from=2015-01-01&apikey={api_key}"
    
    async with aiohttp.ClientSession() as session:
        response = await session.get(url)
        data = await response.json()
    
    # In case the API returns a dict with a key like 'ipoCalendar'
    if isinstance(data, dict) and 'ipoCalendar' in data:
        data = data['ipoCalendar']
    
    # Loop directly over the list of IPO items
    for item in data:
        try:
            if (
                item not in combined_data
                and not any(excluded in item.get('company', '') for excluded in ['USD', 'Funds', 'Trust', 'ETF', 'Rupee'])
                and any(item.get('exchange', '').lower() == exchange for exchange in ['nasdaq global', 'nasdaq capital', 'nasdaq global select', 'nyse', 'nasdaq', 'amex'])
            ):
                if item.get('priceRange') is not None:
                    # Assume priceRange is a string like "12.5-15.0"
                    item['priceRange'] = round(float(item['priceRange'].split('-')[0]), 2)
                elif item.get('shares') is not None and item.get('marketCap') is not None:
                    item['priceRange'] = round(item['marketCap'] / item['shares'], 2)
                combined_data.append(item)
        except Exception as e:
            continue

    res = []
    for entry in combined_data:
        try:
            symbol = entry.get('symbol')
            # Try to load local quote data; if it fails, use defaults.
            try:
                with open(f"json/quote/{symbol}.json", "r") as file:
                    quote_data = ujson.load(file)
            except Exception as e:
                quote_data = {'price': None, 'changesPercentage': None}
            
            entry['currentPrice'] = quote_data.get('price', None)
            try:
                query = query_open_price.format(ticker=entry['symbol'])
                df = pd.read_sql_query(query, con)
                if not df.empty and df['open'].iloc[0]:
                    entry['ipoPrice'] = round(df['open'].iloc[0], 2)
                else:
                    entry['ipoPrice'] = None
            except Exception as e:
                # Fallback to calculated priceRange if SQL fails
                entry['ipoPrice'] = round(entry['priceRange'], 2) if entry.get('priceRange') is not None else None

            if entry['ipoPrice'] is not None:
                try:
                    if entry['ipoPrice'] in (0, None) or entry['currentPrice'] in (0, None):
                        entry['return'] = None
                    else:
                        entry['return'] = round(((entry['currentPrice'] / entry['ipoPrice'] - 1) * 100), 2)
                except Exception as e:
                    entry['return'] = None

                required_fields = ("ipoPrice", "currentPrice", "return")

                with open(f"json/one-day-price/{symbol}.json","rb") as file:
                    daily_price_history = orjson.loads(file.read())

                with open(f"json/historical-price/adj/{symbol}.json","rb") as file:
                    total_price_history = orjson.loads(file.read())

                if all(entry.get(field) is not None for field in required_fields) and len(daily_price_history) > 0 and len(total_price_history) > 0:
                    res.append({
                        "symbol": symbol,
                        "name": entry.get("company"),
                        "ipoDate": entry.get("date"),
                        "ipoPrice": entry["ipoPrice"],
                        "currentPrice": round(entry["currentPrice"], 2),
                        "return": entry["return"],
                    })

        except Exception as e:
            continue

    # Sort results by ipoDate descending
    res_sorted = sorted(res, key=lambda x: x.get('ipoDate'), reverse=True)

    return res_sorted


def save_json(data, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(f"{directory}/data.json","w") as file:
        ujson.dump(data, file)


def save_json(data, directory):
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the data to the file
    with open(f"{directory}/data.json", "w") as file:
        ujson.dump(data, file)

async def save_json_files():
    con = sqlite3.connect('stocks.db')
    etf_con = sqlite3.connect('etf.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    symbols = [row[0] for row in cursor.fetchall()]

    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]


    # Save IPO calendar
    data = await get_ipo_calendar(con, symbols)
    save_json(data, "json/ipo-calendar")

    # Save stock screener data
    stock_screener_data = await get_stock_screener(con)
    save_json(stock_screener_data, "json/stock-screener")
    


    # Save economic calendar
    economic_list = await get_economic_calendar()
    if len(economic_list) > 0:
        save_json(economic_list, "json/economic-calendar")

    
    # Save congress trading data
    data = await get_congress_rss_feed(symbols, etf_symbols)
    save_json(data, "json/congress-trading/rss-feed")
    
    # Save earnings calendar
    earnings_list = await get_earnings_calendar(con, symbols)
    save_json(earnings_list, "json/earnings-calendar")

    # Save dividends calendar
    dividends_list = await get_dividends_calendar(con, symbols)
    save_json(dividends_list, "json/dividends-calendar")

    # Save ETF providers data
    data = await etf_providers(etf_con, etf_symbols)
    save_json(data, "json/all-etf-providers")
    


    # Close connections
    con.close()
    etf_con.close() 
try:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(save_json_files())
except Exception as e:
    print(e)