import os
import asyncio
import aiofiles
import orjson
import sqlite3
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar, Set, Tuple, cast
from rapidfuzz import process, fuzz
import aiohttp
import re

from agents import function_tool

load_dotenv()

api_key = os.getenv('FMP_API_KEY')

# Source metadata mapping for each function
# This defines user-friendly names, descriptions, and URL patterns for each function
FUNCTION_SOURCE_METADATA = {
    # Quote and Price Data
    "get_ticker_quote": {
        "name": "Real-Time Market Data",
        "description": "Live stock price, volume, and market metrics",
        "url_pattern": "/{asset_type}/{ticker}"
    },
    "get_ticker_pre_post_quote": {
        "name": "Pre/Post Market Data", 
        "description": "Extended hours trading data",
        "url_pattern": "/{asset_type}/{ticker}"
    },
    
    # Financial Statements
    "get_ticker_balance_sheet_statement": {
        "name": "Balance Sheet",
        "description": "Assets, liabilities, and equity data",
        "url_pattern": "/{asset_type}/{ticker}/financials/balance-sheet"
    },
    "get_ticker_income_statement": {
        "name": "Income Statement", 
        "description": "Revenue, expenses, and profit data",
        "url_pattern": "/{asset_type}/{ticker}/financials"
    },
    "get_ticker_cash_flow_statement": {
        "name": "Cash Flow Statement",
        "description": "Operating, investing, and financing cash flows",
        "url_pattern": "/{asset_type}/{ticker}/financials/cash-flow"
    },
    "get_ticker_ratios_statement": {
        "name": "Financial Ratios",
        "description": "Profitability, liquidity, and efficiency ratios",
        "url_pattern": "/{asset_type}/{ticker}/financials/ratios"
    },
    
    # Analyst Data
    "get_ticker_analyst_rating": {
        "name": "Analyst Ratings",
        "description": "Buy/sell recommendations from analysts",
        "url_pattern": "/{asset_type}/{ticker}/forecast/analyst"
    },
    "get_ticker_analyst_estimate": {
        "name": "Analyst Estimates",
        "description": "Earnings and revenue forecasts",
        "url_pattern": "/{asset_type}/{ticker}/forecast"
    },
    
    # Company Information
    "get_ticker_news": {
        "name": "Company News",
        "description": "Latest news and press releases",
        "url_pattern": "/{asset_type}/{ticker}"
    },
    "get_ticker_insider_trading": {
        "name": "Insider Trading",
        "description": "Insider buying and selling activity",
        "url_pattern": "/{asset_type}/{ticker}/insider"
    },
    "get_ticker_shareholders": {
        "name": "Shareholder Information",
        "description": "Institutional and insider ownership",
        "url_pattern": "/{asset_type}/{ticker}/insider/institute"
    },
    "get_ticker_earnings": {
        "name": "Earnings Reports",
        "description": "Quarterly earnings and guidance",
        "url_pattern": "/{asset_type}/{ticker}"
    },
    "get_ticker_dividend": {
        "name": "Dividend Information", 
        "description": "Dividend payments and yield history",
        "url_pattern": "/{asset_type}/{ticker}/dividends"
    },
    "get_ticker_statistics": {
        "name": "Company Statistics",
        "description": "Key financial and trading metrics",
        "url_pattern": "/{asset_type}/{ticker}/statistics"
    },
    "get_ticker_key_metrics": {
        "name": "Key Metrics",
        "description": "Financial performance indicators",
        "url_pattern": "/{asset_type}/{ticker}/statistics"
    },    
    "get_ticker_options_overview_data": {
        "name": "Options Overview",
        "description": "Options chain and volatility data", 
        "url_pattern": "/{asset_type}/{ticker}/options"
    },
    "get_ticker_dark_pool": {
        "name": "Dark Pool Activity",
        "description": "Off-exchange trading volumes",
        "url_pattern": "/{asset_type}/{ticker}/dark-pool"
    },
    "get_ticker_unusual_activity": {
        "name": "Unusual Options Activity",
        "description": "Notable options volume and flow",
        "url_pattern": "/{asset_type}/{ticker}/options/unusual-activity"
    },
    "get_ticker_max_pain": {
        "name": "Options Max Pain",
        "description": "Price level causing maximum loss",
        "url_pattern": "/{asset_type}/{ticker}/options/max-pain"
    },
    "get_ticker_open_interest_by_strike_and_expiry": {
        "name": "Options Open Interest",
        "description": "Contract positions by strike and expiry",
        "url_pattern": "/{asset_type}/{ticker}/options/oi"
    },
    
    # Market Data
    "get_market_news": {
        "name": "Market News",
        "description": "General market news and updates",
        "url_pattern": "/market-news/general"
    },
    "get_economic_calendar": {
        "name": "Economic Calendar", 
        "description": "Upcoming economic events and indicators",
        "url_pattern": "/economic-calendar"
    },
    "get_earnings_releases": {
        "name": "Earnings Calendar",
        "description": "Upcoming earnings announcements", 
        "url_pattern": "/earnings-calendar"
    },
    "get_dividend_calendar": {
        "name": "Dividend Calendar",
        "description": "Upcoming dividend payments",
        "url_pattern": "/dividends-calendar"
    },
    
    # Trading Activity
    "get_congress_activity": {
        "name": "Congressional Trading",
        "description": "Stock trades by members of Congress",
        "url_pattern": "/politicians"
    },
    "get_insider_tracker": {
        "name": "Insider Tracker", 
        "description": "Insider trading across all companies",
        "url_pattern": "/insider-tracker"
    },
    "get_analyst_tracker": {
        "name": "Analyst Tracker",
        "description": "Analyst recommendations and changes", 
        "url_pattern": "/analysts/analyst-data"
    },
    "get_market_flow": {
        "name": "Market Flow",
        "description": "Real-time market sentiment and flows",
        "url_pattern": "/market-flow"
    },
    
    # Market Movers
    "get_top_gainers": {
        "name": "Top Gainers",
        "description": "Best performing stocks today",
        "url_pattern": "/market-mover/gainers"
    },
    "get_top_losers": {
        "name": "Top Losers", 
        "description": "Worst performing stocks today",
        "url_pattern": "/market-mover/losers"
    },
    "get_latest_options_flow_feed": {
        "name": "Options Flow",
        "description": "Real-time options trading activity",
        "url_pattern": "/options-flow"
    },
    "get_latest_dark_pool_feed": {
        "name": "Dark Pool Flow", 
        "description": "Off-exchange trading activity",
        "url_pattern": "/dark-pool-flow"
    },
    "get_company_data": {
        "name": "Company Overview",
        "description": "Basic company information and profile data",
        "url_pattern": "/{asset_type}/{ticker}/profile"
    },
    "get_why_priced_moved": {
        "name": "Why Priced Moved", 
        "description": "Reason of recent price changes",
        "url_pattern": "/{asset_type}/{ticker}"
    },
    "get_ticker_business_metrics": {
        "name": "Business Metrics",
        "description": "Key business performance indicators",
        "url_pattern": "/{asset_type}/{ticker}/metrics"
    },
    "get_ticker_bull_vs_bear": {
        "name": "Bull vs Bear Analysis",
        "description": "Bullish and bearish sentiment analysis",
        "url_pattern": "/{asset_type}/{ticker}"
    },
    "get_ticker_hottest_options_contracts": {
        "name": "Hot Options Contracts",
        "description": "Most active options contracts",
        "url_pattern": "/{asset_type}/{ticker}/options/hottest-contracts"
    },
    "get_monthly_dividend_stocks": {
        "name": "Monthly Dividend Stocks",
        "description": "Stocks That Pay Monthly Dividends ",
        "url_pattern": "/list/monthly-dividend-stocks"
    },
    "get_ticker_short_data": {
        "name": "Short Interest Data",
        "description": "Short selling metrics and statistics",
        "url_pattern": "/{asset_type}/{ticker}/statistics/short-interest"
    },
    "get_ticker_earnings_price_reaction": {
        "name": "Earnings Price Reaction",
        "description": "Stock price movement after earnings",
        "url_pattern": "/{asset_type}/{ticker}/statistics/price-reaction"
    },
    "get_top_rating_stocks": {
        "name": "Top Rated Stocks",
        "description": "Highest rated stocks by analysts",
        "url_pattern": "/analysts/top-stocks"
    },
    "get_top_premarket_gainers": {
        "name": "Pre-Market Gainers",
        "description": "Top gaining stocks in pre-market",
        "url_pattern": "/market-mover/premarket/gainers"
    },
    "get_top_aftermarket_gainers": {
        "name": "After-Market Gainers", 
        "description": "Top gaining stocks in after-hours",
        "url_pattern": "/market-mover/aftermarket/gainers"
    },
    "get_top_premarket_losers": {
        "name": "Pre-Market Losers",
        "description": "Top losing stocks in pre-market",
        "url_pattern": "/market-mover/premarket/losers"
    },
    "get_top_aftermarket_losers": {
        "name": "After-Market Losers",
        "description": "Top losing stocks in after-hours", 
        "url_pattern": "/market-mover/aftermarket/losers"
    },
    "get_top_active_stocks": {
        "name": "Most Active Stocks",
        "description": "Stocks with highest trading volume",
        "url_pattern": "/market-mover/active"
    },
    "get_potus_tracker": {
        "name": "POTUS Tracker",
        "description": "Presidential stock trading activity",
        "url_pattern": "/potus-tracker"
    },
    "get_latest_congress_trades": {
        "name": "Latest Congress Trades",
        "description": "Recent congressional stock transactions",
        "url_pattern": "/politicians/flow-data"
    },
    "get_ticker_financial_score": {
        "name": "Financial Score",
        "description": "Overall financial health rating",
        "url_pattern": "/{asset_type}/{ticker}/statistics"
    },
    "get_ticker_owner_earnings": {
        "name": "Owner Earnings",
        "description": "Berkshire-style owner earnings calculation",
        "url_pattern": "/{asset_type}/{ticker}/statistics"
    },
    "get_oversold_tickers": {
        "name": "Oversold Stocks",
        "description": "Stocks with oversold technical indicators",
        "url_pattern": "/list/oversold-stocks"
    },
    "get_overbought_tickers": {
        "name": "Overbought Stocks", 
        "description": "Stocks with overbought technical indicators",
        "url_pattern": "/list/overbought-stocks"
    },
    "get_dividend_kings": {
        "name": "Dividend Kings",
        "description": "Stocks with 50+ years of dividend increases",
        "url_pattern": "/list/dividend/dividend-kings"
    },
    "get_dividend_aristocrats": {
        "name": "Dividend Aristocrats",
        "description": "S&P 500 stocks with 25+ years of dividend increases",
        "url_pattern": "/list/dividend/dividend-aristocrats"
    },
    "get_top_rated_dividend_stocks": {
        "name": "Top Dividend Stocks",
        "description": "Highest rated dividend paying stocks",
        "url_pattern": "/list/top-rated-dividend-stocks"
    },
    "get_ticker_trend_forecast": {
        "name": "Trend Forecast",
        "description": "Technical analysis and trend predictions",
        "url_pattern": "/{asset_type}/{ticker}/forecast/ai"
    },
    "get_ipo_calendar": {
        "name": "IPO Calendar",
        "description": "Upcoming initial public offerings",
        "url_pattern": "/ipos"
    },
    "get_penny_stocks": {
        "name": "Penny Stocks",
        "description": "Low-priced stocks under $5",
        "url_pattern": "/list/penny-stocks"
    },
    "get_most_shorted_stocks": {
        "name": "Most Shorted Stocks",
        "description": "Stocks with highest short interest",
        "url_pattern": "/list/most-shorted-stocks"
    },
    "get_bitcoin_etfs": {
        "name": "Bitcoin ETFs",
        "description": "Bitcoin and cryptocurrency ETFs",
        "url_pattern": "/list/bitcoin-etfs"
    },
    "get_all_sector_overview": {
        "name": "Sector Overview",
        "description": "Performance overview of all market sectors",
        "url_pattern": "/industry/sectors"
    },
    "get_ticker_earnings_call_transcripts": {
        "name": "Earnings Call Transcripts",
        "description": "Quarterly earnings call transcripts and analysis",
        "url_pattern": "/{asset_type}/{ticker}/insider/transcripts"
    },
    "get_stock_screener": {
        "name": "Stock Screener",
        "description": "Filter stocks by financial criteria and metrics",
        "url_pattern": "/stock-screener"
    },
    "get_fear_and_greed_index": {
        "name": "Fear & Greed Index",
        "description": "Track market sentiment based on fear and greed signals",
        "url_pattern": "/fear-and-greed"
    },
    "get_reddit_tracker": {
        "name": "Reddit Tracker",
        "description": "Retrieve trending tickers from WallStreetBets.",
        "url_pattern": "/reddit-tracker"
    },
}


con = sqlite3.connect('stocks.db')

cursor = con.cursor()
cursor.execute("PRAGMA journal_mode = wal")
cursor.execute("SELECT DISTINCT symbol FROM stocks")
stock_symbols = [row[0] for row in cursor.fetchall()]


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
"dividendPerShare",
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


key_screener = [
  "avgVolume",
  "volume",
  "rsi",
  "stochRSI",
  "mfi",
  "cci",
  "atr",
  "sma20",
  "sma50",
  "sma100",
  "sma200",
  "ema20",
  "ema50",
  "ema100",
  "ema200",
  "grahamNumber",
  "grahamUpside",
  "price",
  "change1W",
  "change1M",
  "change3M",
  "change6M",
  "change1Y",
  "change3Y",
  "marketCap",
  "workingCapital",
  "totalAssets",
  "tangibleAssetValue",
  "revenue",
  "revenueGrowthYears",
  "epsGrowthYears",
  "netIncomeGrowthYears",
  "grossProfitGrowthYears",
  "growthRevenue",
  "costOfRevenue",
  "growthCostOfRevenue",
  "costAndExpenses",
  "growthCostAndExpenses",
  "netIncome",
  "growthNetIncome",
  "grossProfit",
  "growthGrossProfit",
  "researchAndDevelopmentExpenses",
  "growthResearchAndDevelopmentExpenses",
  "payoutRatio",
  "dividendYield",
  "payoutFrequency",
  "annualDividend",
  "dividendGrowth",
  "eps",
  "growthEPS",
  "interestIncome",
  "interestExpense",
  "growthInterestExpense",
  "operatingExpenses",
  "growthOperatingExpenses",
  "ebit",
  "operatingIncome",
  "growthOperatingIncome",
  "growthFreeCashFlow",
  "growthOperatingCashFlow",
  "growthStockBasedCompensation",
  "growthTotalLiabilities",
  "growthTotalDebt",
  "growthTotalStockholdersEquity",
  "researchDevelopmentRevenueRatio",
  "cagr3YearRevenue",
  "cagr5YearRevenue",
  "cagr3YearEPS",
  "cagr5YearEPS",
  "returnOnInvestedCapital",
  "returnOnCapitalEmployed",
  "relativeVolume",
  "institutionalOwnership",
  "priceToEarningsGrowthRatio",
  "forwardPE",
  "forwardPS",
  "priceToBookRatio",
  "priceToSalesRatio",
  "beta",
  "ebitda",
  "growthEBITDA",
  "var",
  "currentRatio",
  "quickRatio",
  "debtToEquityRatio",
  "inventoryTurnover",
  "returnOnAssets",
  "returnOnEquity",
  "returnOnTangibleAssets",
  "enterpriseValue",
  "evToSales",
  "evToEBITDA",
  "evToEBIT",
  "evToFCF",
  "freeCashFlowPerShare",
  "cashPerShare",
  "priceToFreeCashFlowRatio",
  "interestCoverageRatio",
  "sharesShort",
  "shortRatio",
  "shortFloatPercent",
  "shortOutstandingPercent",
  "failToDeliver",
  "relativeFTD",
  "freeCashFlow",
  "operatingCashFlow",
  "operatingCashFlowPerShare",
  "revenuePerShare",
  "netIncomePerShare",
  "shareholdersEquityPerShare",
  "interestDebtPerShare",
  "capexPerShare",
  "freeCashFlowMargin",
  "totalDebt",
  "operatingCashFlowSalesRatio",
  "priceToOperatingCashFlowRatio",
  "priceToEarningsRatio",
  "stockBasedCompensation",
  "stockBasedCompensationToRevenue",
  "totalStockholdersEquity",
  "sharesQoQ",
  "sharesYoY",
  "grossProfitMargin",
  "netProfitMargin",
  "pretaxProfitMargin",
  "ebitdaMargin",
  "ebitMargin",
  "operatingMargin",
  "interestIncomeToCapitalization",
  "assetTurnover",
  "earningsYield",
  "freeCashFlowYield",
  "effectiveTaxRate",
  "fixedAssetTurnover",
  "sharesOutStanding",
  "employees",
  "revenuePerEmployee",
  "profitPerEmployee",
  "totalLiabilities",
  "altmanZScore",
  "piotroskiScore",
  'floatShares',
    'evToOperatingCashFlow',
    'evToFreeCashFlow',
    'debtToFreeCashFlowRatio',
    'debtToEBITDARatio',
    'cashFlowToDebtRatio',
    'debtToMarketCap',
    'operatingProfitMargin',
    'cashAndCashEquivalents',
    'retainedEarnings',
    'capitalExpenditure',
    'lastStockSplit',
    'splitType',
    'splitRatio',
    'analystRating',
    'analystCounter',
    'priceTarget',
    'upside',
    'incomeTaxExpense',
    'lynchFairValue',
    'lynchUpside'
]





key_metrics = [
    "marketCap",
    "enterpriseValueTTM",
    "evToSalesTTM",
    "evToOperatingCashFlowTTM",
    "evToFreeCashFlowTTM",
    "evToEBITDATTM",
    "netDebtToEBITDATTM",
    "currentRatioTTM",
    "incomeQualityTTM",
    "grahamNumberTTM",
    "grahamNetNetTTM",
    "taxBurdenTTM",
    "interestBurdenTTM",
    "workingCapitalTTM",
    "investedCapitalTTM",
    "returnOnAssetsTTM",
    "operatingReturnOnAssetsTTM",
    "returnOnTangibleAssetsTTM",
    "returnOnEquityTTM",
    "returnOnInvestedCapitalTTM",
    "returnOnCapitalEmployedTTM",
    "earningsYieldTTM",
    "freeCashFlowYieldTTM",
    "capexToOperatingCashFlowTTM",
    "capexToDepreciationTTM",
    "capexToRevenueTTM",
    "salesGeneralAndAdministrativeToRevenueTTM",
    "researchAndDevelopementToRevenueTTM",
    "stockBasedCompensationToRevenueTTM",
    "intangiblesToTotalAssetsTTM",
    "averageReceivablesTTM",
    "averagePayablesTTM",
    "averageInventoryTTM",
    "daysOfSalesOutstandingTTM",
    "daysOfPayablesOutstandingTTM",
    "daysOfInventoryOutstandingTTM",
    "operatingCycleTTM",
    "cashConversionCycleTTM",
    "freeCashFlowToEquityTTM",
    "freeCashFlowToFirmTTM",
    "tangibleAssetValueTTM",
    "netCurrentAssetValueTTM"
]

key_owner_earnings = [
    "fiscalYear",
    "period",
    "date",
    "averagePPE",
    "maintenanceCapex",
    "ownersEarnings",
    "growthCapex",
    "ownersEarningsPerShare"
]



# Type variables for better typing
T = TypeVar('T')
JsonDict = Dict[str, Any]
TickerData = Dict[str, List[JsonDict]]

# Constants
DEFAULT_FINANCIAL_REMOVE_KEYS: Set[str] = {"symbol", "reportedCurrency", "acceptedDate", "cik", "filingDate"}
FINANCIAL_REQUIRED_KEYS: Set[str] = {"date", "fiscalYear", "period"}
STATEMENT_DIRS: Dict[str, str] = {
    "income": "income-statement",
    "balance": "balance-sheet-statement",
    "cash": "cash-flow-statement",
}

# Dynamic date variables
current_year = datetime.now().year
week_ago = datetime.now().date() - timedelta(days=5)
today = date.today()

# Operator mapping for stock screener
OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    '>': lambda x, y: x > y,
    '>=': lambda x, y: x >= y,
    '<': lambda x, y: x < y,
    '<=': lambda x, y: x <= y,
    '==': lambda x, y: x == y,
    '!=': lambda x, y: x != y
}

# Common base directory
BASE_DIR = Path("json")

def load_congress_db():
    data = {}
    directory = "json/congress-trading/politician-db/"
    
    try:
        files = os.listdir(directory)
        json_files = [f for f in files if f.endswith('.json')]
        
        for filename in json_files:
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "rb") as file:
                    file_data = orjson.loads(file.read())
                    
                    if 'history' in file_data and len(file_data['history']) > 0:
                        politician_id = file_data['history'][0]['id']
                        name = file_data['history'][0]['office']
                        data[name] = politician_id
                        
            except (KeyError, IndexError, orjson.JSONDecodeError) as e:
                print(f"Error processing {filename}: {e}")
                continue
                
    except FileNotFoundError:
        print(f"Directory {directory} not found")
    return data

key_congress_db = load_congress_db()


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


# Generic file fetching function
async def fetch_ticker_data(ticker: str, base_dir: Path) -> Optional[Any]:
    """Generic function to fetch data for a ticker from a JSON file."""
    file_path = base_dir / f"{ticker}.json"
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            content = await f.read()
            return orjson.loads(content)
    except FileNotFoundError:
        return None
    except (orjson.JSONDecodeError, KeyError) as e:
        print(f"Error processing {ticker}: {e}")
        return None

async def _load_and_filter(
    file_path: Path,
    keep_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load JSON data from file and filter keys based on criteria."""
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            raw = orjson.loads(await f.read())
            
        if not raw:
            return []
            
        if keep_keys:
            keys_to_keep = set(keep_keys) | FINANCIAL_REQUIRED_KEYS
            return [
                {k: v for k, v in entry.items() if k in keys_to_keep} 
                for entry in raw
            ]
        else:
            return [
                {k: v for k, v in entry.items() if k not in DEFAULT_FINANCIAL_REMOVE_KEYS} 
                for entry in raw
            ]
    except Exception as e:
        print(f"Error loading/filtering file {file_path}: {e}")
        return []

# Generic function for fetching ticker-specific data
async def get_ticker_specific_data(
    tickers: List[str], 
    base_path: str,
    process_func: Optional[Callable[[Any], Any]] = None
) -> Dict[str, Any]:
    """
    Generic function to fetch and process data for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        base_path: Path to the data directory
        process_func: Optional function to process each ticker's data
        
    Returns:
        Dictionary mapping tickers to their processed data
    """
    base_dir = BASE_DIR / base_path
    
    # Create tasks for each ticker
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    
    # Gather all results concurrently
    results = await asyncio.gather(*tasks)
    
    # Process results if needed
    filtered_results = {}
    for ticker, result in zip(tickers, results):
        if result is not None:
            filtered_results[ticker] = process_func(result) if process_func else result
    
    return filtered_results

async def get_financial_statements(
    tickers: List[str],
    statement_type: str = "income",
    time_period: str = "annual",
    keep_keys: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generic function to retrieve financial statements for multiple companies.
    
    Args:
        tickers: List of stock ticker symbols
        statement_type: Type of statement ('income', 'balance', 'cash')
        time_period: Period of data ('annual', 'quarter', 'ttm')
        keep_keys: List of specific keys to keep in the output
        
    Returns:
        Dictionary mapping tickers to their financial statement data
    """
    dir_name = STATEMENT_DIRS.get(statement_type)
    if not dir_name:
        raise ValueError(f"Invalid statement_type '{statement_type}'. Must be one of {list(STATEMENT_DIRS)}.")

    base_dir = BASE_DIR / "financial-statements" / dir_name / time_period
    
    # Create all tasks at once for efficient concurrency
    tasks = [_load_and_filter(base_dir / f"{ticker}.json", keep_keys) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    # Build result dictionary with non-empty results
    return {ticker: result for ticker, result in zip(tickers, results) if result}


@function_tool
async def get_ticker_income_statement(
    tickers: List[str], time_period: str = "ttm", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves historical income statements (profit and loss reports) for one or more stock tickers.

    This includes key financial metrics such as: revenue, gross profit, operating income, net income, and more.
    Data is available by fiscal year (annual), fiscal quarter (quarter), or trailing twelve months (ttm).

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).
        time_period (str): Time range for the statements: 
            - "annual": Returns year-by-year data.
            - "quarter": Returns data for each fiscal quarter (e.g., Q1, Q2, etc.).
            - "ttm": Returns trailing twelve months data.
        keep_keys (Optional[List[str]]): A list of specific financial metrics to return (e.g., ["revenue", "netIncome"]).
            If None, all available fields will be returned.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of historical income statement records.
            The list is ordered from most recent to oldest.
            Each record is a dictionary with financial metrics for that specific period.

    Note:
        To answer questions like "What is Apple's revenue for the first quarter?", call this function with:
            - tickers=["AAPL"]
            - time_period="quarter"
            Then extract the revenue from the most recent Q1 entry.
    """
    return await get_financial_statements(tickers, "income", time_period, keep_keys)


@function_tool
async def get_ticker_balance_sheet_statement(
    tickers: List[str], time_period: str = "ttm", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves historical balance sheet statements for one or more stock tickers.

    This includes key financial metrics such as: total assets, total liabilities, shareholders' equity, cash, inventory, and more.
    Data is available by fiscal year ("annual"), fiscal quarter ("quarter"), or trailing twelve months ("ttm").

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).
        time_period (str): Time range for the statements:
            - "annual": Returns data for each fiscal year.
            - "quarter": Returns data for each fiscal quarter (e.g., Q1, Q2, etc.).
            - "ttm": Returns data for the trailing twelve months.
        keep_keys (Optional[List[str]]): A list of specific balance sheet metrics to include 
            (e.g., ["totalAssets", "totalLiabilities"]). If None, all available fields will be returned.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of historical balance sheet records.
            The list is ordered from most recent to oldest.
            Each record is a dictionary with financial metrics for that specific period.

    Note:
        To answer questions like "What are Apple's total assets in the first quarter?", call this function with:
            - tickers=["AAPL"]
            - time_period="quarter"
        Then extract the "totalAssets" field from the most recent Q1 entry.
    """
    return await get_financial_statements(tickers, "balance", time_period, keep_keys)



@function_tool
async def get_ticker_cash_flow_statement(
    tickers: List[str], time_period: str = "ttm", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves historical cash flow statements for one or more stock tickers.

    This includes key cash flow metrics such as: operating cash flow, investing cash flow, financing cash flow, 
    capital expenditures, and free cash flow. Data is available by fiscal year ("annual"), fiscal quarter ("quarter"), 
    or trailing twelve months ("ttm").

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).
        time_period (str): Time range for the statements:
            - "annual": Returns data for each fiscal year.
            - "quarter": Returns data for each fiscal quarter (e.g., Q1, Q2, etc.).
            - "ttm": Returns data for the trailing twelve months.
        keep_keys (Optional[List[str]]): A list of specific cash flow metrics to include 
            (e.g., ["operatingCashFlow", "freeCashFlow"]). If None, all available fields will be returned.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of historical cash flow records.
            The list is ordered from most recent to oldest.
            Each record is a dictionary with financial metrics for that specific period.

    Note:
        To answer questions like "What is Apple's free cash flow in Q1?", call this function with:
            - tickers=["AAPL"]
            - time_period="quarter"
        Then extract the "freeCashFlow" field from the most recent Q1 entry.
    """
    return await get_financial_statements(tickers, "cash", time_period, keep_keys)



@function_tool
async def get_ticker_ratios_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves historical financial ratios for one or more stock tickers.

    Includes key metrics such as: return on equity (ROE), return on assets (ROA), debt-to-equity ratio, 
    current ratio, price-to-earnings (P/E), and more. Available for both annual and quarterly periods.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).
        time_period (str): Time period for the data:
            - "annual": Returns year-by-year ratio data.
            - "quarter": Returns data for each fiscal quarter.
        keep_keys (Optional[List[str]]): A list of specific ratio metrics to return 
            (e.g., ["roe", "peRatio"]). If None, all available fields will be returned.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of financial ratio records.
            The list is ordered from most recent to oldest.
            Each record is a dictionary with ratio values for that period.

    Raises:
        ValueError: If `time_period` is not "annual" or "quarter".

    Note:
        To answer questions like "What is Apple's return on equity for the latest quarter?", use:
            - tickers=["AAPL"]
            - time_period="quarter"
        Then extract the "roe" field from the most recent entry.
    """
    if time_period not in ["annual", "quarter"]:
        raise ValueError(f"Invalid time_period '{time_period}'. For ratios, must be 'annual' or 'quarter'.")

    base_dir = BASE_DIR / "financial-statements/ratios" / time_period
    tasks = [_load_and_filter(base_dir / f"{ticker}.json", keep_keys) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    return {ticker: result for ticker, result in zip(tickers, results) if result}




@function_tool
async def get_ticker_hottest_options_contracts(
    tickers: List[str],
    category: str = "volume"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves the hottest options contracts for stock tickers.

    Parameters:
    - tickers: List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).
    - category: Category to sort contracts by. Must be either "volume" or "openInterest".
    """
    if category not in ["volume", "openInterest"]:
        raise ValueError(f"Invalid category '{category}'. Must be 'volume' or 'openInterest'.")

    base_dir = BASE_DIR / "hottest-contracts/companies"

    # Create tasks for each ticker
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]

    # Gather all results concurrently
    results = await asyncio.gather(*tasks)

    # Return top 5 for each ticker
    return {
        ticker: result[category][:10]
        for ticker, result in zip(tickers, results)
        if result is not None and category in result
    }

@function_tool
async def get_company_data(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch financial and organizational overview data for multiple companies.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping each ticker to its cleaned company data.
    """
    def process_company_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove specific fields from company data."""
        result = data.copy()
        result.pop("website", None)
        result.pop("financialPerformance", None)
        return result

    return await get_ticker_specific_data(tickers, "stockdeck", process_company_data)


@function_tool
async def get_ticker_short_data(tickers: List[str]) -> Dict[str, Any]:
    """
    Retrieve the most recent and historical short interest data for multiple companies.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Any]: Dictionary mapping each ticker to its short interest data.
    """
    return await get_ticker_specific_data(tickers, "share-statistics")


@function_tool
async def get_why_priced_moved(tickers: List[str]) -> Dict[str, Any]:
    """
    Retrieve recent news explaining the price movement of multiple stocks based on their ticker symbols.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Any]: Dictionary mapping each ticker to the corresponding price movement explanation data.
    """
    return await get_ticker_specific_data(tickers, "wiim/company")


@function_tool
async def get_ticker_business_metrics(tickers: List[str]) -> Dict[str, Any]:
    """
    Fetch business metrics for multiple stocks, including revenue breakdown by sector and geographic region.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Any]: Dictionary mapping each ticker to its business metrics.
    """
    return await get_ticker_specific_data(tickers, "business-metrics")


@function_tool
async def get_ticker_analyst_estimate(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch forward-looking analyst estimates for multiple stocks, including average, low, and high projections
    for EPS, revenue, EBITDA, and net income. Call this function when 'Analyst' is mentioned in the query.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping each ticker to its filtered analyst estimates.
    """
    def filter_by_year(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [entry for entry in data if entry.get("date", 0) >= current_year]
    
    result = await get_ticker_specific_data(tickers, "analyst-estimate", filter_by_year)
    return {ticker: data for ticker, data in result.items() if data}


@function_tool
async def get_earnings_releases() -> List[Dict[str, Any]]:
    """
    Retrieve today/upcoming company earnings releases/announcements with key financial metrics.

    Use this function when the user asks about:
      - Which companies have earnings today, tomorrow, or on a specific date
      - Companies reporting earnings "BMO" (Before Market Opens) or "AMC" (After Market Close)
      - EPS estimates or revenue forecasts for upcoming earnings
      - Market cap or historical EPS/revenue comparisons for earnings events

    Returns:
        List[Dict[str, Any]]: A list of upcoming earnings events starting from today.
        Each dictionary contains:
          - symbol (str): Company ticker symbol
          - name (str): Company name
          - date (str): Scheduled earnings date in YYYY-MM-DD format
          - marketCap (float): Market capitalization in USD
          - epsPrior (float): EPS from the previous comparable period
          - epsEst (float): Analyst estimated EPS for the upcoming report
          - revenuePrior (float): Revenue from the previous comparable period in USD
          - revenueEst (float): Analyst estimated revenue for the upcoming report in USD
          - release (str): "BMO" for Before Market Opens, "AMC" for After Market Close
    """
    file_path = BASE_DIR / "earnings-calendar/data.json"
    today = datetime.today().date()
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            data = orjson.loads(await f.read())
        
        # Filter for events from today onwards
        filtered_data = []
        
        # Sort by date to ensure today's earnings appear first
        filtered_data.sort(key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d").date())
        
        return filtered_data[:20]
    except FileNotFoundError:
        return []
    except (orjson.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing earnings calendar: {e}")
        return []


@function_tool
async def get_economic_calendar() -> List[Dict[str, Any]]:
    """
    Retrieve a list of upcoming USA economic events for macroeconomic analysis, including event name, date, time,
    previous/consensus/actual values (if available), event importance, and associated country code.

    Returns:
        List[Dict[str, Any]]: Filtered list of upcoming US economic calendar events (max 20 items).
    """
    file_path = BASE_DIR / "economic-calendar/data.json"
    today = datetime.today().date()
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            data = orjson.loads(await f.read())
            data = [item for item in data if item['countryCode'] == 'us']
        return [item for item in data if today <= datetime.strptime(item['date'], "%Y-%m-%d").date()][:20]
    except FileNotFoundError:
        return []
    except (orjson.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing economic calendar: {e}")
        return []


@function_tool
async def get_top_rating_stocks() -> list[dict]:
    """
    Retrieves the top rating stocks from analysts.
    Returns a list of stocks with the highest analyst ratings.

    Returns:
        list[dict]: A list of dictionaries containing top rated stocks from analysts.
    """
    file_path = BASE_DIR / "analyst/top-stocks.json"
    
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            data = orjson.loads(await f.read())
        return data
    except FileNotFoundError:
        return []
    except (orjson.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing get top rating stocks: {e}")
        return []

@function_tool
async def get_ticker_earnings_price_reaction(tickers: List[str]) -> Dict[str, Any]:
    """
    Fetch past earnings price reactions before and after earnings releases of multiple stocks based on their ticker symbols.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Any]: Dictionary mapping each ticker to its earnings reaction history.
    """
    return await get_ticker_specific_data(tickers, "earnings/past")


@function_tool
async def get_ticker_earnings(tickers: List[str]) -> Dict[str, Any]:
    """
    Retrieve the historical, latest, and upcoming earnings dates for multiple stocks, including EPS and revenue estimates,
    as well as prior EPS and revenue figures for comparison.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Any]: Dictionary mapping each ticker to its earnings data.
    """
    return await get_ticker_specific_data(tickers, "earnings/raw")


@function_tool
async def get_ticker_bull_vs_bear(tickers: List[str]) -> Dict[str, Any]:
    """
    Get historical bull vs. bear sentiment data for multiple stocks.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Any]: Dictionary mapping each ticker to its bull vs. bear sentiment data.
    """
    return await get_ticker_specific_data(tickers, "bull_vs_bear")



async def get_feed_data(
    tickers: List[str], 
    file_path: Path,
    filter_keys: Set[str],
    sort_key: str,
    limit: int = 20
) -> Dict[str, List[Dict[str, Any]]]:
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            data = orjson.loads(await f.read())

        # Sanitize tickers list
        valid_tickers = {
            t.strip().upper()
            for t in tickers
            if isinstance(t, str) and t.strip() and t.strip() != '{}'
        }

        if not valid_tickers:
            # No valid tickers: return globally sorted top items
            cleaned_data = [
                {k: v for k, v in item.items() if k not in filter_keys}
                for item in data
            ]
            sorted_data = sorted(
                cleaned_data,
                key=lambda x: x.get(sort_key, 0),
                reverse=True
            )[:limit]
            return {"ALL": sorted_data}

        # Filter and group by ticker
        filtered_results = defaultdict(list)
        for item in data:
            try:
                ticker = item.get("ticker", "").upper()
                if ticker in valid_tickers:
                    cleaned_item = {k: v for k, v in item.items() if k not in filter_keys}
                    filtered_results[ticker].append(cleaned_item)
            except Exception as e:
                print(f"Skipping item due to error: {e}")
                continue

        # Sort and limit results for each ticker
        result = {
            ticker: sorted(
                items,
                key=lambda x: x.get(sort_key, 0),
                reverse=True
            )[:limit]
            for ticker, items in filtered_results.items()
        }

        return result

    except FileNotFoundError:
        return {}
    except (orjson.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing feed data: {e}")
        return {}




@function_tool
async def get_latest_options_flow_feed(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves the top options flow orders with the highest premiums for multiple stocks, highlighting activity from hedge funds and major traders.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]). If no tickers are available, set it to an empty list [].

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of its top options flow entries, sorted by premium.
    """
    return await get_feed_data(
        tickers=tickers,
        file_path=BASE_DIR / "options-flow/feed/data.json",
        filter_keys={'aggressor_ind', "exchange", "tradeCount", "trade_count", "underlying_type", "description"},
        sort_key="cost_basis"
    )

@function_tool
async def get_latest_dark_pool_feed(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves the top dark pool trades for multiple stocks, sorted by the average price paid, highlighting significant activity from hedge funds and major traders.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]). If no tickers are available, set it to an empty list [].

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of its top dark pool trade entries, sorted by premium.
    """
    return await get_feed_data(
        tickers=tickers,
        file_path=BASE_DIR / "dark-pool/feed/data.json",
        filter_keys={"assetType", "sector", "trackingID"},
        sort_key="premium",
    )

@function_tool
async def get_ticker_news(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves the latest news for multiple stocks.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of recent news items.
    """
    base_dir = BASE_DIR / "market-news/companies"
    
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    filtered_results: Dict[str, List[Dict[str, Any]]] = {}
    for ticker, data in zip(tickers, results):
        if data is not None:
            filtered_data = [
                {k: v for k, v in item.items() if k not in ['image', 'symbol', 'url', 'site']}
                for item in data
                if 'publishedDate' in item and 
                week_ago <= datetime.strptime(item['publishedDate'], "%Y-%m-%d %H:%M:%S").date()
            ]
            if filtered_data:
                filtered_results[ticker] = filtered_data
                
    return filtered_results




@function_tool
async def get_ticker_analyst_rating(tickers: List[str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Retrieves the latest analyst ratings for multiple stocks.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Dict[str, List[Dict[str, Any]]]]: A dictionary where each key is a ticker and each value is another dictionary with:
            - 'analyst_rating': a list of up to 30 detailed analyst rating entries (each entry excludes the 'analystId' field).
            - 'rating_summary': a single-item list containing a summary dictionary.
    """
    history_dir = BASE_DIR / "analyst" / "history"
    summary_dir = BASE_DIR / "analyst" / "summary" / "all_analyst"

    history_tasks = [fetch_ticker_data(t, history_dir) for t in tickers]
    summary_tasks = [fetch_ticker_data(t, summary_dir) for t in tickers]

    histories, summaries = await asyncio.gather(
        asyncio.gather(*history_tasks),
        asyncio.gather(*summary_tasks)
    )

    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for ticker, hist_data, summ_data in zip(tickers, histories, summaries):
        results[ticker] = {}

        if hist_data:
            cleaned_history = [
                {k: v for k, v in entry.items() if k != 'analystId'}
                for entry in hist_data
            ][:30]
            results[ticker]['analyst_rating'] = cleaned_history

        if isinstance(summ_data, dict):
            summary_clean = {
                k: v for k, v in summ_data.items() 
                if k not in {'recommendationList', 'pastPriceList'}
            }
            results[ticker]['rating_summary'] = [summary_clean]
            
    return results


async def get_stock_screener(
    rule_of_list: Optional[List[Dict[str, Any]]] = None, 
    sort_by: Optional[str] = None, 
    sort_order: str = "desc", 
    limit: int = 15
) -> Dict[str, Any]:
    f"""
    Screens stocks based on a list of specified financial criteria.
    This function filters stocks that meet defined thresholds for various metrics
    (e.g., marketCap > 10E9, P/E ratio < 15, etc.).
    If no 'rule_of_list' is provided, it returns a default list of stocks (excluding 'OTC' exchange).
    All available screening metrics: {', '.join(key_screener)}.

    """
    try:
        file_path = BASE_DIR / "stock-screener/data.json"
        async with aiofiles.open(file_path, 'rb') as file:
            data = orjson.loads(await file.read())

        # Initial filter to exclude OTC exchange
        filtered_data = [item for item in data if item.get('exchange') != 'OTC']

        # Exit early if no rules provided
        if not rule_of_list:
            result = filtered_data
        else:
            # Apply filtering rules
            result = []
            for stock in filtered_data:
                meets_criteria = True
                
                # Check each rule
                for rule in rule_of_list:
                    # Get rule components
                    metric = rule.get('metric', rule.get('name'))
                    value = rule.get('value')
                    operator = rule.get('operator', '>')
                    
                    # Skip invalid rules
                    if not metric or metric not in stock or operator not in OPERATORS:
                        meets_criteria = False
                        break
                    
                    stock_value = stock[metric]
                    
                    # Handle None values
                    if stock_value is None:
                        meets_criteria = False
                        break

                    try:
                        if not OPERATORS[operator](stock_value, value):
                            meets_criteria = False
                            break
                    except (TypeError, ValueError):
                        meets_criteria = False
                        break
                
                if meets_criteria:
                    result.append(stock)

        # Sort results if requested
        if sort_by and result and sort_by in result[0]:
            result.sort(
                key=lambda x: (x.get(sort_by) is None, x.get(sort_by)),
                reverse=(sort_order.lower() == "desc")
            )

        # Apply limit
        if limit and isinstance(limit, int):
            result = result[:limit]

        # Format output
        filtered_result = []
        for stock in result:
            try:
                filtered_stock = {
                    "symbol": stock.get("symbol", ""),
                    "company_name": stock.get("companyName", stock.get("name", "")),
                }
                
                # Add metrics from rule_of_list
                if rule_of_list:
                    metrics = {}
                    for rule in rule_of_list:
                        metric_name = rule.get('metric', rule.get('name'))
                        if metric_name and metric_name in stock:
                            metrics[metric_name] = stock[metric_name]
                    
                    # Add sort_by field if used for sorting
                    if sort_by and sort_by not in metrics and sort_by in stock:
                        metrics[sort_by] = stock[sort_by]
                    
                    if metrics:
                        filtered_stock["metrics"] = metrics
                
                filtered_result.append(filtered_stock)
            except Exception as e:
                print(f"Error processing stock in screener: {e}")

        return {
            "matched_stocks": filtered_result,
            "count": len(filtered_result)
        }

    except FileNotFoundError:
        return {"matched_stocks": [], "count": 0, "error": "Screener data file not found"}
    except (orjson.JSONDecodeError, Exception) as e:
        return {"matched_stocks": [], "count": 0, "error": f"Error processing screener data: {str(e)}"}

@function_tool
async def get_top_gainers() -> list[dict]:
    """
    Retrieves a list of stocks with the highest percentage gains for the current trading day.
    Returns the top 10 securities ranked by their daily price increase percentage.

    Returns:
        list[dict]: A list of up to 10 dictionaries representing top gaining stocks.
        Each dictionary includes key fields such as symbol and percent gain.
    """
    try:
        with open("json/market-movers/markethours/gainers.json", "rb") as f:
            raw = f.read()
        data = orjson.loads(raw)["1D"][:10]
        return data
    except Exception as e:
        return [{"error": f"Error processing top gainers data: {str(e)}"}]

@function_tool
async def get_top_premarket_gainers() -> list[dict]:
    """
    Retrieves a list of stocks with the highest percentage gains for the premarket trading session.
    Returns the top 20 securities ranked by their price increase percentage in the premarket.

    Returns:
        list[dict]: A list of up to 20 dictionaries representing top premarket gaining stocks.
            Each dictionary includes key fields such as symbol and percent gain.
    """
    try:
        with open("json/market-movers/premarket/gainers.json", "rb") as file:
            data = orjson.loads(file.read())[:20]
            return data
    except Exception as e:
        return [{"error": f"Error processing top premarket gainers data: {str(e)}"}]

@function_tool
async def get_top_aftermarket_gainers() -> list[dict]:
    """
    Retrieves a list of stocks with the highest percentage gains for the aftermarket trading session.
    Returns the top 20 securities ranked by their price increase percentage in the aftermarket.

    Returns:
        list[dict]: A list of up to 20 dictionaries representing top aftermarket gaining stocks.
            Each dictionary includes key fields such as symbol and percent gain.
    """
    try:
        with open("json/market-movers/afterhours/gainers.json", "rb") as file:
            data = orjson.loads(file.read())[:20]
            return data
    except Exception as e:
        return [{"error": f"Error processing top aftermarket gainers data: {str(e)}"}]


@function_tool
async def get_top_losers() -> list[dict]:
    """
    Retrieves a list of stocks with the highest percentage losses for the current trading day.
    Returns the top 10 securities ranked by their daily percentage decrease.

    Returns:
        list[dict]: A list of up to 10 dictionaries representing top losing stocks of the trading day.
            Each dictionary includes key fields such as symbol and percent loss.
    """
    try:
        with open("json/market-movers/markethours/losers.json", "rb") as file:
            data = orjson.loads(file.read())["1D"][:10]
            return data
    except Exception as e:
        return [{"error": f"Error processing top losers data: {str(e)}"}]


@function_tool
async def get_top_premarket_losers() -> list[dict]:
    """
    Retrieves a list of stocks with the highest percentage losses for the premarket trading session.
    Returns the top 20 securities ranked by their price decrease percentage in the premarket.

    Returns:
        list[dict]: A list of up to 20 dictionaries representing top premarket losing stocks.
            Each dictionary includes key fields such as symbol and percent loss.
    """
    try:
        with open("json/market-movers/premarket/losers.json", "rb") as file:
            data = orjson.loads(file.read())[:20]
            return data
    except Exception as e:
        return [{"error": f"Error processing top premarket losers data: {str(e)}"}]


@function_tool
async def get_top_aftermarket_losers() -> list[dict]:
    """
    Retrieves a list of stocks with the highest percentage losses for the aftermarket trading session.
    Returns the top 20 securities ranked by their price decrease percentage in the aftermarket.

    Returns:
        list[dict]: A list of up to 20 dictionaries representing top aftermarket losing stocks.
            Each dictionary includes key fields such as symbol and percent loss.
    """
    try:
        with open("json/market-movers/afterhours/losers.json", "rb") as file:
            data = orjson.loads(file.read())[:20]
            return data
    except Exception as e:
        return [{"error": f"Error processing top aftermarket losers data: {str(e)}"}]



@function_tool
async def get_top_active_stocks() -> list[dict]:
    """
    Retrieves a list of stocks with the largest trading volume for the current trading day.
    Returns stocks ranked by their daily volume, showing the most actively traded securities.

    Returns:
        list[dict]: A list of up to 10 dictionaries representing the most active stocks.
    """
    try:
        with open("json/market-movers/markethours/active.json", "rb") as file:
            data = orjson.loads(file.read())["1D"][:10]
            return data
    except Exception as e:
        return [{"error": f"Error processing most active stock data: {str(e)}"}]

@function_tool
async def get_potus_tracker() -> dict:
    """
    Gets the latest POTUS tracker data, including presidential schedule, Truth Social posts,
    executive orders, and S&P 500 (SPY) performance since the current president's inauguration.

    Returns:
        dict: A dictionary containing POTUS tracker data.
    """
    try:
        with open("json/tracker/potus/data.json", "rb") as file:
            data = orjson.loads(file.read())
            return data
    except Exception as e:
        return {"error": f"Error processing potus tracker data: {str(e)}"}

@function_tool
async def get_insider_tracker() -> list[dict]:
    """
    Retrieves the latest insider trading activity, including recent stock sales or purchases by company executives,
    with relevant company info like symbol, price, market cap, and filing date.

    Returns:
        list[dict]: A list of up to 20 dictionaries representing recent insider trades.
    """
    try:
        with open("json/tracker/insider/data.json", "rb") as file:
            data = orjson.loads(file.read())
            return data[:20]
    except Exception as e:
        return [{"error": f"Error processing insider tracker data: {str(e)}"}]

@function_tool
async def get_latest_congress_trades() -> list[dict]:
    """
    Retrieves the latest congressional stock trading disclosures including transactions by members of Congress or their spouses.
    Returns details such as ticker, transaction type, amount, representative name, and disclosure dates.
    Some less relevant fields are excluded.

    Returns:
        list[dict]: A list of up to 20 cleaned congressional trade entries.
    """
    try:
        with open("json/congress-trading/rss-feed/data.json", "rb") as file:
            data = orjson.loads(file.read())

            fields_to_remove = {
                "link",
                "district",
                "id",
                "assetType",
                "capitalGainsOver200USD"
            }

            data = [{k: v for k, v in entry.items() if k not in fields_to_remove} for entry in data]
            return data[:20]
    except Exception as e:
        return [{"error": f"Error processing congress tracker data: {str(e)}"}]

@function_tool
async def get_analyst_tracker() -> list[dict]:
    """
    Retrieves the latest analyst stock ratings, including upgrades, downgrades, maintained ratings,
    price targets, analyst names, firms, and expected upside.
    Removes analyst ID from the results for privacy.

    Returns:
        list[dict]: A list of up to 20 analyst rating entries.
    """
    try:
        with open("json/analyst/flow-data.json", "rb") as file:
            data = orjson.loads(file.read())

            fields_to_remove = {"analystId"}

            data = [{k: v for k, v in entry.items() if k not in fields_to_remove} for entry in data]
            return data[:20]
    except Exception as e:
        return [{"error": f"Error processing analyst tracker data: {str(e)}"}]

@function_tool
async def get_market_news() -> list[dict]:
    """
    Retrieves the latest general news for the stock market.
    Removes some less relevant fields such as site, url, and image.

    Returns:
        list[dict]: A list of up to 30 news articles with essential fields.
    """
    try:
        with open("json/market-news/general-news.json", "rb") as file:
            data = orjson.loads(file.read())

            fields_to_remove = {"site", "url", "image"}

            data = [{k: v for k, v in entry.items() if k not in fields_to_remove} for entry in data]
            return data[:30]
    except Exception as e:
        return [{"error": f"Error processing market news data: {str(e)}"}]


@function_tool
async def get_market_flow() -> Dict[str, Any]:
    """
    Retrieves the current options market flow sentiment data for the S&P 500.

    Reads precomputed data from `json/market-flow/data.json` and filters it to include
    only relevant entries from the market tide.

    Returns:
        Dict[str, Any]: A dictionary with:
            - "marketTide": a list of time-series entries showing net call/put premium, 
              call/put volume, and net volume at each timestamp.  
              Only entries that contain a 'close' key are included.
            - "overview": aggregated metrics including total put/call volume and open interest, 
              put/call ratios, averages over the past 30 days, and the reference date.
            
        If an error occurs, returns a dictionary with an "error" key and message.
    """
    try:
        with open("json/market-flow/data.json", "rb") as file:
            data = orjson.loads(file.read())
            market_tide = []
            for item in data['marketTide']:
                if item.get('close'):
                    market_tide.append(item)
            data['marketTide'] = market_tide
            return data
    except Exception as e:
        return {"error": f"Error processing market flow data: {str(e)}"}


@function_tool
async def get_congress_activity(congress_ids: Union[str, List[str]]) -> Dict[str, List[Any]]:
    """
    Retrieves and filters congressional trading activity for one or more congresspeople based on their unique IDs or names.
    Groups results by the office of each congressperson.

    Args:
        congress_ids (Union[str, List[str]]): Single congressperson name/ID or list of names/IDs 
            (e.g., "Nancy Pelosi" or ["Nancy Pelosi", "a9c12796a0"]).

    Returns:
        Dict[str, List[Any]]: A dictionary where keys are offices (e.g., "Senate", "House"), and values are lists of activity data objects.
            If an error occurs for a specific ID, an "error_<ID>" key maps to the error message.
    """
    # Normalize input to a list
    if isinstance(congress_ids, str):
        congress_ids = [congress_ids]

    result: Dict[str, List[Any]] = {}

    # Prepare a list of known names for matching
    known_names = list(key_congress_db.keys())

    for identifier in congress_ids:
        # Attempt to resolve name to ID using fuzzy matching
        if identifier not in key_congress_db.values():
            match = process.extractOne(
                identifier,
                known_names,
                scorer=fuzz.WRatio,
                score_cutoff=80  # Adjust the cutoff as needed
            )
            if match:
                resolved_name = match[0]
                congress_id = key_congress_db[resolved_name]
            else:
                error_key = f"error_{identifier}"
                result[error_key] = [f"No matching congressperson found for '{identifier}'."]
                continue
        else:
            congress_id = identifier

        try:
            with open(f"json/congress-trading/politician-db/{congress_id}.json", "rb") as file:
                data = orjson.loads(file.read())

                office = None
                if "history" in data:
                    fields_to_remove = {
                        "assetDescription",
                        "firstName",
                        "lastName",
                        "capitalGainsOver200USD",
                        "comment",
                        "congress",
                        "office",
                        "representative",
                        "link",
                        "id",
                    }
                    office = data["history"][0].get("office", "Unknown")
                    data["history"] = [
                        {k: v for k, v in entry.items() if k not in fields_to_remove}
                        for entry in data["history"]
                    ]

                if office not in result:
                    result[office] = []
                result[office].append(data)

        except Exception as e:
            error_key = f"error_{congress_id}"
            result[error_key] = [f"Error processing congress activity data: {str(e)}"]

    return result

@function_tool
async def get_ticker_quote(tickers: List[str]) -> Dict[str, Any]:
    """
    Retrieves the most recent stock quote data for one or more ticker symbols.
    Returns real-time information including:
    - latest trading price
    - percentage change
    - trading volume
    - day high and low
    - 52-week high and low
    - market capitalization
    - previous close
    - earnings per share (EPS)
    - price-to-earnings (P/E) ratio
    - current ask and bid prices

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Any]: A dictionary mapping each ticker to its quote data object.
    """
    return await get_ticker_specific_data(tickers, "quote")


@function_tool
async def get_ticker_pre_post_quote(tickers: List[str]) -> Dict[str, Any]:
    """
    Retrieves the most recent stock premarket/aftermarket quote data for one or more ticker symbols.
    Returns real-time information including:
    - latest trading price
    - percentage change

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Any]: A dictionary mapping each ticker to its pre/post market quote data object.
    """
    return await get_ticker_specific_data(tickers, "pre-post-quote")


@function_tool
async def get_ticker_insider_trading(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch detailed insider trading records—such as buys, sells, grant dates, and volumes—executed by corporate insiders 
    (officers, directors, major shareholders) for the specified stock tickers.
    Returns up to the 50 most recent entries per ticker, excluding 'companyCik' and 'url' fields.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of up to 50 insider trading entries.
    """
    base_dir = BASE_DIR / "insider-trading/history"
    
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    filtered_results: Dict[str, List[Dict[str, Any]]] = {}
    for ticker, data in zip(tickers, results):
        if data is not None:
            filtered_data = [
                {k: v for k, v in item.items() if k not in ["companyCik", "url"]}
                for item in data
            ]
            if filtered_data:
                filtered_results[ticker] = filtered_data[:50]
                
    return filtered_results

@function_tool
async def get_ticker_shareholders(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch current institutional and major shareholder data for the specified stock tickers, including top institutional holders 
    and ownership statistics such as investor counts, total invested value, and put/call ratios.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary with two keys:
            - "top-shareholders": maps each ticker to a list of shareholder entries (excluding the 'cik' field).
            - "ownership-stats": maps each ticker to its ownership statistics entries.
    """
    final_res: Dict[str, Dict[str, Any]] = {}
    
    for path in ['shareholders', 'ownership-stats']:
        base_dir = BASE_DIR / path
        
        tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
        results = await asyncio.gather(*tasks)

        filtered_results: Dict[str, List[Dict[str, Any]]] = {}
        for ticker, data in zip(tickers, results):
            if data is None:
                continue
            
            if path == 'shareholders':
                # Remove "cik" key from each shareholder entry
                cleaned_data = [
                    {k: v for k, v in entry.items() if k != 'cik'}
                    for entry in data
                ]
                filtered_results[ticker] = cleaned_data
            else:
                filtered_results[ticker] = data

        if path == 'shareholders':
            final_res['top-shareholders'] = filtered_results
        else:
            final_res['ownership-stats'] = filtered_results
                
    return final_res


@function_tool
async def get_ticker_options_overview_data(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch comprehensive options statistics for the most recent trading day for the specified stock tickers.
    Includes overview, implied volatility, open interest, volume, and expiration table.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping each ticker to its options data.
    """
    base_dir = BASE_DIR / "options-chain-statistics"
    
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    final_results: Dict[str, Dict[str, Any]] = {}
    for ticker, data in zip(tickers, results):
        if data:
            # Assuming data is already a dict in the expected format (like your example JSON)
            final_results[ticker] = data

    return final_results


@function_tool
async def get_ticker_max_pain(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve max pain analysis for multiple stock tickers, including expiration dates, strike prices, call and put payouts,
    and the calculated max pain point per expiration.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to its list of max pain analysis entries.
    """
    base_dir = BASE_DIR / "max-pain"
    
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    return {ticker: result for ticker, result in zip(tickers, results) if result}



@function_tool
async def get_ticker_open_interest_by_strike_and_expiry(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch and aggregate options open interest data for one or more equity tickers, returning:
        - expiry-data: total call and put OI grouped by each expiration date (only future expiries)
        - strike-data: call and put OI broken down by individual strike price

    Args:
        tickers (List[str]): List of stock ticker symbols to analyze (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary with two keys:
            - "expiry-data": maps each ticker to its list of expiry-level OI entries.
            - "strike-data": maps each ticker to its list of strike-level OI entries.
            Each entry is a dict containing fields like "call_oi", "put_oi", and the associated expiry or strike value.
    """
    final_res = {}
    today = datetime.today().date()

    for category_type in ['expiry', 'strike']:
        base_dir = BASE_DIR / f"oi/{category_type}"
        tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        
        res: Dict[str, List[Dict[str, Any]]] = {}
        for ticker, result in zip(tickers, results):
            if not result:
                continue
            
            # Filter only valid expiry dates
            filtered_result = [
                item for item in result 
                if 'expiry' not in item or datetime.strptime(item['expiry'], '%Y-%m-%d').date() >= today
            ]
            if filtered_result:
                res[ticker] = filtered_result
        
        final_res[f"{category_type}-data"] = res
                
    return final_res


@function_tool
async def get_ticker_unusual_activity(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve recent unusual options activity for one or more stock tickers, including large trades, sweeps, and high-premium orders.
    Returns the top 10 most recent entries per ticker, sorted by date descending.

    Args:
        tickers (List[str]): List of stock ticker symbols to analyze (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of its top 10 unusual activity entries.
            Each entry is a dict containing fields like "date", "strike", "volume", etc., sorted by most recent first.
    """
    base_dir = BASE_DIR / "unusual-activity"
    
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    sorted_results: Dict[str, List[Dict[str, Any]]] = {}
    for ticker, result in zip(tickers, results):
        if result:
            sorted_result = sorted(result, key=lambda x: x['date'], reverse=True)
            sorted_results[ticker] = sorted_result[:10]

    return sorted_results


@function_tool
async def get_ticker_dark_pool(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves dark pool trading data and related analytics for a list of stock ticker symbols, including:
        - volume-summary: latest volume summaries per ticker
        - hottest_trades_and_price_level: full list of dark pool entries per ticker

    Args:
        tickers (List[str]): List of stock ticker symbols to analyze (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary with two keys:
            - "volume-summary": maps each ticker to its most recent volume summary entry (a dict).
            - "hottest_trades_and_price_level": maps each ticker to its list of dark pool entries.
    """
    final_res: Dict[str, Dict[str, Any]] = {}

    for category_type in ['companies', 'price-level']:
        base_dir = BASE_DIR / f"dark-pool/{category_type}"
        tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        
        res: Dict[str, Any] = {}
        for ticker, result in zip(tickers, results):
            if not result:
                continue
            if category_type == 'companies':
                # Use only the latest volume summary entry per ticker
                res[ticker] = result[-1]
            else:
                # Return the full list of entries for price-level analysis
                res[ticker] = result

        if category_type == 'companies':
            final_res["volume-summary"] = res
        else:
            final_res["hottest_trades_and_price_level"] = res

    return final_res


@function_tool
async def get_ticker_dividend(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve dividend data and related metrics for a list of stock ticker symbols, 
    including payout frequency, annual dividend, dividend yield, payout ratio, 
    and dividend growth. Also returns historical records with detailed information 
    such as declaration date, record date, payment date, and adjusted dividend amount.

    Args:
        tickers (List[str]): List of stock ticker symbols to analyze.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping each ticker to its dividend info.
    """
    base_dir = BASE_DIR / "dividends/companies"
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    res = {}
    for ticker, result in zip(tickers, results):
        history = result.get('history', [])
        if history:
            result['history'] = history[0]
            res[ticker] = result

    return res


@function_tool
async def get_ticker_statistics(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    f"""
    Retrieves a snapshot of all latest data for a list of stock ticker symbols.
    This includes key quantities such as: {', '.join(key_screener)}.
    Upside in this context is compared the percentage change compared to the current stock price. Positive upside indicates stock is undervalued and negative overvalued.

    Args:
        tickers (List[str]): List of stock ticker symbols to analyze.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping each ticker to its statistics.
    """
    file_path = BASE_DIR / "stock-screener/data.json"
    async with aiofiles.open(file_path, 'rb') as file:
        data = orjson.loads(await file.read())

    # Initial filter to exclude OTC exchange
    stock_screener_data = [item for item in data if item.get('exchange') != 'OTC']
    stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

    result = {}
    if len(tickers) > 0:
        for symbol in tickers:
            try:
                if symbol in stock_screener_data_dict:
                    item_data = stock_screener_data_dict[symbol]
                    result[symbol] = {col: item_data.get(col, None) for col in key_screener}
            except:
                pass
    return result



@function_tool
async def get_ticker_key_metrics(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    f"""
    Retrieves fundamental key metrics data for a list of stock ticker symbols.
    This includes key data such as: {', '.join(key_metrics)}.

    Args:
        tickers (List[str]): List of stock ticker symbols to analyze.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of filtered key metrics.
    """
    base_dir = BASE_DIR / "financial-statements/key-metrics/ttm"
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    filtered_results = {}
    for ticker, data in zip(tickers, results):
        if data is not None:
            filtered_data = [
                {k: v for k, v in item.items() if k not in ['symbol']} for item in data
            ]
            if filtered_data:
                filtered_results[ticker] = filtered_data

    return filtered_results


@function_tool
async def get_ticker_financial_score(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve fundamental financial score data for a list of stock ticker symbols. Includes metrics such as
    Altman Z-Score, Piotroski Score, working capital, and total assets.

    Args:
        tickers (List[str]): List of stock ticker symbols to analyze (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping each ticker to its financial score data.
    """
    base_dir = BASE_DIR / "financial-score"
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    return {ticker: result for ticker, result in zip(tickers, results)}


@function_tool
async def get_ticker_owner_earnings(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    f"""
    Retrieves fundamental owner earnings data for a list of stock ticker symbols.
    This includes key data such as: {', '.join(key_owner_earnings)}.

    Args:
        tickers (List[str]): List of stock ticker symbols to analyze (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of filtered owner earnings entries.
    """
    base_dir = BASE_DIR / "financial-statements/owner-earnings/quarter"
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    filtered_results = {}
    for ticker, data in zip(tickers, results):
        if data is not None:
            filtered_data = [
                {k: v for k, v in item.items() if k not in ['symbol', 'reportedCurrency']}
                for item in data
            ]
            if filtered_data:
                filtered_results[ticker] = filtered_data

    return filtered_results


@function_tool
async def get_oversold_tickers() -> Dict[str, Dict[str, Any]]:
    """
    Retrieves the top 10 most oversold stocks based on the Relative Strength Index (RSI), 
    sorted by the lowest RSI values.

    Each stock entry includes:
        - symbol: Stock ticker symbol
        - name: Full company name
        - price: Latest trading price
        - changesPercentage: Percent change in price
        - marketCap: Market capitalization in USD
        - rsi: Relative Strength Index value
        - rank: Position in the list based on RSI

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary of the top 10 oversold stocks, keyed by their rank.
    """
    base_dir = BASE_DIR / "stocks-list/list/oversold-stocks.json"
    with open(base_dir, "rb") as file:
        result = orjson.loads(file.read())[:10]

    return result


@function_tool
async def get_overbought_tickers() -> Dict[str, Dict[str, Any]]:
    """
    Retrieves the top 10 most overbought stocks based on the Relative Strength Index (RSI), 
    sorted by the highest RSI values.

    Each stock entry includes:
        - symbol: Stock ticker symbol
        - name: Full company name
        - price: Latest trading price
        - changesPercentage: Percent change in price
        - marketCap: Market capitalization in USD
        - rsi: Relative Strength Index value
        - rank: Position in the list based on RSI

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary of the top 10 oversold stocks, keyed by their rank.
    """
    base_dir = BASE_DIR / "stocks-list/list/overbought-stocks.json"
    with open(base_dir, "rb") as file:
        result = orjson.loads(file.read())[:10]

    return result


@function_tool
async def get_dividend_kings() -> List[Dict[str, Any]]:
    """
    Retrieves the top 10 Dividend Kings—companies that have increased their dividend payouts 
    for at least 50 consecutive years—sorted by the longest streak.

    Each stock entry includes:
        - symbol: Stock ticker symbol
        - name: Full company name
        - price: Latest trading price
        - changesPercentage: Percent change in price
        - dividendYield: Current dividend yield (percentage)
        - years: Number of consecutive years of dividend increases
        - rank: Position in the list based on number of years

    Returns:
        List[Dict[str, Any]]: A list of the top 10 Dividend Kings.
    """
    base_dir = BASE_DIR / "dividends/list/dividend-kings.json"
    with open(base_dir, "rb") as file:
        result = orjson.loads(file.read())[:10]

    return result

@function_tool
async def get_dividend_aristocrats() -> List[Dict[str, Any]]:
    """
    Retrieves the top 10 Dividend Aristocrats—S&P 500 companies that have increased their 
    dividend payouts for at least 25 consecutive years—sorted by the longest streak.

    Each stock entry includes:
        - symbol: Stock ticker symbol
        - name: Full company name
        - price: Latest trading price
        - changesPercentage: Percent change in price
        - dividendYield: Current dividend yield (percentage)
        - years: Number of consecutive years of dividend increases
        - rank: Position in the list based on number of years

    Returns:
        List[Dict[str, Any]]: A list of the top 10 Dividend Aristocrats.
    """
    base_dir = BASE_DIR / "dividends/list/dividend-aristocrats.json"
    with open(base_dir, "rb") as file:
        result = orjson.loads(file.read())[:10]
    return result


@function_tool
async def get_top_rated_dividend_stocks() -> List[Dict[str, Any]]:
    """
    Retrieves the top 10 dividend-paying stocks with strong analyst ratings 
    and attractive dividend yields.

    Each stock meets the following criteria:
        - Dividend yield of at least 2%
        - Average analyst rating of 'buy' or 'strong buy' (from at least 10 analysts)
        - Sustainable payout ratio (typically below 60%)
        - Large market capitalization indicating stability

    Each entry includes:
        - symbol: Stock ticker symbol
        - name: Full company name
        - price: Latest trading price
        - changesPercentage: Percent change in price
        - marketCap: Market capitalization in USD
        - dividendYield: Current dividend yield (percentage)
        - rank: Position in the list based on composite score

    Returns:
        List[Dict[str, Any]]: A list of the top 10 top-rated dividend-paying stocks.
    """
    base_dir = BASE_DIR / "stocks-list/list/top-rated-dividend-stocks.json"
    with open(base_dir, "rb") as file:
        result = orjson.loads(file.read())[:10]
    return result

@function_tool
async def get_monthly_dividend_stocks() -> List[Dict[str, Any]]:
    """
    Retrieves the top 10 monthly dividend-paying stocks with high yields 
    and strong income-generating potential.

    These stocks typically distribute dividends every month rather than quarterly,
    making them attractive to income-focused investors seeking regular cash flow.

    Each entry includes:
        - symbol: Stock ticker symbol
        - name: Full company name
        - price: Latest trading price
        - changesPercentage: Percent change in price
        - marketCap: Market capitalization in USD
        - dividendYield: Current dividend yield (percentage)
        - rank: Position in the list based on dividend consistency and yield

    Returns:
        List[Dict[str, Any]]: A list of the top 10 monthly dividend-paying stocks.
    """
    base_dir = BASE_DIR / "stocks-list/list/monthly-dividend-stocks.json"
    with open(base_dir, "rb") as file:
        result = orjson.loads(file.read())[:10]
    return result

@function_tool
async def get_ticker_trend_forecast(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    AI Trend Forecast: Uses an AI model analyzing historical price data to forecast stock trends.
    It detects patterns, seasonality, and trends to predict future price movements and estimate
    price targets over the next 12 months.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping each ticker symbol to a dictionary of
        price target estimates including:
            - avgPriceTarget: Average predicted price
            - highPriceTarget: Highest predicted price target
            - lowPriceTarget: Lowest predicted price target
            - medianPriceTarget: Median predicted price target
    """
    result = await get_ticker_specific_data(tickers, "price-analysis")
    
    filtered_result = {}
    for ticker, data in result.items():
        if data:
            data_copy = dict(data)  # shallow copy to avoid mutating original
            data_copy.pop("pastPriceList", None)  # remove unnecessary large data
            filtered_result[ticker] = data_copy

    return filtered_result

@function_tool
async def get_dividend_calendar() -> List[Dict[str, Any]]:
    """
    Fetches a list of upcoming dividend-paying stocks along with key payout details.

    This function is designed for income-focused investors and financial analysts who 
    track dividend events. It retrieves data on upcoming ex-dividend dates and other 
    relevant information, enabling timely decision-making around dividend capture strategies.

    Each returned entry contains:
        - symbol (str): The stock's ticker symbol.
        - name (str): The full name of the company.
        - adjDividend (float): The adjusted dividend amount per share.
        - date (str): The ex-dividend date—when the stock starts trading without the dividend.
        - recordDate (str): The date by which investors must own the stock to be eligible for the dividend.
        - paymentDate (str): The date on which the dividend will be paid.
        - marketCap (float): The company's market capitalization in USD.

    Returns:
        List[Dict[str, Any]]: A filtered list of dictionaries, each representing 
        an upcoming dividend event with essential payout information.
    """

    base_dir = BASE_DIR / "dividends-calendar/data.json"
    with open(base_dir, "rb") as file:
        raw_data = orjson.loads(file.read())

    # Remove unwanted keys from each dictionary item
    result = [
        {
            key: value
            for key, value in item.items()
            if key not in ["declarationDate", "revenue", "label","dividend"]
        }
        for item in raw_data
    ]

    return result


@function_tool
async def get_ipo_calendar() -> List[Dict[str, Any]]:
    """
    Retrieves data on the 50 most recent initial public offerings (IPOs).

    This function provides a snapshot of newly listed companies, including pricing 
    and performance data since going public. It's useful for investors and analysts 
    tracking recent IPO activity and post-IPO performance.

    Each entry contains:
        - symbol (str): The stock ticker symbol.
        - name (str): The full company name.
        - ipoDate (str): The date the company went public.
        - ipoPrice (float): The offering price at IPO.
        - currentPrice (float | None): The latest trading price, if available.
        - return (float | None): Percentage return since IPO, calculated as 
          (currentPrice - ipoPrice) / ipoPrice * 100, if currentPrice is available.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a recently 
        listed IPO with relevant market data.
    """


    base_dir = BASE_DIR / "ipo-calendar/data.json"
    with open(base_dir, "rb") as file:
        result = orjson.loads(file.read())[:50]

    return result

@function_tool
async def get_penny_stocks() -> List[Dict[str, Any]]:
    """
    Retrieves a list of actively traded penny stocks, defined as stocks priced below $5 per share
    and with a trading volume exceeding 10,000 shares.

    The result is sorted by volume in descending order and limited to the top 10 entries.

    Returns:
        List[Dict[str, Any]]: A list of the top 10 most active penny stocks with:
            - symbol: Ticker symbol
            - name: Company name
            - price: Current trading price
            - changesPercentage: Percent price change
            - marketCap: Market capitalization
            - volume: Daily trading volume
            - rank: Rank based on trading activity
    """
    base_dir = BASE_DIR / "stocks-list/list/penny-stocks.json"
    with open(base_dir, "rb") as file:
        data = orjson.loads(file.read())[:10]

    return data

@function_tool
async def get_most_shorted_stocks() -> List[Dict[str, Any]]:
    """
    Retrieves a list of the most heavily shorted stocks based on short float percentage,
    which indicates the percentage of a company's float that is sold short.

    This list is useful for identifying stocks with high short interest that may be
    vulnerable to short squeezes or strong price movements.

    Returns:
        List[Dict[str, Any]]: A list of the top 10 most shorted stocks, each with:
            - symbol: Stock ticker symbol
            - name: Full company name
            - price: Latest trading price
            - changesPercentage: Percent change in price
            - shortFloatPercent: Short float percentage
            - rank: Position based on short interest
    """
    base_dir = BASE_DIR / "stocks-list/list/most-shorted-stocks.json"
    with open(base_dir, "rb") as file:
        data = orjson.loads(file.read())[:10]

    return data

@function_tool
async def get_bitcoin_etfs() -> List[Dict[str, Any]]:
    """
    Retrieves a list of Bitcoin Exchange-Traded Funds (ETFs), which are financial instruments
    designed to track the price of Bitcoin and are traded on stock exchanges.

    These ETFs allow investors to gain exposure to Bitcoin without directly holding the cryptocurrency.

    Returns:
        List[Dict[str, Any]]: A list of up to 30 Bitcoin ETFs, each including:
            - symbol: ETF ticker symbol
            - name: Full ETF name
            - expenseRatio: Annual fee as a percentage of total assets
            - totalAssets: Total assets under management (in USD)
            - price: Latest trading price
            - changesPercentage: Percent change in price
            - rank: Rank based on assets or popularity
    """
    base_dir = BASE_DIR / "etf-bitcoin-list/data.json"
    with open(base_dir, "rb") as file:
        data = orjson.loads(file.read())[:30]

    return data

@function_tool
async def get_all_sector_overview() -> List[Dict[str, Any]]:
    """
    Retrieves an overview of all stock market sectors, including aggregated and median-based metrics.

    For each sector:
    - `totalMarketCap` is the sum of all stocks in the sector.
    - All other fields (e.g. dividend yield, profit margin) represent median values across stocks.

    Returns:
        List[Dict[str, Any]]: A list of sectors with the following fields:
            - sector: Name of the sector
            - numStocks: Number of stocks in the sector
            - totalMarketCap: Total market capitalization of the sector
            - avgDividendYield: Median dividend yield (%)
            - profitMargin: Median profit margin (%)
            - avgChange1D: Median 1-day % change
            - avgChange1Y: Median 1-year % change
            - pe: Median price-to-earnings ratio
    """
    base_dir = BASE_DIR/"industry/sector-overview.json"
    with open(base_dir, "rb") as file:
        data = orjson.loads(file.read())

    return data


@function_tool
async def get_ticker_earnings_call_transcripts(
    tickers: List[str],
    year: int,
    quarter: int
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves earnings call transcripts for a list of stock tickers for a specific year and quarter.

    Each transcript includes the call date and a parsed summary of participant dialogues.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "MSFT"]).
        year (int): The fiscal year of the earnings call (e.g., 2024).
        quarter (int): The fiscal quarter (1–4).

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping each ticker to its earnings call data:
            {
                "AAPL": {
                    "date": "2024-01-25",
                    "chat": [ ... parsed transcript content ... ]
                },
                ...
            }
        If no data is found or an error occurs, that ticker will be omitted.
    """
    results = {}

    async with aiohttp.ClientSession() as session:
        for ticker in tickers:
            try:
                url = (
                    f"https://financialmodelingprep.com/stable/earning-call-transcript"
                    f"?symbol={ticker}&year={year}&quarter={quarter}&apikey={api_key}"
                )

                async with session.get(url) as response:
                    raw_data = await response.json()
                    if not raw_data:
                        continue

                    entry = raw_data[0]
                    content = remove_text_before_operator(entry.get("content", ""))
                    chat = extract_names_and_descriptions(content)

                    results[ticker] = {
                        "date": entry.get("date"),
                        "chat": chat
                    }

            except Exception as e:
                print(f"Error retrieving transcript for {ticker}: {e}")

    return results


@function_tool
async def get_fear_and_greed_index() -> Dict[str, Any]:
    """
    Retrieves the current Fear & Greed index.

    The index provides a numeric value, a sentiment category, and the timestamp
    of the latest update, representing market sentiment ranging from fear to greed.

    Returns:
        Dict[str, Any]: A dictionary containing the current Fear & Greed index:
            {
                "value": 64,                # Numeric index value
                "category": "greed",        # Sentiment category
                "last_update": "2025-08-29T23:59:42+00:00"  # ISO 8601 timestamp
            }
        If the file is missing or malformed, a FileNotFoundError or JSON parsing
        error may be raised.
    """
    base_dir = BASE_DIR / "fear-and-greed/data.json"
    with open(base_dir, "rb") as file:
        data = orjson.loads(file.read()).get("current")

    return data

@function_tool
async def get_reddit_tracker() -> Dict[str, Any]:
    """
    Retrieves WallStreetBets trending tickers for 1 week, 1 month, and 3 months.  

    Each period contains a list of trending tickers with their discussion metrics.  

    Example item:
        {
            "symbol": "RBLX",                  # Stock ticker
            "count": 6,                        # Number of mentions
            "sentiment": "Bearish",            # Overall sentiment (Bullish/Bearish/Neutral)
            "weightPercentage": 3.45,          # Relative weight in discussions (%)
            "name": "Roblox Corporation",      # Company name
            "price": 123.99,                   # Last traded price
            "changesPercentage": -0.66,        # Price change percentage
            "marketCap": 85955687223,          # Market capitalization
            "assetType": "stocks",             # Asset type
            "rank": 8                          # Rank among trending tickers
        }

    Returns:
        Dict[str, Any]: A dictionary with three keys:
            {
                "oneWeek": [ ... top 10 tickers ... ],
                "oneMonth": [ ... top 10 tickers ... ],
                "threeMonths": [ ... top 10 tickers ... ]
            }

        If the file is missing or malformed, a FileNotFoundError or JSON parsing
        error may be raised.
    """
    base_dir = BASE_DIR / "reddit-tracker/wallstreetbets/trending.json"
    with open(base_dir, "rb") as file:
        data = orjson.loads(file.read())

        # Remove oneDay key if present
        data.pop("oneDay", None)

        data['oneWeek'] = data.get('oneWeek', [])[:10]
        data['oneMonth'] = data.get('oneMonth', [])[:10]
        data['threeMonths'] = data.get('threeMonths', [])[:10]
    return data



#Testing purposes
#data = asyncio.run(get_reddit_tracker())
#print(data)