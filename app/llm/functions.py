import os
import asyncio
import aiofiles
import orjson
from utils.helper import load_congress_db
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar, Set, Tuple, cast

from agents import function_tool

load_dotenv()


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
  "piotroskiScore"
]

key_statistics = ['sharesOutStanding', 'sharesQoQ', 'sharesYoY','institutionalOwnership','floatShares',
    'priceToEarningsGrowthRatio','priceToEarningsRatio','forwardPE','priceToSalesRatio','forwardPS','priceToBookRatio','priceToFreeCashFlowRatio',
    'sharesShort','shortOutstandingPercent','shortFloatPercent','shortRatio',
    'enterpriseValue','evToSales','evToEBITDA','evToOperatingCashFlow','evToFreeCashFlow',
    'currentRatio','quickRatio','debtToFreeCashFlowRatio','debtToEBITDARatio','debtToEquityRatio','interestCoverageRatio','cashFlowToDebtRatio','debtToMarketCap',
    'returnOnEquity','returnOnAssets','returnOnInvestedCapital','revenuePerEmployee','profitPerEmployee',
    'employees','assetTurnover','inventoryTurnover','incomeTaxExpense','effectiveTaxRate','beta',
    'change1Y','sma50','sma200','rsi','avgVolume','revenue','netIncome','grossProfit','operatingIncome','ebitda','ebit','eps',
    'cashAndCashEquivalents','totalDebt','retainedEarnings','totalAssets','workingCapital','operatingCashFlow',
    'capitalExpenditure','freeCashFlow','freeCashFlowPerShare','grossProfitMargin','operatingProfitMargin','pretaxProfitMargin',
    'netProfitMargin','ebitdaMargin','ebitMargin','freeCashFlowMargin','failToDeliver','relativeFTD',
    'annualDividend','dividendYield','payoutRatio','dividendGrowth','earningsYield','freeCashFlowYield','altmanZScore','piotroskiScore',
    'lastStockSplit','splitType','splitRatio','analystRating','analystCounter','priceTarget','upside'
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


key_congress_db = load_congress_db()

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

# Specialized financial statement functions using the generic function
@function_tool
async def get_ticker_income_statement(
    tickers: List[str], time_period: str = "ttm", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    f"""
    Retrieves historical income statements (profit and loss) for a list of stock tickers.
    Key metrics include: {', '.join(key_income)}.
    Available for annual, quarter, and trailing twelve months (ttm).

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).
        time_period (str): Time period for the data: "annual", "quarter", or "ttm".
        keep_keys (Optional[List[str]]): List of data keys to retain in the output. If omitted, defaults will be used.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of its income statement entries.
            Each entry is a dict containing only the requested keys (or all keys if keep_keys is None).
    """
    return await get_financial_statements(tickers, "income", time_period, keep_keys)


@function_tool
async def get_ticker_balance_sheet_statement(
    tickers: List[str], time_period: str = "ttm", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    f"""
    Fetches historical balance sheet statements for stock tickers.
    Includes metrics: {', '.join(key_balance_sheet)}.
    Available for annual, quarter, and trailing twelve months (ttm).

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).
        time_period (str): Time period for the data: "annual", "quarter", or "ttm".
        keep_keys (Optional[List[str]]): List of data keys to retain in the output. If omitted, defaults will be used.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of its balance sheet entries.
            Each entry is a dict containing only the requested keys (or all keys if keep_keys is None).
    """
    return await get_financial_statements(tickers, "balance", time_period, keep_keys)


@function_tool
async def get_ticker_cash_flow_statement(
    tickers: List[str], time_period: str = "ttm", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    f"""
    Obtains historical cash flow statements for stock tickers.
    Key items: {', '.join(key_cash_flow)}.
    Available for annual, quarter, and trailing twelve months (ttm).

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).
        time_period (str): Time period for the data: "annual", "quarter", or "ttm".
        keep_keys (Optional[List[str]]): List of data keys to retain in the output. If omitted, defaults will be used.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of its cash flow entries.
            Each entry is a dict containing only the requested keys (or all keys if keep_keys is None).
    """
    return await get_financial_statements(tickers, "cash", time_period, keep_keys)


@function_tool
async def get_ticker_ratios_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    f"""
    Retrieves various historical financial ratios for stock tickers.
    Examples: {', '.join(key_ratios)}.
    Available for annual and quarter periods.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).
        time_period (str): Time period for the data: "annual" or "quarter". For ratios, must be "annual" or "quarter".
        keep_keys (Optional[List[str]]): List of data keys to retain in the output. If omitted, defaults will be used.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping each ticker to a list of its financial ratios entries.
            Each entry is a dict containing only the requested keys (or all keys if keep_keys is None).
    Raises:
        ValueError: If time_period is not "annual" or "quarter".
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
async def get_earnings_calendar() -> List[Dict[str, Any]]:
    """
    Retrieve a list of upcoming earnings announcements including company name, ticker symbol, scheduled date,
    market capitalization, prior and estimated EPS, prior and estimated revenue, and time of release.

    Returns:
        List[Dict[str, Any]]: Filtered earnings events starting from today (max 20 items).
    """
    file_path = BASE_DIR / "earnings-calendar/data.json"
    today = datetime.today().date()
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            data = orjson.loads(await f.read())
        return [item for item in data if today <= datetime.strptime(item['date'], "%Y-%m-%d").date()][:20]
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



'''
@function_tool
async def get_stock_screener(
    rule_of_list: Optional[List[Dict[str, Any]]] = None,
    sort_by: Optional[str] = None,
    sort_order: str = "desc",
    limit: int = 10
) -> Dict[str, Any]:
    f"""
    Screen stocks based on specified financial criteria to help filter stocks 
    that meet certain thresholds (e.g., revenue > $10M, P/E ratio < 15, etc.).
    All available screening metrics: {', '.join(key_screener)}.
    
    Args:
        rule_of_list: List of filtering rules with metric e.g. 'marketCap', operator e.g '>'', and value e.g. '10B'
        sort_by: Field to sort results by
        sort_order: Sort direction ('asc' or 'desc')
        limit: Maximum number of results to return
    
    Returns:
        Dict[str, Any]: Dictionary containing matched stocks and metadata
    """
    try:
        file_path = BASE_DIR / "stock-screener/data.json"
        async with aiofiles.open(file_path, 'rb') as file:
            data = orjson.loads(await file.read())

        filtered_data = [item for item in data if item.get('exchange') != 'PNK']

        if not rule_of_list:
            result = filtered_data
        else:
            result = []
            for stock in filtered_data:
                meets_criteria = True
                for rule in rule_of_list:
                    metric = rule.get('metric')
                    operator_str = rule.get('operator', '>')
                    value = rule.get('value')

                    if not metric or metric not in stock or operator_str not in OPERATORS:
                        meets_criteria = False
                        break

                    stock_value = stock[metric]

                    if stock_value is None:
                        meets_criteria = False
                        break

                    try:
                        if not OPERATORS[operator_str](stock_value, value):
                            meets_criteria = False
                            break
                    except Exception:
                        meets_criteria = False
                        break

                if meets_criteria:
                    result.append(stock)

        if sort_by and result and sort_by in result[0]:
            result.sort(
                key=lambda x: (x.get(sort_by) is None, x.get(sort_by)),
                reverse=(sort_order.lower() == "desc")
            )

        if limit and isinstance(limit, int):
            result = result[:limit]

        filtered_result = []
        for stock in result:
            try:
                filtered_stock = {
                    "symbol": stock.get("symbol", ""),
                    "company_name": stock.get("companyName", stock.get("name", "")),
                }
                if rule_of_list:
                    metrics = {}
                    for rule in rule_of_list:
                        metric_name = rule.get('metric')
                        if metric_name and metric_name in stock:
                            metrics[metric_name] = stock[metric_name]

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
'''

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
    Retrieves the current market flow option sentiment of the S&P 500.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "marketFlow": a list of market tide entries (only those with a 'close' value).
            - "topPosNetPremium": a list of top tickers by net premium.
    """
    market_flow_data = []
    res_dict: Dict[str, Any] = {}
    try:
        with open("json/market-flow/overview.json", "rb") as file:
            data = orjson.loads(file.read())
            for item in data["marketTide"]:
                try:
                    if item.get("close"):
                        market_flow_data.append(item)
                except:
                    pass
            res_dict["marketFlow"] = market_flow_data
            res_dict["topPosNetPremium"] = data.get("topPosNetPremium", [])
            return res_dict
    except Exception as e:
        return {"error": f"Error processing market flow data: {str(e)}"}

@function_tool
async def get_congress_activity(congress_ids: Union[str, List[str]]) -> Dict[str, List[Any]]:
    """
    Retrieves and filters congressional trading activity for one or more congresspeople based on their unique IDs.
    Groups results by the office of each congressperson.

    Args:
        congress_ids (Union[str, List[str]]): Single congressperson ID or list of IDs 
            (e.g., "61b59ab669" or ["61b59ab669", "anotherID"]).

    Returns:
        Dict[str, List[Any]]: A dictionary where keys are offices (e.g., "Senate", "House"), and values are lists of activity data objects.
            If an error occurs for a specific ID, an "error_<ID>" key maps to the error message.
    """
    # If input is a single string, wrap it in a list for uniform processing
    if isinstance(congress_ids, str):
        congress_ids = [congress_ids]

    result: Dict[str, List[Any]] = {}

    for congress_id in congress_ids:
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
async def get_ticker_options_data(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch comprehensive options statistics for the most recent trading day for the specified stock tickers. 
    Includes volume, open interest, premiums, GEX/DEX, implied volatility metrics, and price changes.

    Args:
        tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping each ticker to its latest options data entry,
        filtered to exclude 'price' and 'changesPercentage' fields.
    """
    base_dir = BASE_DIR / "options-historical-data/companies"
    
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    filtered_results: Dict[str, Dict[str, Any]] = {}
    for ticker, data in zip(tickers, results):
        if data is not None and data:
            # Filter out 'price' and 'changesPercentage' fields, then take the most recent entry
            latest_entry = data[0]
            filtered_entry = {k: v for k, v in latest_entry.items() if k not in ['price', 'changesPercentage']}
            filtered_results[ticker] = filtered_entry
                
    return filtered_results


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
    Retrieves a snapshot of statistical data for a list of stock ticker symbols.
    This includes key statistics such as: {', '.join(key_statistics)}.

    Args:
        tickers (List[str]): List of stock ticker symbols to analyze.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping each ticker to its statistics.
    """
    base_dir = BASE_DIR / "statistics"
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    return {ticker: result for ticker, result in zip(tickers, results)}


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
    This includes key data such as: {', '.join(key_metrics)}.

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



'''
async def get_historical_price(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    data = await get_ticker_specific_data(tickers, "historical-price/max")
    result = {}

    for ticker, price_list in data.items():
        filtered_data = []
        for item in price_list:
            try:
                filtered_data.append({'date': item['time'], 'value': item['close']})
            except:
                pass
        result[ticker] = filtered_data

    return result
'''



#Testing purposes
#data = asyncio.run(get_congress_activity(['18f16e3014','a9c12796a0']))
#print(data)