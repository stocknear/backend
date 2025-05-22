import os
import asyncio
import aiofiles
import orjson
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar, Set, Tuple, cast


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


# Load environment variables
load_dotenv()

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
async def get_income_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Get income statements for multiple companies."""
    return await get_financial_statements(tickers, "income", time_period, keep_keys)

async def get_balance_sheet_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Get balance sheet statements for multiple companies."""
    return await get_financial_statements(tickers, "balance", time_period, keep_keys)

async def get_cash_flow_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Get cash flow statements for multiple companies."""
    return await get_financial_statements(tickers, "cash", time_period, keep_keys)

async def get_ratios_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Get financial ratios for multiple companies."""
    if time_period not in ["annual", "quarter"]:
        raise ValueError(f"Invalid time_period '{time_period}'. For ratios, must be 'annual' or 'quarter'.")
    
    base_dir = BASE_DIR / "financial-statements/ratios" / time_period
    tasks = [_load_and_filter(base_dir / f"{ticker}.json", keep_keys) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    return {ticker: result for ticker, result in zip(tickers, results) if result}

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

async def get_hottest_options_contracts(tickers: List[str], category: str = "volume") -> Dict[str, List[Dict[str, Any]]]:
    """Get the hottest options contracts based on volume or open interest."""
    if category not in ["volume", "openInterest"]:
        raise ValueError(f"Invalid category '{category}'. For hottest contracts, must be 'volume' or 'openInterest'.")
    
    base_dir = BASE_DIR / "hottest-contracts/companies"
    
    # Create tasks for each ticker
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    
    # Gather all results concurrently
    results = await asyncio.gather(*tasks)

    # Return top 5 for each ticker
    return {
        ticker: result[category][:5] 
        for ticker, result in zip(tickers, results) 
        if result is not None and category in result
    }

async def get_company_data(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Get company overview data for multiple companies."""
    def process_company_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove specific fields from company data."""
        # Create a copy to avoid modifying the original
        result = data.copy()
        result.pop('website', None)
        result.pop('financialPerformance', None)
        return result
    
    return await get_ticker_specific_data(tickers, "stockdeck", process_company_data)

async def get_short_data(tickers: List[str]) -> Dict[str, Any]:
    """Get short interest data for multiple companies."""
    return await get_ticker_specific_data(tickers, "share-statistics")

async def get_why_priced_moved(tickers: List[str]) -> Dict[str, Any]:
    """Get data explaining price movements for multiple stocks."""
    return await get_ticker_specific_data(tickers, "wiim/company")

async def get_business_metrics(tickers: List[str]) -> Dict[str, Any]:
    """Get business metrics including revenue breakdowns for multiple stocks."""
    return await get_ticker_specific_data(tickers, "business-metrics")

async def get_analyst_estimate(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Get forward-looking analyst estimates for multiple stocks."""
    def filter_by_year(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter analyst estimates to only include current year or later."""
        return [entry for entry in data if entry.get("date", 0) >= current_year]
    
    result = await get_ticker_specific_data(tickers, "analyst-estimate", filter_by_year)
    return {ticker: data for ticker, data in result.items() if data}  # Remove empty results

async def get_earnings_calendar(upper_threshold: str = "") -> List[Dict[str, Any]]:
    """Get earnings events for stocks within a date range."""
    file_path = BASE_DIR / "earnings-calendar/data.json"
    
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            data = orjson.loads(await f.read())

        # Set upper date threshold
        upper_date = today if not upper_threshold else datetime.strptime(upper_threshold, "%Y-%m-%d").date()

        # Filter data between today and upper_date using list comprehension
        return [
            item for item in data
            if today <= datetime.strptime(item['date'], "%Y-%m-%d").date() <= upper_date
        ]

    except FileNotFoundError:
        return []
    except (orjson.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing earnings calendar: {e}")
        return []

async def get_top_rating_stocks():
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

async def get_earnings_price_reaction(tickers: List[str]) -> Dict[str, Any]:
    """Get historical earnings price reactions for multiple stocks."""
    return await get_ticker_specific_data(tickers, "earnings/past")

async def get_next_earnings(tickers: List[str]) -> Dict[str, Any]:
    """Get upcoming earnings dates and estimates for multiple stocks."""
    return await get_ticker_specific_data(tickers, "earnings/next")



async def get_feed_data(
    tickers: List[str], 
    file_path: Path,
    filter_keys: Set[str],
    sort_key: str,
    limit: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generic function to get feed data (options flow, dark pool) for multiple stocks.
    
    Args:
        tickers: List of ticker symbols
        file_path: Path to the feed data file
        filter_keys: Keys to exclude from the results
        sort_key: Key to sort results by
        limit: Maximum number of items to return per ticker
        
    Returns:
        Dictionary mapping tickers to their feed data
    """
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            data = orjson.loads(await f.read())

        # Group and filter items by ticker
        filtered_results = defaultdict(list)
        for item in data:
            ticker = item.get("ticker")
            if ticker in tickers:
                # Exclude specific keys
                cleaned_item = {k: v for k, v in item.items() if k not in filter_keys}
                filtered_results[ticker].append(cleaned_item)

        # Sort by specified key and take top N items
        result = {}
        for ticker, items in filtered_results.items():
            result[ticker] = sorted(
                items,
                key=lambda x: x.get(sort_key, 0),
                reverse=True
            )[:limit]

        return result

    except FileNotFoundError:
        return {}
    except (orjson.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing feed data: {e}")
        return {}

async def get_latest_options_flow_feed(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Get the top 5 options flow orders for multiple stocks."""
    return await get_feed_data(
        tickers=tickers,
        file_path=BASE_DIR / "options-flow/feed/data.json",
        filter_keys={'aggresor_ind', "exchange", "tradeCount", "underlying_type", "description"},
        sort_key="cost_basis"
    )

async def get_latest_dark_pool_feed(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Get the top 5 dark pool trades for multiple stocks."""
    return await get_feed_data(
        tickers=tickers,
        file_path=BASE_DIR / "dark-pool/feed/data.json",
        filter_keys={"assetType", "sector", "trackingID", "ticker"},
        sort_key="premium"
    )

async def get_market_news(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Get recent news for multiple stocks."""
    base_dir = BASE_DIR / "market-news/companies"
    
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    filtered_results = {}
    for ticker, data in zip(tickers, results):
        if data is not None:
            # Filter for recent news only
            filtered_data = [
                {k: v for k, v in item.items() if k not in ['image', 'symbol', 'url', 'site']}
                for item in data
                if 'publishedDate' in item and 
                week_ago <= datetime.strptime(item['publishedDate'], "%Y-%m-%d %H:%M:%S").date()
            ]
            if filtered_data:  # Only add if there's data
                filtered_results[ticker] = filtered_data
                
    return filtered_results

async def get_analyst_ratings(tickers: List[str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Get recent analyst ratings and summary for multiple stocks."""
    history_dir = BASE_DIR / "analyst" / "history"
    summary_dir = BASE_DIR / "analyst" / "summary" / "all_analyst"

    # Launch two sets of tasks: history (detailed ratings) and summary
    history_tasks = [fetch_ticker_data(t, history_dir) for t in tickers]
    summary_tasks = [fetch_ticker_data(t, summary_dir) for t in tickers]

    # Gather both lists of results concurrently
    histories, summaries = await asyncio.gather(
        asyncio.gather(*history_tasks),
        asyncio.gather(*summary_tasks)
    )

    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for ticker, hist_data, summ_data in zip(tickers, histories, summaries):
        results[ticker] = {}

        # Process detailed history: remove 'analystId', limit to most recent 30
        if hist_data:
            cleaned_history = [
                {k: v for k, v in entry.items() if k != 'analystId'}
                for entry in hist_data
            ][:30]
            results[ticker]['analyst_rating'] = cleaned_history

        # Process summary: remove 'recommendationList' from summary dict
        if isinstance(summ_data, dict):
            summary_clean = {
                k: v for k, v in summ_data.items() if k != 'recommendationList' and k != 'pastPriceList'
            }
            results[ticker]['rating_summary'] = [summary_clean]  # wrap in a list to match type hint
    return results

async def get_stock_screener(
    rule_of_list: Optional[List[Dict[str, Any]]] = None, 
    sort_by: Optional[str] = None, 
    sort_order: str = "desc", 
    limit: int = 10
) -> Dict[str, Any]:
    """
    Screen stocks based on specified criteria.
    
    Args:
        rule_of_list: List of filtering rules
        sort_by: Field to sort results by
        sort_order: Sort direction ('asc' or 'desc')
        limit: Maximum number of results
        
    Returns:
        Dictionary with matched stocks and count
    """
    try:
        file_path = BASE_DIR / "stock-screener/data.json"
        async with aiofiles.open(file_path, 'rb') as file:
            data = orjson.loads(await file.read())

        # Initial filter to exclude PNK exchange
        filtered_data = [item for item in data if item.get('exchange') != 'PNK']

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
                    
                    # Apply comparison
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

# Function definition mapping for LLM use
def get_function_definitions():
    """Return JSON schema definitions for all functions."""
    templates = [
        {
          "name": "get_stock_screener",
          "description": "Retrieves stock data based on specified financial criteria to help filter stocks that meet certain thresholds.",
          "parameters": {
            "rule_of_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "description": "The financial metric to filter by."},
                        "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "==", "!="], "description": "The comparison operator."},
                        "value": {"type": ["number", "string"], "description": "The value to compare against."}
                    },
                    "required": ["metric", "value"]
                },
                "description": "List of screening rules to filter stocks."
            },
            "sort_by": {
                "type": "string",
                "description": "Field name to sort the results by."
            },
            "sort_order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "default": "desc",
                "description": "Sort order for the results: 'asc' for ascending or 'desc' for descending."
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "description": "Maximum number of results to return."
            }
          }
        },
        {
            "name": "get_income_statement",
            "description": "Retrieves historical income statements for a list of stock tickers.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols."
                },
                "time_period": {
                    "type": "string",
                    "enum": ["annual", "quarter", "ttm"],
                    "description": "Time period for the data: annual, quarter, ttm."
                },
                "keep_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of data keys to retain in the output."
                }
            },
            "required": ["tickers", "time_period"]
        },
        # Add remaining function definitions similarly
    ]

    # Convert templates to function definitions
    definitions = []
    for tpl in templates:
        func_def = {
            "name": tpl["name"],
            "description": tpl["description"],
            "strict_json_schema": True,
            "parameters": {
                "type": "object",
                "properties": tpl["parameters"],
            }
        }
        
        # Add required fields if specified
        if "required" in tpl:
            func_def["parameters"]["required"] = tpl["required"]

        definitions.append(func_def)

    return definitions


def get_function_definitions():
    templates = [
        {
          "name": "get_stock_screener",
          "description": f"Retrieves stock data based on specified financial criteria to help filter stocks that meet certain thresholds (e.g., revenue > $10M, P/E ratio < 15, etc.) All rules are defined here: {', '.join(key_screener)}.",
          "parameters": {
            "rule_of_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "description": "The financial metric to filter by (e.g., 'marketCap', 'priceToEarningsRatio', 'revenue')."},
                        "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "==", "!="], "description": "The comparison operator."},
                        "value": {"type": ["number", "string"], "description": "The value to compare against."} # Value can be number or string depending on metric
                    },
                    "required": ["metric", "value"]
                },
                "description": "List of screening rules to filter stocks (e.g., [{\"metric\": \"marketCap\", \"operator\": \">\", \"value\": 100}, {\"metric\": \"priceToEarningsRatio\", \"operator\": \"<\", \"value\": 10}])."
            },
            "sort_by": {
                "type": "string",
                "description": "Field name to sort the results by (e.g., \"marketCap\", \"volume\", \"price\")."
            },
            "sort_order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "default": "desc",
                "description": "Sort order for the results: 'asc' for ascending or 'desc' for descending."
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "description": "Maximum number of results to return."
            }
          }
        },
        {
            "name": "get_income_statement",
            "description": (
                "Retrieves historical income statements (profit and loss) for a list of stock tickers. "
                f"Key metrics include: {', '.join(key_income)}. "
                "Available for annual, quarter, and trailing twelve months (ttm)."
            ),
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                },
                "time_period": {
                    "type": "string",
                    "enum": ["annual", "quarter", "ttm"],
                    "description": "Time period for the data: annual, quarter, ttm."
                },
                "keep_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of data keys to retain in the output. If omitted, defaults will be used."
                }
            },
            "required": ["tickers", "time_period"]
        },
        {
            "name": "get_balance_sheet_statement",
            "description": (
                "Fetches historical balance sheet statements for stock tickers. "
                f"Includes metrics: {', '.join(key_balance_sheet)}. "
                "Available for annual, quarter, and trailing twelve months (ttm)."
            ),
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                },
                "time_period": {
                    "type": "string",
                    "enum": ["annual", "quarter", "ttm"],
                    "description": "Time period for the data: annual, quarter, ttm."
                },
                "keep_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of data keys to retain in the output. If omitted, defaults will be used."
                }
            },
            "required": ["tickers", "time_period"]
        },
        {
            "name": "get_cash_flow_statement",
            "description": (
                "Obtains historical cash flow statements for stock tickers. "
                f"Key items: {', '.join(key_cash_flow)}. "
                "Available for annual, quarter, and trailing twelve months (ttm)."
            ),
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                },
                "time_period": {
                    "type": "string",
                    "enum": ["annual", "quarter", "ttm"],
                    "description": "Time period for the data: annual, quarter, ttm."
                },
                "keep_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of data keys to retain in the output. If omitted, defaults will be used."
                }
            },
            "required": ["tickers", "time_period"]
        },
        {
            "name": "get_ratios_statement",
            "description": (
                "Retrieves various historical financial ratios for stock tickers. "
                f"Examples: {', '.join(key_ratios)}. "
                "Available for annual and quarter periods."
            ),
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                },
                "time_period": {
                    "type": "string",
                    "enum": ["annual", "quarter"],
                    "description": "Time period for the data: annual, quarter."
                },
                "keep_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of data keys to retain in the output. If omitted, defaults will be used."
                }
            },
            "required": ["tickers", "time_period"]
        },
        {
            "name": "get_hottest_options_contracts",
            "description": (
                "Retrieves the hottest options contracts for stock tickers. "
                "Returns contracts sorted by either volume or open interest."
            ),
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                },
                "category": {
                    "type": "string",
                    "enum": ["volume", "openInterest"],
                    "description": "Category to sort contracts by: volume or openInterest."
                }
            },
            "required": ["tickers", "category"]
        },
        {
            "name": "get_company_data",
            "description": "Fetches financial and organizational overview data for multiple companies by their stock symbols",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_short_data",
            "description": "Retrieves the most recent and historical short interest data for multiple companies using their stock ticker symbols. Not useful for filtering or sorting data.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_why_priced_moved",
            "description": "Retrieves recent data explaining the price movement of multiple stocks based on their ticker symbols.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_business_metrics",
            "description": "Fetches business metrics for multiple stocks, including revenue breakdown by sector and geographic region, based on their ticker symbols.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_analyst_estimate",
            "description": "Fetches forward-looking analyst estimates for multiple stocks, including average, low, and high projections for EPS, revenue, EBITDA, and net income. Call it always if @Analyst is in the user query",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_earnings_calendar",
            "description": "Fetches today's and upcoming earnings events for stocks within a specified date range (defaulting to today if no upper threshold is provided).",
            "parameters": {
                "upper_threshold": {
                    "type": "string",
                    "description": "Optional upper date threshold in YYYY-MM-DD format. If not provided, defaults to today."
                }
            },
        },
        {
            "name": "get_earnings_price_reaction",
            "description": "Fetches past earnings price reactions before and after earnings releases of multiple stocks based on their ticker symbols",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_next_earnings",
            "description": "Retrieves the upcoming earnings dates for multiple stocks, along with EPS and revenue estimates. Also includes prior EPS and revenue figures for comparison.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_latest_options_flow_feed",
            "description": "Retrieves the top 5 options flow orders with the highest premiums for multiple stocks, highlighting activity from hedge funds and major traders.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_latest_dark_pool_feed",
            "description": "Retrieves the top 5 dark pool trades for multiple stocks, sorted by the average price paid, highlighting significant activity from hedge funds and major traders.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_market_news",
            "description": "Retrieves the latest news for multiple stocks.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_analyst_ratings",
            "description": "Retrieves the latest analyst ratings for multiple stocks. Call it always if @Analyst is in the user query",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_top_rating_stocks",
            "description": "Retrieves the top rating stocks from analyst.",
            "parameters": {},
        },
    ]

    definitions = []
    for tpl in templates:
        func_def = {
            "name": tpl.get("name", ""),
            "description": tpl.get("description", ""),
            "strict_json_schema": True,
            "parameters": {
                "type": "object",
                "properties": tpl.get("parameters", {}),
            }
        }

        # Only include "required" if explicitly provided
        if "required" in tpl:
            func_def["parameters"]["required"] = tpl["required"]

        definitions.append(func_def)

    return definitions


#Testing purposes

#data = asyncio.run(get_analyst_ratings(tickers=['AMD','TSLA']))
#print(data)

