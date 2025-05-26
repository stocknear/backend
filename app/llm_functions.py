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
async def get_ticker_income_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Get income statements for multiple companies."""
    return await get_financial_statements(tickers, "income", time_period, keep_keys)

async def get_ticker_balance_sheet_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Get balance sheet statements for multiple companies."""
    return await get_financial_statements(tickers, "balance", time_period, keep_keys)

async def get_ticker_cash_flow_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Get cash flow statements for multiple companies."""
    return await get_financial_statements(tickers, "cash", time_period, keep_keys)

async def get_ticker_ratios_statement(
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

async def get_ticker_hottest_options_contracts(tickers: List[str], category: str = "volume") -> Dict[str, List[Dict[str, Any]]]:
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

async def get_ticker_short_data(tickers: List[str]) -> Dict[str, Any]:
    """Get short interest data for multiple companies."""
    return await get_ticker_specific_data(tickers, "share-statistics")

async def get_why_priced_moved(tickers: List[str]) -> Dict[str, Any]:
    """Get data explaining price movements for multiple stocks."""
    return await get_ticker_specific_data(tickers, "wiim/company")

async def get_ticker_business_metrics(tickers: List[str]) -> Dict[str, Any]:
    """Get business metrics including revenue breakdowns for multiple stocks."""
    return await get_ticker_specific_data(tickers, "business-metrics")

async def get_ticker_analyst_estimate(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Get forward-looking analyst estimates for multiple stocks."""
    def filter_by_year(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter analyst estimates to only include current year or later."""
        return [entry for entry in data if entry.get("date", 0) >= current_year]
    
    result = await get_ticker_specific_data(tickers, "analyst-estimate", filter_by_year)
    return {ticker: data for ticker, data in result.items() if data}  # Remove empty results

async def get_earnings_calendar() -> List[Dict[str, Any]]:
    file_path = BASE_DIR / "earnings-calendar/data.json"
    today = datetime.today().date()
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            data = orjson.loads(await f.read())

        # Filter data between today and upper_date using list comprehension
        return [item for item in data if today <= datetime.strptime(item['date'], "%Y-%m-%d").date()][:20]

    except FileNotFoundError:
        return []
    except (orjson.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing earnings calendar: {e}")
        return []

async def get_economic_calendar() -> List[Dict[str, Any]]:
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

async def get_ticker_earnings_price_reaction(tickers: List[str]) -> Dict[str, Any]:
    """Get historical earnings price reactions for multiple stocks."""
    return await get_ticker_specific_data(tickers, "earnings/past")

async def get_ticker_next_earnings(tickers: List[str]) -> Dict[str, Any]:
    """Get upcoming earnings dates and estimates for multiple stocks."""
    return await get_ticker_specific_data(tickers, "earnings/next")




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




async def get_latest_options_flow_feed(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    return await get_feed_data(
        tickers=tickers,
        file_path=BASE_DIR / "options-flow/feed/data.json",
        filter_keys={'aggressor_ind', "exchange","tradeCount", "trade_count", "underlying_type", "description"},
        sort_key="cost_basis"
    )

async def get_latest_dark_pool_feed(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    return await get_feed_data(
        tickers=tickers,
        file_path=BASE_DIR / "dark-pool/feed/data.json",
        filter_keys={"assetType", "sector", "trackingID"},
        sort_key="premium",
    )

async def get_ticker_news(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    
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




async def get_ticker_analyst_rating(tickers: List[str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
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

async def get_top_gainers():
    try:
        with open(f"json/market-movers/markethours/gainers.json", 'rb') as file:
            data = orjson.loads(file.read())['1D'][:10]
            return data
    except Exception as e:
        return f"Error processing top gainers data: {str(e)}"

async def get_top_losers():
    try:
        with open(f"json/market-movers/markethours/losers.json", 'rb') as file:
            data = orjson.loads(file.read())['1D'][:10]
            return data
    except Exception as e:
        return f"Error processing top losers data: {str(e)}"

async def get_top_active_stocks():
    try:
        with open(f"json/market-movers/markethours/active.json", 'rb') as file:
            data = orjson.loads(file.read())['1D'][:10]
            return data
    except Exception as e:
        return f"Error processing most active stock data: {str(e)}"

async def get_potus_tracker():
    try:
        with open(f"json/tracker/potus/data.json", 'rb') as file:
            data = orjson.loads(file.read())
            return data
    except Exception as e:
        return f"Error processing potus tracker data: {str(e)}"

async def get_insider_tracker():
    try:
        with open(f"json/tracker/insider/data.json", 'rb') as file:
            data = orjson.loads(file.read())
            return data[:20]
    except Exception as e:
        return f"Error processing potus tracker data: {str(e)}"

async def get_congress_tracker():
    try:
        with open(f"json/congress-trading/rss-feed/data.json", 'rb') as file:
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
        return f"Error processing congress tracker data: {str(e)}"

async def get_analyst_tracker():
    try:
        with open(f"json/analyst/flow-data.json", 'rb') as file:
            data = orjson.loads(file.read())

            fields_to_remove = {
                "analystId",
            }

            data = [{k: v for k, v in entry.items() if k not in fields_to_remove} for entry in data]
            return data[:20]
    except Exception as e:
        return f"Error processing analyst tracker data: {str(e)}"

async def get_market_news():
    try:
        with open(f"json/market-news/general-news.json", 'rb') as file:
            data = orjson.loads(file.read())

            fields_to_remove = {
                "site",
                "url",
                "image",
            }

            data = [{k: v for k, v in entry.items() if k not in fields_to_remove} for entry in data]
            return data[:30]
    except Exception as e:
        return f"Error processing market news data: {str(e)}"

async def get_ticker_congress_activity(congress_id):
    try:
        with open(f"json/congress-trading/politician-db/{congress_id}.json", 'rb') as file:
            data = orjson.loads(file.read())

            if "history" in data:
                fields_to_remove = {
                    "assetDescription",
                    "firstName",
                    "lastName",
                    "office",
                    "capitalGainsOver200USD",
                    "comment",
                    "link",
                    "id"
                }

                data["history"] = [
                    {k: v for k, v in entry.items() if k not in fields_to_remove}
                    for entry in data["history"]
                ]
            return data
    except Exception as e:
        return f"Error processing congress activity data: {str(e)}"

async def get_market_flow():
    market_flow_data = []
    top_net_premium_tickers = []
    res_dict = {}
    try:
        with open(f"json/market-flow/overview.json", 'rb') as file:
            data = orjson.loads(file.read())
            for item in data['marketTide']:
                try:
                    if item['close']:
                        market_flow_data.append(item)
                except:
                    pass
            res_dict['marketFlow'] = market_flow_data
            res_dict['topPosNetPremium'] = data['topPosNetPremium']
            return res_dict
    except Exception as e:
        return f"Error processing market flow data: {str(e)}"

async def get_ticker_quote(tickers: List[str]) -> Dict[str, Any]:
    """Get upcoming earnings dates and estimates for multiple stocks."""
    return await get_ticker_specific_data(tickers, "quote")


async def get_ticker_insider_trading(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    
    base_dir = BASE_DIR / "insider-trading/history"
    
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    filtered_results = {}
    for ticker, data in zip(tickers, results):
        if data is not None:
            # Filter for recent news only
            filtered_data = [
                {k: v for k, v in item.items() if k not in ['companyCik', 'url']} for item in data]
            if filtered_data:  # Only add if there's data
                filtered_results[ticker] = filtered_data[:50]
                
    return filtered_results

async def get_ticker_shareholders(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    final_res = {}
    
    for path in ['shareholders', 'ownership-stats']:
        base_dir = BASE_DIR / path  # Use the path to differentiate file location
        
        tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
        results = await asyncio.gather(*tasks)

        filtered_results = {}
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
            final_res[path] = filtered_results
                
    return final_res


async def get_ticker_options_data(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    
    base_dir = BASE_DIR / "options-historical-data/companies"
    
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    filtered_results = {}
    for ticker, data in zip(tickers, results):
        if data is not None:
            # Filter for recent news only
            filtered_data = [
                {k: v for k, v in item.items() if k not in ['price', 'changesPercentage']} for item in data]
            if filtered_data:  # Only add if there's data
                filtered_results[ticker] = filtered_data[0]
                
    return filtered_results

async def get_ticker_max_pain(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    
    base_dir = BASE_DIR / "max-pain"
    
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    return {ticker: result for ticker, result in zip(tickers, results) if result}

async def get_ticker_open_interest_by_strike_and_expiry(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    final_res = {}
    today = datetime.today().date()

    for category_type in ['expiry', 'strike']:
        base_dir = BASE_DIR / f"oi/{category_type}"
        tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        
        res = {}
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


async def get_ticker_unusual_activity(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    base_dir = BASE_DIR / "unusual-activity"
    
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    sorted_results = {}
    for ticker, result in zip(tickers, results):
        if result:
            sorted_result = sorted(result, key=lambda x: x['date'], reverse=True)
            sorted_results[ticker] = sorted_result[:10]

    return sorted_results

async def get_ticker_dark_pool(tickers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    final_res = {}

    for category_type in ['companies', 'price-level']:
        base_dir = BASE_DIR / f"dark-pool/{category_type}"
        tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        
        res = {}
        for ticker, result in zip(tickers, results):
            if not result:
                continue
        if category_type == 'companies':
            res[ticker] = result[-1]
            final_res[f"volume-summary"] = res
        else:
            res[ticker] = result
            final_res['hottest_trades_and_price_level'] = res

    return final_res

async def get_ticker_dividend(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    base_dir = BASE_DIR / "dividends/companies"

    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    res = {}
    for ticker, result in zip(tickers, results):
        history = result.get('history', [])
        if history:
            # Replace list of history with only the most recent entry
            result['history'] = history[0]
            res[ticker] = result

    return res

async def get_ticker_statistics(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    base_dir = BASE_DIR / "statistics"

    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    res = {}
    for ticker, result in zip(tickers, results):
        res[ticker] = result

    return res

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
            "name": "get_ticker_income_statement",
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
            "name": "get_ticker_balance_sheet_statement",
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
            "name": "get_ticker_cash_flow_statement",
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
            "name": "get_ticker_ratios_statement",
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
            "name": "get_ticker_hottest_options_contracts",
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
            "name": "get_ticker_short_data",
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
            "name": "get_ticker_business_metrics",
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
            "name": "get_ticker_analyst_estimate",
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
            "name": "get_ticker_earnings_price_reaction",
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
            "name": "get_ticker_next_earnings",
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
            "description": "Retrieves the top options flow orders with the highest premiums for multiple stocks, highlighting activity from hedge funds and major traders.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"]). If no ticker are available set it to an empty list []."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_latest_dark_pool_feed",
            "description": "Retrieves the top dark pool trades for multiple stocks, sorted by the average price paid, highlighting significant activity from hedge funds and major traders.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"]). If no ticker are available set it to an empty list []."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_ticker_news",
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
            "name": "get_ticker_analyst_rating",
            "description": "Retrieves the latest analyst ratings for multiple stocks.",
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
        {
            "name": "get_top_gainers",
            "description": "Retrieves a list of stocks with the highest percentage gains for the current trading day. Returns stocks ranked by their daily price increase percentage, showing which securities have performed best today.",
            "parameters": {},
        },
        {
            "name": "get_top_active_stocks", 
            "description": "Retrieves a list of stocks with the largest trading volume for the current trading day. Returns stocks ranked by their daily volume, showing which securities have traded the most today.",
            "parameters": {},
        },
        {
            "name": "get_potus_tracker",
            "description": "Get the latest POTUS tracker data, including the latest presidential schedule, Truth Social posts, executive orders, and the performance of the S&P 500 (SPY) since the current president's inauguration.",
            "parameters": {}
        },
        {
            "name": "get_insider_tracker",
            "description": "Get the latest insider trading activity, including recent stock sales or purchases by company executives, along with relevant company information such as symbol, price, market cap, and filing date.",
            "parameters": {}
        },
        {
            "name": "get_congress_tracker",
            "description": "Get the latest congressional stock trading disclosures, including transactions by members of Congress or their spouses, with details such as ticker, transaction type, amount, representative name, and disclosure dates.",
            "parameters": {}
        },
        {
            "name": "get_analyst_tracker",
            "description": "Get the latest analyst stock ratings, including upgrades, downgrades, maintained ratings, price targets, analyst names, firms, and expected upside.",
            "parameters": {}
        },
        {
            "name": "get_market_flow",
            "description": "Retrieves the current market flow option sentiment of the S&P 500.",
            "parameters": {}
        },
        {
            "name": "get_market_news",
            "description": "Retrieves the latest general news for the stock market.",
            "parameters": {}
        },
        {
            "name": "get_earnings_calendar",
            "description": "Retrieves a list of upcoming earnings announcements, including company name, ticker symbol, scheduled date, market capitalization, prior and estimated earnings per share (EPS), prior and estimated revenue, and the time of release (e.g., 'bmo' for before market open).",
            "parameters": {},
        },
        {
            "name": "get_economic_calendar",
            "description": "Retrieve a list of upcoming USA economic events, including event name, scheduled date and time, previous, consensus, and actual values (if available), event importance level, and associated country code for macroeconomic analysis and market forecasting.",
            "parameters": {},
        },
        {
            "name": "get_ticker_congress_activity",
            "description": (
                f"Retrieves and filters congressional trading activity for a specific congressperson based on the id which can be found here {key_congress_db}. E.g. Nancy Pelosi, Marjorie Greene, Rob Bresnahan etc."
                "Removes personally identifying and extraneous fields from transaction history. "
                "Useful for analyzing political trading patterns without sensitive details."
            ),
            "parameters": {
                "congress_id": {
                    "type": "string",
                    "description": "Unique identifier for a congressperson."
                },
            },
            "required": ["congress_id"]
        },
        {
            "name": "get_ticker_quote",
            "description": "Retrieves the most recent stock quote data for one or more ticker symbols. Returns real-time information including the latest trading price, percentage change, trading volume, day high and low, 52-week high and low, market capitalization, previous close, earnings per share (EPS), price-to-earnings (P/E) ratio, as well as current ask and bid prices. Ideal for use cases requiring timely and detailed market updates.",
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
            "name": "get_ticker_insider_trading",
            "description": "Fetch detailed insider trading records—such as buys, sells, grant dates, and volumes—executed by corporate insiders (officers, directors, major shareholders) for the specified stock tickers.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
        "name": "get_ticker_shareholders",
        "description": "Fetch current institutional and major shareholder data for the specified stock tickers, including top institutional holders and ownership statistics such as investor counts, total invested value, and put/call ratios.",
        "parameters": {
            "tickers": {
                "type": "array",
                "items": { "type": "string" },
                "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
        "required": ["tickers"]
        },
        {
            "name": "get_ticker_options_data",
            "description": "Fetch comprehensive options statistics for the most recent trading day for the specified stock tickers. Includes volume, open interest, premiums, GEX/DEX, implied volatility metrics, and price changes.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_ticker_max_pain",
            "description": "Retrieve max pain analysis for multiple stock tickers, including expiration dates, strike prices, call and put payouts, and the calculated max pain point per expiration.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of stock ticker symbols to analyze (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_ticker_open_interest_by_strike_and_expiry",
            "description": "Fetch and aggregate options open interest data for one or more equity tickers, returning expiryData with total call and put OI grouped by each expiration date and strikeData with call and put OI broken down by individual strike price, where every entry includes its call_oi, put_oi, and the associated expiry or strike value for seamless analysis across both dimensions.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of stock ticker symbols to analyze (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_ticker_unusual_activity",
            "description": "Retrieve recent unusual options activity for one or more stock tickers, including large trades, sweeps, and high-premium orders.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of stock ticker symbols to analyze (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_ticker_dark_pool",
            "description": "Retrieves dark pool trading data and related analytics for a list of stock ticker symbols, including volume summaries, hottest trades, price levels, and trend data.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of stock ticker symbols to analyze (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_ticker_dividend",
            "description": "Retrieve dividend data and related metrics for a list of stock ticker symbols, including payout frequency, annual dividend, dividend yield, payout ratio, and dividend growth. Also returns historical records with detailed information such as declaration date, record date, payment date, and adjusted dividend amount.",
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of stock ticker symbols to analyze (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
        },
        {
            "name": "get_ticker_statistics",
            "description": (
                "Retrieves a snapshot of statistical data for a list of stock ticker symbols. "
                f"This includes key statistics such as: {', '.join(key_statistics)}."),
            "parameters": {
                "tickers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of stock ticker symbols to analyze (e.g., [\"AAPL\", \"GOOGL\"])."
                }
            },
            "required": ["tickers"]
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
#data = asyncio.run(get_ticker_statistics(['AMD']))
#print(data)