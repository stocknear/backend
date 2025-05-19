import os
import json
import requests
import aiohttp
import asyncio
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime, date, timedelta
import orjson
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiofiles

load_dotenv()


# Default keys to remove
DEFAULT_FINANCIAL_REMOVE_KEYS = {"symbol", "reportedCurrency", "acceptedDate", "cik", "filingDate"}
# Keys always required if filtering
FINANCIAL_REQUIRED_KEYS = {"date", "fiscalYear", "period"}
# Mapping statement types to directory names
STATEMENT_DIRS = {
    "income": "income-statement",
    "balance": "balance-sheet-statement",
    "cash": "cash-flow-statement",
}

current_year = datetime.now().year
week_ago = datetime.now().date() - timedelta(days=5)

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

async def fetch_ticker_data(ticker, base_dir):
    file_path = base_dir / f"{ticker}.json"
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            content = await f.read()
            data = orjson.loads(content)
            return data
    except FileNotFoundError:
        return None
    except (orjson.JSONDecodeError, KeyError) as e:
        print(f"Error processing {ticker}: {e}")
        return None
    

async def _load_and_filter(
    file_path: Path,
    keep_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            raw = orjson.loads(await f.read())
    except Exception:
        return []

    if keep_keys:
        keys_to_keep = set(keep_keys) | FINANCIAL_REQUIRED_KEYS
        return [ {k: v for k, v in entry.items() if k in keys_to_keep} for entry in raw ]
    else:
        return [ {k: v for k, v in entry.items() if k not in DEFAULT_FINANCIAL_REMOVE_KEYS} for entry in raw ]

async def get_financial_statements(
    tickers: List[str],
    statement_type: str = "income",
    time_period: str = "annual",
    keep_keys: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    dir_name = STATEMENT_DIRS.get(statement_type)
    if not dir_name:
        raise ValueError(f"Invalid statement_type '{statement_type}'. Must be one of {list(STATEMENT_DIRS)}.")

    base_dir = Path("json/financial-statements") / dir_name / time_period
    tasks = []
    for ticker in tickers:
        file_path = base_dir / f"{ticker}.json"
        tasks.append(_load_and_filter(file_path, keep_keys))

    results = await asyncio.gather(*tasks)
    return {ticker: result for ticker, result in zip(tickers, results)}



async def get_income_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    return await get_financial_statements(tickers, "income", time_period, keep_keys)

async def get_balance_sheet_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    return await get_financial_statements(tickers, "balance", time_period, keep_keys)

async def get_cash_flow_statement(
    tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    return await get_financial_statements(tickers, "cash", time_period, keep_keys)


async def get_ratios_statement(tickers: List[str], time_period: str = "annual", keep_keys: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
   
    if time_period not in ["annual", "quarter"]:
        raise ValueError(f"Invalid time_period '{time_period}'. For ratios, must be 'annual' or 'quarter'.")
    
    base_dir = Path("json/financial-statements/ratios") / time_period
    tasks = []
    for ticker in tickers:
        file_path = base_dir / f"{ticker}.json"
        tasks.append(_load_and_filter(file_path, keep_keys))
    
    results = await asyncio.gather(*tasks)
    return {ticker: result for ticker, result in zip(tickers, results)}


async def get_hottest_options_contracts(tickers, category="volume"):
    if category not in ["volume", "openInterest"]:
        raise ValueError(f"Invalid category '{category}'. For hottest contracts, must be 'volume' or 'openInterest'.")
    
    base_dir = Path("json/hottest-contracts/companies")
    
    
    # Create tasks for each ticker
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    
    # Gather all results concurrently
    results = await asyncio.gather(*tasks)

    # Create the result dictionary
    return {ticker: result[category][:5] for ticker, result in zip(tickers, results) if result is not None}

async def get_company_data(tickers):
    base_dir = Path("json/stockdeck")
    
    # Create tasks for each ticker
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    
    # Gather all results concurrently
    results = await asyncio.gather(*tasks)
    
    # Create the result dictionary with filtered data
    filtered_results = {}
    for ticker, result in zip(tickers, results):
        if result is not None:
            # Remove website and financialPerformance fields
            if 'website' in result:
                del result['website']
            if 'financialPerformance' in result:
                del result['financialPerformance']
            filtered_results[ticker] = result
    
    return filtered_results

async def get_short_data(tickers):
    base_dir = Path("json/share-statistics")
    
    # Create tasks for each ticker
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    
    # Gather all results concurrently
    results = await asyncio.gather(*tasks)
    
    # Create the result dictionary with filtered data
    filtered_results = {}
    for ticker, result in zip(tickers, results):
        if result is not None:
            filtered_results[ticker] = result
    return filtered_results

async def get_why_priced_moved(tickers):
    base_dir = Path("json/wiim/company")
    
    # Create tasks for each ticker
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    
    # Gather all results concurrently
    results = await asyncio.gather(*tasks)
    
    # Create the result dictionary with filtered data
    filtered_results = {}
    for ticker, result in zip(tickers, results):
        if result is not None:
            filtered_results[ticker] = result
    return filtered_results

async def get_business_metrics(tickers):
    base_dir = Path("json/business-metrics")
    
    # Create tasks for each ticker
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    
    # Gather all results concurrently
    results = await asyncio.gather(*tasks)
    
    # Create the result dictionary with filtered data
    filtered_results = {}
    for ticker, result in zip(tickers, results):
        if result is not None:
            filtered_results[ticker] = result
    return filtered_results

async def get_analyst_estimate(tickers):
    base_dir = Path("json/analyst-estimate")
    current_year = datetime.now().year

    # Create tasks for each ticker
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]

    # Gather all results concurrently
    results = await asyncio.gather(*tasks)

    # Create the result dictionary with filtered data
    filtered_results = {}
    for ticker, result in zip(tickers, results):
        if result is not None:
            # Filter entries within the result that are from this year or later
            filtered_data = [
                entry for entry in result if entry.get("date", 0) >= current_year
            ]
            if filtered_data:
                filtered_results[ticker] = filtered_data

    return filtered_results


async def get_earnings_calendar(upper_threshold: str = ""):
    base_dir = Path("json/earnings-calendar/data.json")
    
    try:
        async with aiofiles.open(base_dir, mode="rb") as f:
            content = await f.read()
            data = orjson.loads(content)

            today = date.today()

            # Default upper threshold to 10 days from today if not provided or empty
            if not upper_threshold:
                upper_date = today #+ timedelta(days=10)
            else:
                upper_date = datetime.strptime(upper_threshold, "%Y-%m-%d").date()

            # Filter data between today and upper_date
            filtered = [
                item for item in data
                if today <= datetime.strptime(item['date'], "%Y-%m-%d").date() <= upper_date
            ]

            return filtered

    except FileNotFoundError:
        return None
    except (orjson.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing file: {e}")
        return None

async def get_earnings_price_reaction(tickers):
    base_dir = Path("json/earnings/past")
    current_year = datetime.now().year

    # Create tasks for each ticker
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]

    # Gather all results concurrently
    results = await asyncio.gather(*tasks)
    
    # Create the result dictionary with filtered data
    filtered_results = {}
    for ticker, result in zip(tickers, results):
        if result is not None:
            filtered_results[ticker] = result
    return filtered_results

async def get_next_earnings(tickers):
    base_dir = Path("json/earnings/next")
    current_year = datetime.now().year

    # Create tasks for each ticker
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]

    # Gather all results concurrently
    results = await asyncio.gather(*tasks)
    
    # Create the result dictionary with filtered data
    filtered_results = {}
    for ticker, result in zip(tickers, results):
        if result is not None:
            filtered_results[ticker] = result
    return filtered_results



async def get_latest_options_flow_feed(tickers):
    base_dir = Path("json/options-flow/feed/data.json")

    try:
        async with aiofiles.open(base_dir, mode="rb") as f:
            content = await f.read()
            data = orjson.loads(content)

        # Prepare filtered results per ticker
        filtered_results = defaultdict(list)
        for item in data:
            ticker = item.get("ticker")
            if ticker in tickers:
                # Exclude specific keys
                cleaned_item = {
                    k: v for k, v in item.items()
                    if k not in {'aggresor_ind',"exchange", "tradeCount", "underlying_type","description"}
                }
                filtered_results[ticker].append(cleaned_item)

        # Sort by 'premium' descending and take top 5
        for ticker in filtered_results:
            filtered_results[ticker] = sorted(
                filtered_results[ticker],
                key=lambda x: x.get("cost_basis", 0),
                reverse=True
            )[:5]

        return dict(filtered_results)

    except FileNotFoundError:
        return None
    except (orjson.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing file: {e}")
        return None

async def get_latest_dark_pool_feed(tickers):
    base_dir = Path("json/dark-pool/feed/data.json")

    try:
        async with aiofiles.open(base_dir, mode="rb") as f:
            content = await f.read()
            data = orjson.loads(content)

        # Prepare filtered results per ticker
        filtered_results = defaultdict(list)
        for item in data:
            ticker = item.get("ticker")
            if ticker in tickers:
                # Exclude specific keys
                cleaned_item = {
                    k: v for k, v in item.items()
                    if k not in {"assetType", "sector", "trackingID","ticker"}
                }
                filtered_results[ticker].append(cleaned_item)

        # Sort by 'premium' descending and take top 5
        for ticker in filtered_results:
            filtered_results[ticker] = sorted(
                filtered_results[ticker],
                key=lambda x: x.get("premium", 0),
                reverse=True
            )[:5]

        return dict(filtered_results)

    except FileNotFoundError:
        return None
    except (orjson.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing file: {e}")
        return None

async def get_market_news(tickers):
    base_dir = Path("json/market-news/companies")
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    # Gather all results concurrently
    results = await asyncio.gather(*tasks)
    filtered_results = {}
    for ticker, data in zip(tickers, results):
        if data is not None:
            filtered_data = []
            for item in data:
                # Check if the item was published within the last week
                if (
                    'publishedDate' in item
                    and week_ago <= datetime.strptime(item['publishedDate'], "%Y-%m-%d %H:%M:%S").date()
                ):
                    # Create a filtered copy of the item
                    filtered_item = {k: v for k, v in item.items() if k not in ['image', 'symbol','url','site']}
                    # Add back the ticker as the symbol
                    filtered_data.append(filtered_item)
            filtered_results[ticker] = filtered_data
    return filtered_results


async def get_analyst_ratings(tickers):
    base_dir = Path("json/analyst/history")
    tasks = [fetch_ticker_data(ticker, base_dir) for ticker in tickers]
    # Gather all results concurrently
    results = await asyncio.gather(*tasks)
    filtered_results = {}
    for ticker, data in zip(tickers, results):
        if data is not None:
            filtered_data = []
            for item in data:
                try:
                    filtered_item = {k: v for k, v in item.items() if k not in ['analystId']}
                    filtered_data.append(filtered_item)
                except:
                    pass
            filtered_results[ticker] = filtered_data[:15] #last 15 ratings
    return filtered_results


def get_function_definitions():
    """
    Dynamically construct function definition metadata for OpenAI function-calling.
    Supports income, balance-sheet, cash-flow statements, financial ratios, and hottest options contracts.
    """
    # Define metadata for each statement type
    templates = [
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
            "description": "Retrieves the most recent and historical short interest data for multiple companies using their stock ticker symbols.",
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
            "description": "Fetches forward-looking analyst estimates for multiple stocks, including average, low, and high projections for EPS, revenue, EBITDA, and net income.",
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
            "description": "Fetches upcoming earnings events for stocks within a specified date range (defaulting to the next 10 days if no upper threshold is provided).",
            "parameters": {
                "upper_threshold": {
                    "type": "string",
                    "description": "Optional upper date threshold in YYYY-MM-DD format. If not provided, defaults to 10 days from today."
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
    ]
    
    definitions = []
    for tpl in templates:
        func_def = {
            "name": tpl["name"],
            "description": tpl["description"],
            "parameters": {
                "type": "object",
                "properties": tpl["parameters"],
            }
        }
        # only include "required" if it's explicitly provided
        if "required" in tpl:
            func_def["parameters"]["required"] = tpl["required"]

        definitions.append(func_def)

    return definitions

'''
{
            "name": "get_historical_stock_price",
            "description": "Fetches historical stock price (open, high, low, close) and volume data for a specific stock ticker. Useful for analyzing past stock performance and trends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol for the company (e.g., 'AAPL' for Apple Inc., 'MSFT' for Microsoft Corp.)."
                    }
                },
                "required": ["ticker"]
            },
        },
    {
        "name": "get_stock_screener",
        "description": "Filters and sorts a list of companies based on various financial metrics and criteria.",
    },
'''

#Testing purposes
#data = asyncio.run(get_analyst_ratings(['AMD','TSLA']))
#print(data)