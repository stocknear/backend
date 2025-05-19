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
from operator import itemgetter


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



async def get_stock_screener(rule_of_list=None, sort_by=None, sort_order="desc", limit=10):
    try:
        # Use aiofiles for non-blocking file operations
        async with aiofiles.open(os.path.join("json", "stock-screener", "data.json"), 'rb') as file:
            content = await file.read()
            data = orjson.loads(content)

        # Initial filter to exclude PNK exchange - use list comprehension for efficiency
        filtered_data = [item for item in data if item.get('exchange') != 'PNK']

        # Exit early if no rules provided
        if not rule_of_list:
            result = filtered_data
        else:
            # Apply each filter rule
            result = []

            # Define operator mapping as a dict for cleaner code
            operators = {
                '>': lambda x, y: x > y,
                '>=': lambda x, y: x >= y,
                '<': lambda x, y: x < y,
                '<=': lambda x, y: x <= y,
                '==': lambda x, y: x == y,
                '!=': lambda x, y: x != y
            }

            # Process each stock
            for stock in filtered_data:
                meets_criteria = True

                # Check each rule
                for rule in rule_of_list:
                    # Use new naming convention from schema
                    metric = rule.get('metric', rule.get('name'))  # Support both new and old format
                    value = rule.get('value')
                    operator = rule.get('operator', '>')

                    # Skip invalid rules
                    if not metric or metric not in stock or operator not in operators:
                        meets_criteria = False
                        break

                    stock_value = stock[metric]

                    # Handle None/null values in data
                    if stock_value is None:
                        meets_criteria = False
                        break

                    # Handle type mismatches gracefully
                    try:
                        # Apply comparison using operator mapping
                        if not operators[operator](stock_value, value):
                            meets_criteria = False
                            break
                    except (TypeError, ValueError):
                        # Type mismatch in comparison
                        meets_criteria = False
                        break

                if meets_criteria:
                    result.append(stock)

        # Sort results if requested
        if sort_by and result and sort_by in result[0]:
            # Handle None values in sorting
            result.sort(
                key=lambda x: (x.get(sort_by) is None, x.get(sort_by)),
                reverse=(sort_order.lower() == "desc")
            )

        # Apply limit
        if limit and isinstance(limit, int):
            result = result[:limit]

        # Extract only the key elements defined in rule_of_list for each matched stock
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

                    # Add sort_by field if it's not already included but was used for sorting
                    if sort_by and sort_by not in metrics and sort_by in stock:
                        metrics[sort_by] = stock[sort_by]

                    if metrics: # Only add 'metrics' key if there are metrics to add
                        filtered_stock["metrics"] = metrics

                filtered_result.append(filtered_stock)
            except Exception:
                # Log the exception if needed for debugging
                pass

        # Return in the format specified in the schema
        return {
            "matched_stocks": filtered_result,
            "count": len(filtered_result)
        }

    except FileNotFoundError:
        return {"matched_stocks": [], "count": 0, "error": "Screener data file not found"}
    except (orjson.JSONDecodeError, Exception) as e:
        return {"matched_stocks": [], "count": 0, "error": f"Error processing screener data: {str(e)}"}


def get_function_definitions():
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
'''
data = asyncio.run(get_stock_screener(
    rule_of_list=[
        {"metric": "marketCap", "operator": ">", "value": 100},
        {"metric": "priceToEarningsRatio", "operator": "<", "value": 10}
    ],
    sort_by="marketCap",
    sort_order="desc",
    limit=10
))
print(data)
'''
