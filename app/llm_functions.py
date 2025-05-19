import os
import json
import requests
import aiohttp
import asyncio
from dotenv import load_dotenv
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


async def get_stock_screener():
    pass




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
    ]
    
    definitions = []
    for tpl in templates:
        function_def = {
            "name": tpl["name"],
            "description": tpl["description"],
            "parameters": {
                "type": "object",
                "properties": tpl["parameters"],
                "required": tpl["required"]
            }
        }
        definitions.append(function_def)
    
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
#data = asyncio.run(get_hottest_options_contracts(['GME'], 'volume'))
#print(data)