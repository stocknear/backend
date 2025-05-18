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

    base_dir = Path("../json/financial-statements") / dir_name / time_period
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


async def get_ratios_statement(ticker,time_period = "annual",keep_keys = None):
    file_path = Path(f"../json/financial-statements/ratios/{time_period}/{ticker}.json")
    
    # Load the raw data
    with file_path.open("rb") as f:
        raw_data = orjson.loads(f.read())
    
    # Keys we want to strip out by default
    remove_keys = [
        "symbol", "reportedCurrency",
        "acceptedDate", "cik", "filingDate"
    ]

    
    if keep_keys:
        # Make a local copy and ensure required fields are present
        keys_to_keep = list(keep_keys)
        if "date" not in keys_to_keep:
            keys_to_keep.append("date")
        if "fiscalYear" not in keys_to_keep:
            keys_to_keep.append("fiscalYear")
        if "period" not in keys_to_keep:
            keys_to_keep.append("period")
        
        # Keep only those keys
        filtered = [
            {k: v for k, v in stmt.items() if k in keys_to_keep}
            for stmt in raw_data
        ]
    else:
        # Remove the unwanted keys
        filtered = [
            {k: v for k, v in stmt.items() if k not in remove_keys}
            for stmt in raw_data
        ]
    return filtered


async def get_hottest_contracts(ticker,time_period = "annual",keep_keys = None):
    file_path = Path(f"../json/financial-statements/ratios/{time_period}/{ticker}.json")
    
    # Load the raw data
    with file_path.open("rb") as f:
        raw_data = orjson.loads(f.read())
    
    # Keys we want to strip out by default
    remove_keys = [
        "symbol", "reportedCurrency",
        "acceptedDate", "cik", "filingDate"
    ]

    
    if keep_keys:
        # Make a local copy and ensure required fields are present
        keys_to_keep = list(keep_keys)
        if "date" not in keys_to_keep:
            keys_to_keep.append("date")
        if "fiscalYear" not in keys_to_keep:
            keys_to_keep.append("fiscalYear")
        if "period" not in keys_to_keep:
            keys_to_keep.append("period")
        
        # Keep only those keys
        filtered = [
            {k: v for k, v in stmt.items() if k in keys_to_keep}
            for stmt in raw_data
        ]
    else:
        # Remove the unwanted keys
        filtered = [
            {k: v for k, v in stmt.items() if k not in remove_keys}
            for stmt in raw_data
        ]
    return filtered



async def get_stock_screener():
    pass




def get_function_definitions():
    """
    Dynamically construct function definition metadata for OpenAI function-calling.
    Supports income, balance-sheet, cash-flow statements, and financial ratios.
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
            "periods": ["annual", "quarter", "ttm"],
        },
        {
            "name": "get_balance_sheet_statement",
            "description": (
                "Fetches historical balance sheet statements for stock tickers. "
                f"Includes metrics: {', '.join(key_balance_sheet)}. "
                "Available for annual, quarter, and trailing twelve months (ttm)."
            ),
            "periods": ["annual", "quarter", "ttm"],
        },
        {
            "name": "get_cash_flow_statement",
            "description": (
                "Obtains historical cash flow statements for stock tickers. "
                f"Key items: {', '.join(key_cash_flow)}. "
                "Available for annual, quarter, and trailing twelve months (ttm)."
            ),
            "periods": ["annual", "quarter", "ttm"],
        },
        {
            "name": "get_ratios_statement",
            "description": (
                "Retrieves various historical financial ratios for stock tickers. "
                f"Examples: {', '.join(key_ratios)}. "
                "Available for annual and quarter periods."
            ),
            "periods": ["annual", "quarter"],
        },
    ]

    definitions = []
    for tpl in templates:
        definitions.append({
            "name": tpl["name"],
            "description": tpl["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of stock ticker symbols (e.g., [\"AAPL\", \"GOOGL\"])."
                        ),
                    },
                    "time_period": {
                        "type": "string",
                        "enum": tpl["periods"],
                        "description": (
                            "Time period for the data: " + ", ".join(tpl["periods"]) + "."
                        ),
                    },
                    "keep_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of data keys to retain in the output. If omitted, defaults will be used."
                        ),
                    },
                },
                "required": ["tickers", "time_period"],
            }
        })

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
#data = asyncio.run(get_cash_flow_statement(['AAPL','NVDA'], 'annual',['freeCashFlow']))
#print(data)