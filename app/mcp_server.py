import orjson
import aiofiles
import asyncio
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar, Set, Tuple, cast


mcp = FastMCP("Stocknear MCP Server")


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


async def load_json_file(filepath: str) -> Union[Dict, List, str]:
    try:
        async with aiofiles.open(filepath, 'rb') as file:
            content = await file.read()
            return orjson.loads(content)
    except FileNotFoundError:
        return {"error": f"File not found: {filepath}"}
    except orjson.JSONDecodeError:
        return {"error": f"Failed to decode JSON in file: {filepath}"}
    except Exception as e:
        return {"error": f"Unhandled error: {str(e)}"}


@mcp.tool()
async def get_potus_tracker(ticker: str) -> Union[Dict, str]:
    return await load_json_file("json/tracker/potus/data.json")

@mcp.tool()
async def get_top_gainers():
    try:
        with open(f"json/market-movers/markethours/gainers.json", 'rb') as file:
            data = orjson.loads(file.read())['1D'][:10]
            return data
    except Exception as e:
        return f"Error processing top gainers data: {str(e)}"

@mcp.tool()
async def get_top_losers() -> Union[List[Dict], Dict[str, str]]:
    result = await load_json_file("json/market-movers/markethours/losers.json")
    if isinstance(result, dict) and "1D" in result:
        return result["1D"][:10]
    return result if isinstance(result, dict) else {"error": "Invalid JSON structure"}


@mcp.tool()
async def get_top_active_stocks() -> Union[List[Dict], Dict[str, str]]:
    result = await load_json_file("json/market-movers/markethours/active.json")
    if isinstance(result, dict) and "1D" in result:
        return result["1D"][:10]
    return result if isinstance(result, dict) else {"error": "Invalid JSON structure"}

@mcp.tool()
async def get_stock_screener(
    rule_of_list: Optional[List[Dict[str, Any]]] = None, 
    sort_by: Optional[str] = None, 
    sort_order: str = "desc", 
    limit: int = 10
) -> Dict[str, Any]:
    f"""
    Retrieves stock data based on user-defined financial screening criteria.

    This endpoint allows users to filter and sort stocks by various financial metrics, 
    such as market capitalization, P/E ratio, revenue, and more. The filtering is 
    performed using a list of rules where each rule defines a metric, a comparison 
    operator, and a value. The results can also be sorted and limited in number.

    Rules must be defined using available metrics listed in: {', '.join(key_screener)}.

    Parameters:
    - rule_of_list (List[Dict[str, Union[str, int, float]]]): 
        A list of filtering rules to apply to the screener. Each rule must specify:
        - metric (str): The financial metric to filter by 
          (e.g., "marketCap", "priceToEarningsRatio", "revenue").
        - operator (str): The comparison operator to use ("<", "<=", ">", ">=", "==", "!=").
        - value (Union[str, float, int]): The value to compare against. 
          Can be a number or a string depending on the metric.

        Example:
        [
            {"metric": "marketCap", "operator": ">", "value": 1000000000},
            {"metric": "priceToEarningsRatio", "operator": "<", "value": 15}
        ]

    - sort_by (str, optional): 
        The field name to sort the results by (e.g., "marketCap", "volume", "price").

    - sort_order (str, optional): 
        Sort order for the results: 
        - "asc" for ascending 
        - "desc" for descending (default).

    - limit (int, optional): 
        The maximum number of results to return (default is 10).

    Returns:
    - A filtered and sorted list of stocks that match the specified criteria.
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


if __name__ == "__main__":
    mcp.run(transport="sse")
