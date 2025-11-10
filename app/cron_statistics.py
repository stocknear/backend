from datetime import datetime
import orjson
import sqlite3
import asyncio
from tqdm import tqdm


# Load stock screener data
with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}


async def save_json(symbol, data):
    """Save JSON data to a file."""
    with open(f"json/statistics/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))


async def get_data(symbol):
    """Extract specified columns data for a given symbol."""
    columns = ['sharesOutStanding', 'sharesQoQ', 'sharesYoY','institutionalOwnership','floatShares',
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
    'lastStockSplit','splitType','splitRatio','analystRating','analystCounter','priceTarget','upside',
    'grahamNumber','grahamUpside','lynchFairValue','lynchUpside',
    ]
    
    if symbol in stock_screener_data_dict:
        result = {}
        for column in columns:
            try:
                result[column] = stock_screener_data_dict[symbol].get(column, None)
            except:
                pass
        return result
    return {}


async def run():
    """Main function to run the data extraction process."""
    # Connect to SQLite database
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    total_symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    
    # Process symbols with progress bar
    for symbol in tqdm(total_symbols, desc="Extracting data"):
        data = await get_data(symbol)
        if data:  # Only save if we have data
            await save_json(symbol, data)
    
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())