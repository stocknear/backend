import argparse
import sys
from datetime import datetime
import orjson
import os

def save_json(data, symbol, quarter, fiscal_year):
    dir_path = "json/blog"
    os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist
    file_path = os.path.join(dir_path, f"{symbol}-{quarter}-{fiscal_year}.json")
    with open(file_path, 'wb') as file:
        file.write(orjson.dumps(data))


def get_cumulative_returns(symbol):

    with open(f"json/historical-price/max/{symbol}.json", "rb") as file:
        symbol_price = orjson.loads(file.read())[-252:]
    initial_close = symbol_price[0]['close']

    # Load SPY benchmark data
    with open("json/historical-price/max/SPY.json", "rb") as file:
        spy_price = orjson.loads(file.read())[-252:]
    initial_spy_close = spy_price[0]['close']

    # Calculate cumulative returns for both the stock and SPY benchmark
    cumulative_returns = []
    for i in range(len(symbol_price)):
        try:
            date = symbol_price[i]['time']
            
            # Stock cumulative return
            close = symbol_price[i]['close']
            cumulative_roi = round(((close / initial_close) - 1) * 100, 2)
            
            # SPY cumulative return
            spy_close = spy_price[i]['close']
            cumulative_spy = round(((spy_close / initial_spy_close) - 1) * 100, 2)
            
            # Append combined result
            cumulative_returns.append({
                "date": date,
                "cumulativeTicker": cumulative_roi,
                "cumulativeBenchmark": cumulative_spy
            })
        except Exception as e:
            # In case of any error, you could log the exception if needed
            pass

    # Example output
    return cumulative_returns

def get_overview(symbol, screener_data):
    res = {}
    with open(f"json/profile/{symbol}.json","rb") as file:
        data = orjson.loads(file.read())
        res['description'] = data['description']
    
    with open(f"json/quote/{symbol}.json", 'r') as file:
        data = orjson.loads(file.read())
        res['marketCap'] = data['marketCap']
        dt = datetime.strptime(data['earningsAnnouncement'], '%Y-%m-%dT%H:%M:%S.%f%z')
        res['nextEarning'] = dt.strftime("%B %d, %Y")
        res['epsTTM'] = data['eps']
        res['peTTM'] = data['pe']
    
    res['annualDividend'] = screener_data.get('annualDividend',None)
    res['dividendYield'] = screener_data.get('dividendYield',None)
    res['priceToSalesRatio'] = screener_data.get('priceToSalesRatio',None)
    res['priceToBookRatio'] = screener_data.get('priceToBookRatio',None)
    res['sharesOutstanding'] = screener_data.get('sharesOutStanding',None)
    res['shortFloatPercent'] = screener_data.get('shortFloatPercent',None)
    res['shortOutstandingPercent'] = screener_data.get('shortOutstandingPercent',None)
    res['forwardPE'] = screener_data.get('forwardPE',None)
    res['sector'] = screener_data.get('sector',None)

    res['cumulativeReturns'] = get_cumulative_returns(symbol)
    return res

def get_sentiment(value):
    if value is None:
        return "Very Bad"
    elif value < 0:
        return "Very Bad"
    elif value < 5:
        return "Bad"
    elif value < 15:
        return "Average"
    elif value < 25:
        return "Good"
    else:
        return "Very Good"

def get_financial_health(symbol, screener_data):
    fields = [
        ("Gross Profit Margin", "grossProfitMargin"),
        ("Operating Profit Margin", "operatingProfitMargin"),
        ("Net Margin", "netProfitMargin"),
        ("FCF Margin", "freeCashFlowMargin"),
        ("EBITDA Margin", "ebitdaMargin"),
    ]

    res = []
    for label, key in fields:
        value = screener_data.get(key)
        sentiment = get_sentiment(value)
        res.append({
            "label": label,
            "value": value,
            "sentiment": sentiment
        })

    return res


def get_sentiment_growth(value):
    if value is None:
        return "Very Bad"
    elif value < -10:
        return "Very Bad"
    elif value < 0:
        return "Bad"
    elif value < 10:
        return "Average"
    elif value < 30:
        return "Good"
    else:
        return "Very Good"

def get_avg_sentiment(value):
    if value is None:
        return "Very Bad"
    elif value < -30:
        return "Very Good"
    elif value < -15:
        return "Good"
    elif value < 0:
        return "Average"
    elif value < 20:
        return "Bad"
    else:
        return "Very Bad"

def calculate_growth(current, previous):
    if current is None or previous is None or previous == 0:
        return None
    return ((current - previous) / abs(previous)) * 100


def get_growth(symbol):
    # Define the metrics in a way that is easy to extend
    metrics = [
        ("Revenue Growth", "revenue", "income-statement"),
        ("Gross Profit Growth", "grossProfit", "income-statement"),
        ("Operating Income Growth", "operatingIncome", "income-statement"),
        ("Net Income Growth", "netIncome", "income-statement"),
        ("Free Cash Flow Growth", "freeCashFlow", "cash-flow-statement"),
        ("Operating Cash Flow Growth", "freeCashFlow", "cash-flow-statement"),
    ]

    # Cache loaded data by statement type
    data_cache = {}

    summary = []

    for label, key, statement_type in metrics:
        # Load and cache the data for each statement type only once
        if statement_type not in data_cache:
            with open(f"json/financial-statements/{statement_type}/annual/{symbol}.json", "rb") as file:
                data_cache[statement_type] = orjson.loads(file.read())

        current = data_cache[statement_type][0]
        previous = data_cache[statement_type][1]

        growth = calculate_growth(current.get(key), previous.get(key))
        sentiment = get_sentiment_growth(growth)

        summary.append({
            "label": label,
            "value": round(growth, 2) if growth is not None else None,
            "sentiment": sentiment
        })

    return summary


def get_valuation(symbol, screener_data):
    
    keys = ['priceToEarningsRatio', 'priceToFreeCashFlowRatio', 'priceToSalesRatio','priceToBookRatio','priceToEarningsGrowthRatio']

    # Load the ratio data
    with open(f"json/financial-statements/ratios/annual/{symbol}.json", "rb") as file:
        data = orjson.loads(file.read())

    # Ensure we have at least 5 years of data (excluding the most recent one)
    last_5_years = data[1:6]

    result = {}

    for key in keys:
        # Extract last 5 years of the specified key
        historical_values = [entry.get(key) for entry in last_5_years if entry.get(key) is not None]

        if not historical_values:
            result[key] = {
                "error": f"No valid {key} values found"
            }
            continue

        # Calculate the average of the last 5 years
        avg_value = sum(historical_values) / len(historical_values)
        latest_value = screener_data.get(key)

        if latest_value is None:
            result[key] = {
                "error": f"No latest {key} value found"
            }
            continue

        # Calculate upside/downside
        if avg_value == 0:
            upside = None  # or float('inf') or custom handling for division by zero
        else:
            upside = round((latest_value - avg_value) / abs(avg_value) * 100, 2)

        sentiment = get_avg_sentiment(upside)

        result[key] = {
            "fiveYearAvg": round(avg_value,2),
            "value": round(latest_value,2),
            "upside": upside,
            "sentiment": sentiment
        }
    return result
        
def get_industry(symbol, screener_data):
    result = []
    industry = screener_data.get('industry')

    keys = [
        'evToFreeCashFlow', 'evToEBIT', 'evToEBITDA', 'priceToFreeCashFlowRatio',
        'priceToSalesRatio', 'priceToEarningsRatio', 'cagr5YearRevenue',
        'cagr5YearEPS', 'revenuePerShare', 'grossProfitMargin', 'netProfitMargin',
        'operatingProfitMargin', 'revenuePerEmployee', 'altmanZScore',
        'returnOnInvestedCapital', 'returnOnEquity', 'returnOnInvestedCapital'
    ]

    labels = {
        'evToFreeCashFlow': 'EV/FCF',
        'evToEBIT': 'EV/EBIT',
        'evToEBITDA': 'EV/EBITDA',
        'priceToFreeCashFlowRatio': 'P/FCF',
        'priceToSalesRatio': 'P/S',
        'priceToEarningsRatio': 'P/E',
        'cagr5YearRevenue': '5Y Revenue CAGR',
        'cagr5YearEPS': '5Y EPS CAGR',
        'revenuePerShare': 'Revenue/Share',
        'grossProfitMargin': 'Gross Margin',
        'netProfitMargin': 'Net Margin',
        'operatingProfitMargin': 'Operating Margin',
        'revenuePerEmployee': 'Revenue/Employee',
        'altmanZScore': 'Altman Z-Score',
    }

    with open(f"json/average/industry/data.json") as file:
        industry_avg = orjson.loads(file.read())[industry]


    for key in keys:
        avg_value = industry_avg.get(key)
        latest_value = screener_data.get(key)

        if avg_value is None or latest_value is None:
            continue  # Skip if data is missing

        label = labels.get(key, key)  # Fallback to key if label is missing

        upside = round((latest_value - avg_value) / abs(avg_value) * 100, 2) if avg_value != 0 else None

        result.append({
            "label": label,
            "key": key,
            "industryAvg": round(avg_value, 2),
            "value": round(latest_value, 2),
            "upside": upside,
        })

    return result


def get_management(symbol, screener_data):
    result = []

    # Extract necessary values from screener_data
    industry = screener_data.get('industry')
    sbc = screener_data.get('stockBasedCompensation')
    revenue = screener_data.get('revenue')
    operating_cash_flow = screener_data.get('operatingCashFlow')
    free_cash_flow = screener_data.get('freeCashFlow')

    # Safely compute SBC ratios (avoiding division by zero)
    def safe_ratio(numerator, denominator):
        return (numerator/denominator)*100 if numerator is not None and denominator else None

    sbc_to_revenue = safe_ratio(sbc, revenue)
    sbc_to_ocf = safe_ratio(sbc, operating_cash_flow)
    sbc_to_fcf = safe_ratio(sbc, free_cash_flow)

    # Add the computed ratios into screener_data for unified processing
    screener_data = screener_data.copy()  # avoid mutating original
    screener_data.update({
        "sbcToRevenueRatio": sbc_to_revenue,
        "sbcToOperatingCashFlowRatio": sbc_to_ocf,
        "sbcToFreeCashFlowRatio": sbc_to_fcf
    })

    # Define the keys and corresponding labels
    keys = [
        'sbcToRevenueRatio',
        'sbcToOperatingCashFlowRatio',
        'sbcToFreeCashFlowRatio',
        "returnOnEquity",
        "returnOnAssets",
        "returnOnInvestedCapital",
        "returnOnCapitalEmployed",
    ]

    labels = {
        'sbcToRevenueRatio': "SBC as % of Revenue",
        'sbcToOperatingCashFlowRatio': "SBC as % of Operating Cash Flow",
        'sbcToFreeCashFlowRatio': 'SBC as % of Free Cash Flow',
        "returnOnEquity": "Return on Equity",
        "returnOnAssets": "Return on Assets",
        "returnOnInvestedCapital": "Return on Invested Capital",
        "returnOnCapitalEmployed": "Return on Capital Employed",
    }

    for key in keys:
        latest_value = screener_data.get(key)
        print(latest_value)
        label = labels.get(key, key)

        # You may define `previous` elsewhere, otherwise set growth to 0
        sentiment = get_sentiment_growth(latest_value)
        result.append({
            "label": label,
            "key": key,
            "value": round(latest_value, 2),
            "sentiment": sentiment,
        })
    print(result)
    return result



def main():
    symbol = "NVDA"
    res = {}
    try:
        with open(f"json/earnings/next/{symbol}.json","rb") as file:
            earnings_data = orjson.loads(file.read())
        with open(f"json/earnings/raw/{symbol}.json","rb") as file:
            data = orjson.loads(file.read())[0] #next quarter and fiscal year
            quarter = data.get('period')
            fiscal_year = data.get('period_year')
    except:
        print("No earnings data found.")
        return

    if earnings_data and quarter and fiscal_year:
        res['nextEarningsData'] = earnings_data

        with open("json/stock-screener/data.json", "rb") as file:
            stock_screener_data = orjson.loads(file.read())

        # Convert list to a dict keyed by symbol
        stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}
        screener_data = stock_screener_data_dict.get(symbol)

        res['overview'] = get_overview(symbol, screener_data)
        res['name'] = screener_data.get('name',None)
        res['symbol'] = screener_data.get('symbol',None)
        
        res['financialHealth'] = get_financial_health(symbol, screener_data)
        res['growth'] = get_growth(symbol)
        res['valuation'] = get_valuation(symbol, screener_data)
        res['industry'] = get_industry(symbol, screener_data)
        res['management'] = get_management(symbol, screener_data)


        save_json(res, symbol, quarter, fiscal_year)
    else:
        print("No earnings data found.")
        return

    

    '''
    # Load the business metrics for the symbol
    metrics_path = f"json/business-metrics/{symbol}.json"
    try:
        with open(metrics_path, "rb") as file:
            business_metrics = orjson.loads(file.read())
    except FileNotFoundError:
        print(f"Error: Business metrics file for '{symbol}' not found.", file=sys.stderr)
        sys.exit(1)

    # Display results
    print("Business Metrics Data:")
    print(business_metrics)
    print("\n\n")
    print("Fundamental & Key Data:")
    print(symbol_data)
    '''

if __name__ == "__main__":
    main()
