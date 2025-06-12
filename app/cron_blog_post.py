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

    print(summary)
    return summary




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
