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
