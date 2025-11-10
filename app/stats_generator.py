import argparse
import sys
from datetime import datetime
import orjson

# -----------------------------------------------------------------------------
# Script to load and display stock screener data and business metrics for a given
# ticker symbol passed via command-line argument.
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load stock screener and business metrics data for a ticker symbol"
    )
    parser.add_argument(
        '--ticker', '-t',
        required=True,
        help="Stock ticker symbol (e.g., AMD)"
    )
    args = parser.parse_args()

    # Normalize the symbol to uppercase
    symbol = args.ticker.upper()

    # Load the stock screener data
    try:
        with open("json/stock-screener/data.json", "rb") as file:
            stock_screener_data = orjson.loads(file.read())
    except FileNotFoundError:
        print("Error: stock-screener data file not found.", file=sys.stderr)
        sys.exit(1)

    # Convert list to a dict keyed by symbol
    stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}
    symbol_data = stock_screener_data_dict.get(symbol)

    if symbol_data is None:
        print(f"Error: Symbol '{symbol}' not found in stock-screener data.", file=sys.stderr)
        sys.exit(1)

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

if __name__ == "__main__":
    main()
