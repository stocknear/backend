from datetime import datetime
import orjson

# Load the JSON file
with open("json/stock-screener/data.json", "rb") as file:
    stock_screener_data = orjson.loads(file.read())

# Convert list to a dictionary with symbols as keys
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

# Choose a symbol
symbol = 'WMT'

# Save the data for the symbol in a new dictionary
symbol_data = stock_screener_data_dict.get(symbol, {})


with open(f"json/business-metrics/{symbol}.json", "rb") as file:
    businss_metrics = orjson.loads(file.read())

print("Business Metrics Data:")
print(businss_metrics)
print("\n\n")
print("Fundamental & Key Data:")
print(symbol_data)
