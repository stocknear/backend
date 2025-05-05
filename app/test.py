from datetime import datetime
import orjson

# Load the JSON file
with open("json/stock-screener/data.json", "rb") as file:
    stock_screener_data = orjson.loads(file.read())

# Convert list to a dictionary with symbols as keys
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

# Choose a symbol
symbol = 'WOLF'

# Save the data for the symbol in a new dictionary
symbol_data = stock_screener_data_dict.get(symbol, {})

# Optional: print to verify
print(symbol_data)
