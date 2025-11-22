import fear_and_greed
import requests
from datetime import datetime, date
import json
import orjson
import os

current = fear_and_greed.get()

print(f"Current value: {current.value}")
print(f"Category: {current.description}")
print(f"Last updated: {current.last_update}")

BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
start_date = f"{date.today().year - 3}-01-01"


def save_json(data, path="json/fear-and-greed/data.json"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save JSON as bytes
    with open(path, "wb") as file:
        file.write(orjson.dumps(data))

# Add headers to avoid bot detection
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.cnn.com/markets/fear-and-greed',
    'Origin': 'https://www.cnn.com'
}

try:
    response = requests.get(BASE_URL + start_date, headers=headers)
    response.raise_for_status()
    data = response.json()
    # Extract historical data
    history = data.get("fear_and_greed_historical", {}).get("data", [])
    
    # Helper function to categorize values
    def get_category(value):
        if value <= 25:
            return "Extreme Fear"
        elif value <= 45:
            return "Fear"
        elif value <= 55:
            return "Neutral"
        elif value <= 75:
            return "Greed"
        else:
            return "Extreme Greed"    

    data = {
        "current": {
            "value": round(current.value,0),
            "category": current.description,
            "date": current.last_update.isoformat() if hasattr(current.last_update, 'isoformat') else str(current.last_update)
        },
    }

    if data:
        print(data)
        save_json(data)
        
except Exception as e:
    print(f"Error fetching historical data: {e}")
