from google import genai
from google.genai import types
import re
import os
from dotenv import load_dotenv
from typing import Dict, Any
import orjson
from pathlib import Path

# Load environment variables
load_dotenv()

# Define BASE_DIR (assuming you have this defined elsewhere)
BASE_DIR = Path("json")  # Adjust this to your actual base directory

def get_reddit_tracker() -> Dict[str, Any]:
    """
    Retrieves WallStreetBets trending tickers for 1 week, 1 month, and 3 months.  
    Each period contains a list of trending tickers with their discussion metrics.  
    Example item:
        {
            "symbol": "RBLX",                  # Stock ticker
            "count": 6,                        # Number of mentions
            "sentiment": "Bearish",            # Overall sentiment (Bullish/Bearish/Neutral)
            "weightPercentage": 3.45,          # Relative weight in discussions (%)
            "name": "Roblox Corporation",      # Company name
            "price": 123.99,                   # Last traded price
            "changesPercentage": -0.66,        # Price change percentage
            "marketCap": 85955687223,          # Market capitalization
            "assetType": "stocks",             # Asset type
            "rank": 8                          # Rank among trending tickers
        }
    
    Returns:
        Dict[str, Any]: A dictionary with three keys:
            {
                "oneWeek": [ ... top 10 tickers ... ],
                "oneMonth": [ ... top 10 tickers ... ],
                "threeMonths": [ ... top 10 tickers ... ]
            }
        If the file is missing or malformed, a FileNotFoundError or JSON parsing
        error may be raised.
    """
    base_dir = BASE_DIR / "reddit-tracker/wallstreetbets/trending.json"
    with open(base_dir, "rb") as file:
        data = orjson.loads(file.read())
        # Remove oneDay key if present
        data.pop("oneDay", None)
        data['oneWeek'] = data.get('oneWeek', [])[:10]
        data['oneMonth'] = data.get('oneMonth', [])[:10]
        data['threeMonths'] = data.get('threeMonths', [])[:10]
    return data

# Configure the client
print(get_reddit_tracker())
gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Create config with the function directly passed to tools
config = types.GenerateContentConfig(
    tools=[get_reddit_tracker]  # Pass the function directly
)

# Make the request - SDK will handle function calls automatically
response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What are the latest reddit updates and trending stocks on WallStreetBets?",
    config=config,
)

# The SDK handles the function call and returns the final text
print(response.text)