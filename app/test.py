import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('STOCKNEAR_API_KEY')

# Define the URL and the API key
origin = "https://stocknear.com"
url = f"{origin}/api/sendPushSubscription"

# Define the data payload for the notification
data = {
    "title": "Hast du nix besseres zu tun am Sonntag",
    "body": "Du Opfer",
    "url": f"{origin}/stocks/nvda",
    "key": api_key,
}

# Set headers
headers = {
    "Content-Type": "application/json"
}

# Make the POST request with the API key in the payload
response = requests.post(url, headers=headers, data=json.dumps(data))

# Print the response from the server
print(response.status_code)
print(response.json())
