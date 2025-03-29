import os
import requests
from PIL import Image
from io import BytesIO

# List of stock symbols
total_symbols = ["AAPL", "GOOGL", "MSFT"]  # Add more symbols as needed

# Create output directory if it doesn't exist
output_dir = "json/logos/"
os.makedirs(output_dir, exist_ok=True)

for symbol in total_symbols:
    url = f"https://financialmodelingprep.com/image-stock/{symbol}.png"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for failed requests

        # Convert to WebP
        image = Image.open(BytesIO(response.content))
        output_path = os.path.join(output_dir, f"{symbol}.webp")
        image.save(output_path, "WEBP")

        print(f"Successfully converted {symbol} to WebP.")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {symbol}: {e}")

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
