import os
import pandas as pd
import ujson
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
import sqlite3
from datetime import datetime

def load_json(file_path):
    """Load existing JSON data from file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return ujson.load(file)
        except (ValueError, IOError):
            print(f"Warning: Could not read or parse {file_path}. Starting with an empty list.")
    return []

def save_latest_ratings(combined_data, json_file_path, limit=700):
    """
    Saves the latest `limit` ratings to the JSON file, ensuring no duplicates.
    
    Args:
        combined_data (list): List of dictionaries containing stock data.
        json_file_path (str): Path to the JSON file.
        limit (int): The maximum number of entries to save (default is 500).
    """
    try:
        # Create a set to track unique entries based on a combination of 'ticker' and 'date'
        seen = set()
        unique_data = []

        for item in combined_data:
            # Create a unique identifier (e.g., 'ticker|date')
            identifier = f"{item['ticker']}|{item['date']}"
            if identifier not in seen:
                seen.add(identifier)
                unique_data.append(item)

        # Sort the data by date (assumes date is in 'YYYY-MM-DD' format)
        sorted_data = sorted(unique_data, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=True)

        # Keep only the latest `limit` entries
        latest_data = sorted_data[:limit]

        # Save the trimmed and deduplicated data to the JSON file
        with open(json_file_path, 'w') as file:
            ujson.dump(latest_data, file)

        print(f"Saved {len(latest_data)} unique and latest ratings to {json_file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

query_template = """
    SELECT 
        name
    FROM 
        stocks 
    WHERE
        symbol = ?
"""

SENTIMENT_MAP = {
    "Bullish": "Strong Buy",
    "Buy": "Buy",
    "Buy on a Pullback": "Buy",
    "Speculative - Good": "Buy",
    "Trim": "Sell",
    "Bearish": "Sell",
    "Sell": "Strong Sell",
    "Sell on a Pop": "Strong Sell",
    "Hold": "Hold",
    "Not Recommending": "Hold",
    "Start a Small Position": "Hold",
    "Long": "Hold",
    "Final Trade": "Hold",
    "Speculative": "Hold"
}


def replace_sentiments_in_data(combined_data):
    """
    Replaces sentiments in the given data based on the sentiment mapping.
    
    Args:
        combined_data (list): List of dictionaries containing stock data.

    Returns:
        list: Updated data with replaced sentiments.
    """
    for item in combined_data:
        # Get the original sentiment and map it to the new value
        original_sentiment = item.get('sentiment', 'Hold')
        item['sentiment'] = SENTIMENT_MAP.get(original_sentiment, "Hold")
    
    return combined_data

def format_date(date_str):
    """Convert date from 'Nov. 21, 2024' to '2024-11-21'."""
    try:
        return datetime.strptime(date_str, '%b. %d, %Y').strftime('%Y-%m-%d')
    except:
        return date_str

def main():
    # Load environment variables
    con = sqlite3.connect('stocks.db')
    load_dotenv()
    url = os.getenv('CRAMER_WEBSITE')

    # Set up the WebDriver options
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Initialize the WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(options=options)

    json_file_path = 'json/cramer-tracker/data.json'

    try:
        # Fetch the website
        driver.get(url)
        # Find the table element
        table = driver.find_element(By.TAG_NAME, 'table')
        # Extract the table HTML
        table_html = table.get_attribute('outerHTML')
        # Use pandas to read the HTML table
        df = pd.read_html(table_html)[0]
        # Rename the columns
        df = df.rename(columns={
            'Ticker': 'ticker',
            'Direction': 'sentiment',
            'Date': 'date',
            'Return Since': 'returnSince'
        })
        # Convert the DataFrame to JSON
        data = ujson.loads(df.to_json(orient='records'))

        # Load existing data
        existing_data = load_json(json_file_path)

        # Transform existing data into a set of unique identifiers
        existing_keys = {(item['ticker'], item['date']) for item in existing_data}

        # Prepare results with only new data
        res = []
        for item in data:
            symbol = item['ticker']
            try:
                # Convert 'Return Since' to float and round it
                item['returnSince'] = round(float(item['returnSince'].replace('%', '')), 2)

                if not item['date']:
                    continue  # Skip if date parsing fails
                
                # Check if the data is already in the file
                if (item['ticker'], item['date']) not in existing_keys:
                    db_data = pd.read_sql_query(query_template, con, params=(symbol,))
                    res.append({
                        **item,
                        'name': db_data['name'].iloc[0]
                    })
            except Exception as e:
                print(f"Error processing {symbol}: {e}")

        # Append new data to existing data and save
        combined_data = existing_data + res
        updated_data = replace_sentiments_in_data(combined_data)
        for item in combined_data:
            item['date'] = format_date(item['date'])
        save_latest_ratings(combined_data, json_file_path)
        
    
    finally:
        # Ensure the WebDriver is closed
        driver.quit()
        con.close()

if __name__ == '__main__':
    main()
