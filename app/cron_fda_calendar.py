import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sqlite3
import ujson
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv('FDA_CALENDAR')

def save_json(data):
    with open("json/fda-calendar/data.json", 'w', encoding='utf-8') as file:
        ujson.dump(data, file, ensure_ascii=False, indent=2)

def main():
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Initialize WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Connect to the database to get stock symbols
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0].strip() for row in cursor.fetchall()]  # Ensure symbols are stripped
    con.close()

    try:
        # Navigate to FDA calendar
        driver.get(url)
        
        # Wait for the table to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.flow-full-table"))
        )
        
        # Extract table data
        entries = []
        rows = driver.find_elements(By.CSS_SELECTOR, "table.flow-full-table tbody tr")
        
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >=6:  # Check for minimum required columns
                try:
                    # Extract ticker from the anchor tag, stripping whitespace
                    ticker_element = cols[0].find_element(By.TAG_NAME, "a")
                    ticker = ticker_element.text.strip() if ticker_element else ""
                    ticker = ticker or None  # Set to None if empty after strip
                except:
                    ticker = None  # If no anchor tag found
                

                # Extract other fields, converting empty strings to None
                date = cols[1].text.strip() or None
                drug = cols[2].text.strip() or None
                indication = cols[3].text.strip() or None
                status = cols[4].text.strip() or None
                description = cols[5].text.strip() or None
                
                entry = {
                    "ticker": ticker,
                    "date": date,
                    "drug": drug,
                    "indication": indication,
                    "status": status,
                    "description": description
                }
                entries.append(entry)
        
        # Filter entries to include only those with tickers present in the database
        filtered_entries = [
            entry for entry in entries
            if entry['ticker'] is not None and entry['ticker'] in stock_symbols
        ]
        

        if filtered_entries:
            save_json(filtered_entries)
            print("Successfully scraped FDA calendar data")
        
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()