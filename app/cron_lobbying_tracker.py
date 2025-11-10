import os
import pandas as pd
import ujson
import orjson
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
import sqlite3

load_dotenv()
url = os.getenv('CORPORATE_LOBBYING')

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        ujson.dump(data, file)


def main():
    # Load environment variables
    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]



    # Set up the WebDriver options
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Initialize the WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(options=options)

    try:
        # Fetch the website
        driver.get(url)

        # Find the table element
        table = driver.find_element(By.ID, 'myTable')

        # Extract data from the table
        data = []
        rows = table.find_elements(By.TAG_NAME, 'tr')[1:]  # Skip the header row
        for row in rows:
            columns = row.find_elements(By.TAG_NAME, 'td')
            if len(columns) == 3:
                ticker = columns[0].find_element(By.TAG_NAME, 'strong').text
                company = columns[0].find_element(By.TAG_NAME, 'span').text
                amount = columns[1].text.strip()
                date = columns[2].text.strip()
                amount_int = int(amount.replace('$', '').replace(',', ''))
                
                data.append({
                    'ticker': ticker,
                    'company': company,
                    'amount': amount_int,
                    'date': date
                })

        # Fetch additional data from the database
        res = []
        for item in data:
            item['ticker'] = item['ticker'].replace('BRK.A','BRK-A').replace("BRK.B","BRK-B")
            symbol = item['ticker']
            if symbol in stock_symbols:
                try:
                    with open(f"json/quote/{symbol}.json") as file:
                        quote_data = orjson.loads(file.read())

                        item['date'] = item['date'].replace('p.m.', 'PM').replace('a.m.', 'AM')
                        res.append({
                            **item,
                            'name': quote_data['name'], 
                            'price': round(quote_data['price'],2),
                            'changesPercentage': round(quote_data['changesPercentage'],2)
                        })
                    
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")

        # Save the JSON data
        if len(res) > 0:
            save_json(res, 'json/corporate-lobbying/tracker/data.json')

    finally:
        # Ensure the WebDriver is closed
        driver.quit()
        con.close()

if __name__ == '__main__':
    main()