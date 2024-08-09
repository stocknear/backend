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

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        ujson.dump(data, file)

query_template = """
    SELECT 
        name, sector
    FROM 
        stocks 
    WHERE
        symbol = ?
"""

def main():
    # Load environment variables
    con = sqlite3.connect('stocks.db')
    load_dotenv()
    url = os.getenv('CORPORATE_LOBBYING')

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
            symbol = item['ticker']
            try:
                db_data = pd.read_sql_query(query_template, con, params=(symbol,))
                if not db_data.empty:
                    item['date'] = item['date'].replace('p.m.', 'PM')
                    item['date'] = item['date'].replace('a.m.', 'AM')
                    res.append({
                        **item, 
                        'name': db_data['name'].iloc[0], 
                        'sector': db_data['sector'].iloc[0]
                    })
                else:
                    res.append(item)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                res.append(item)

        # Save the JSON data
        if len(res) > 0:
            save_json(res, 'json/corporate-lobbying/tracker/data.json')

    finally:
        # Ensure the WebDriver is closed
        driver.quit()
        con.close()

if __name__ == '__main__':
    main()