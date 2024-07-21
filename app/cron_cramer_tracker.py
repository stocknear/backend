import os
import pandas as pd
import ujson
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
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
    url = os.getenv('CRAMER_WEBSITE')

    # Set up the WebDriver options
    options = Options()
    options.headless = True  # Run in headless mode

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

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

        res = []
        for item in data:
            symbol = item['ticker']
            try:
                item['returnSince'] = round(float(item['returnSince'].replace('%','')),2)
                db_data = pd.read_sql_query(query_template, con, params=(symbol,))
                res.append({**item, 'name': db_data['name'].iloc[0], 'sector': db_data['sector'].iloc[0]})
            except Exception as e:
                pass

        # Save the JSON data
        save_json(res, 'json/cramer-tracker/data.json')
    
    finally:
        # Ensure the WebDriver is closed
        driver.quit()
        con.close()

if __name__ == '__main__':
    main()
