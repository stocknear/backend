import os
import pandas as pd
import ujson
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        ujson.dump(data, file)


def main():
    # Load environment variables
    load_dotenv()
    url = os.getenv('DIVIDEND_KINGS')

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
        table = driver.find_element(By.TAG_NAME, 'table')
        # Extract the table HTML
        table_html = table.get_attribute('outerHTML')
        # Use pandas to read the HTML table
        df = pd.read_html(table_html)[0]
        # Rename the columns
        df = df.rename(columns={
            'Symbol': 'symbol',
            'Company Name': 'name',
            'Stock Price': 'price',
            '% Change': 'changesPercentage',
            'Div. Yield': 'dividendYield',
            'Years': 'years'
        })
        df = df.drop(columns=['No.'])
        # Convert the DataFrame to JSON
        data = ujson.loads(df.to_json(orient='records'))
        res = []
        for item in data:
            symbol = item['symbol']
            try:
                with open(f"json/quote/{symbol}.json") as file:
                    quote_data = ujson.load(file)

                    item['changesPercentage'] = round(quote_data['changesPercentage'],2)
                    item['price'] = round(quote_data['price'],2)
                    item['dividendYield'] = round(float(item['dividendYield'].replace('%','')),2)
                    res.append({**item})
            except Exception as e:
                print(e)
                pass

        # Save the JSON data
        if len(res) > 0:
            res = sorted(res, key=lambda x: x['years'], reverse=True)
            for rank, item in enumerate(res, start=1):
                item['rank'] = rank

            save_json(res, 'json/dividends/list/dividend-kings.json')
    
    finally:
        # Ensure the WebDriver is closed
        driver.quit()

if __name__ == '__main__':
    main()