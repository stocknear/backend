import os
import pandas as pd
import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv

def clean_link(url):
    """
    Clean the article link to extract the actual URL if it's wrapped in a redirect.
    """
    if 'url=' in url:
        return url.split('url=')[-1]
    return url

def main():
    # Load environment variables
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    load_dotenv()
    url = os.getenv('IPO_NEWS')  # IPO news URL

    # Set up the WebDriver options
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Initialize the WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    json_file_path = 'json/market-news/ipo-news.json'

    try:
        # Fetch the website
        driver.get(url)
        # Wait for the page to load
        driver.implicitly_wait(5)

        # Find all the news containers
        news_items = driver.find_elements(By.CSS_SELECTOR, ".gap-4.border-gray-300.bg-white.p-4.shadow.last\\:pb-1")

        # Extract data from the containers
        news_data = []
        for item in news_items:
            try:
                # Extract relevant elements
                title_element = item.find_element(By.CSS_SELECTOR, "h3 a")
                description_element = item.find_element(By.CSS_SELECTOR, "p")
                timestamp_element = item.find_element(By.CSS_SELECTOR, ".text-sm.text-faded")
                stocks_element = item.find_elements(By.CSS_SELECTOR, ".ticker")
                img_element = item.find_element(By.CSS_SELECTOR, "img.h-full.w-full.rounded.object-cover")

                # Get element data
                title = title_element.text
                description = description_element.text
                timestamp = timestamp_element.text
                link = title_element.get_attribute("href")
                stocks = [stock.text for stock in stocks_element]
                img_link = img_element.get_attribute("src")

                # Filter stocks that exist in the database
                stock_list = [symbol for symbol in stocks if symbol in stock_symbols]

                # Add to news data
                news_data.append({
                    "title": title,
                    "description": description,
                    "timestamp": timestamp,
                    "link": clean_link(link),
                    "stocks": stock_list,
                    "img": img_link
                })

            except Exception as e:
                print(f"Error extracting news item: {e}")

        # Convert the data into a DataFrame
        df = pd.DataFrame(news_data)

        # Save the DataFrame to a JSON file
        df.to_json(json_file_path, orient='records', indent=2)

    finally:
        # Ensure the WebDriver is closed
        driver.quit()


if __name__ == '__main__':
    main()
