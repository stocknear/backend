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
    # Load stock symbols from the database
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    # Optionally load environment variables; you may also hardcode the URL below.
    load_dotenv()
    # Use the correct URL for scraping IPO news:
    url = os.getenv('IPO_NEWS', 'https://stockanalysis.com/ipos/news/')

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
        driver.implicitly_wait(20)

        # Use a flexible selector for news containers
        news_items = driver.find_elements(
            By.CSS_SELECTOR, 
            "div[class*='border-gray-300'][class*='bg-white'][class*='p-4']"
        )

        news_data = []
        for item in news_items:
            try:
                # Extract elements using flexible selectors
                title_element = item.find_element(By.CSS_SELECTOR, "h3 a")
                description_element = item.find_element(By.CSS_SELECTOR, "p.overflow-auto")
                timestamp_element = item.find_element(By.CSS_SELECTOR, "div.text-sm.text-faded")
                stocks_elements = item.find_elements(By.CSS_SELECTOR, "a.ticker")
                img_element = item.find_element(By.CSS_SELECTOR, "img.w-full.rounded.object-cover")

                # Use textContent and strip whitespace
                title = title_element.get_attribute("textContent").strip()
                description = description_element.get_attribute("textContent").strip()
                timestamp = timestamp_element.get_attribute("textContent").strip()
                link = title_element.get_attribute("href")
                stocks = [stock.text.strip() for stock in stocks_elements]
                img_link = img_element.get_attribute("src")

                # Skip the news item if the title is empty
                if not title:
                    continue

                # Filter stocks that exist in your database
                stock_list = [symbol for symbol in stocks if symbol in stock_symbols]

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

        # Convert the collected data into a DataFrame and save it to a JSON file if not empty
        df = pd.DataFrame(news_data)
        if not df.empty:
            df.to_json(json_file_path, orient='records', indent=2)
        else:
            print("No news items were found.")
    finally:
        driver.quit()

if __name__ == '__main__':
    main()
