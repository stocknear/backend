from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import aiohttp
import asyncio
import orjson
import ujson
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
from openai import AsyncOpenAI

from datetime import datetime, timedelta
import hashlib

def generate_unique_id(data):
    # Concatenate the title and date to form a string
    unique_str = f"{data['title']}-{data['date']}"
    
    # Hash the concatenated string to ensure uniqueness
    unique_id = hashlib.md5(unique_str.encode()).hexdigest()
    
    return unique_id


load_dotenv()

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chat_model = os.getenv("CHAT_MODEL")

query_template = """
    SELECT
        date, close
    FROM
        "{symbol}"
    WHERE
        date BETWEEN ? AND ?
"""

today = datetime.now().date()
cutoff = today - timedelta(days=9)

def save_json(data):
    path = "json/tracker/potus"
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/data.json", "wb") as file:
        file.write(orjson.dumps(data))

# Set up the Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run browser in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Replace 'path/to/chromedriver' with your actual chromedriver path
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

url ="https://www.whitehouse.gov/presidential-actions/"
driver.get(url)


async def get_summary(data):
    unique_id = generate_unique_id(data)  # Assuming this function exists
    
    # Check if the file exists
    file_path = f"json/executive-orders/{unique_id}.json"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists, skipping summary generation.")
        return
    
    try:
        data_string = f"Analyze this executive order: {data['description']}"
        response = await async_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Don't use quotes or titles or bullet points. Provide a clear and concise summary of the US president's executive order. To break the section use <br> to make it html compatible. Explain its potential impact on the stock market, indicating whether it is likely to be bullish, bearish, or neutral, and justify your reasoning based on key aspects of the order. Keep it under 600 characters."
                },
                {"role": "user", "content": data_string}
            ],
        )
        
        summary = response.choices[0].message.content
        data['description'] = summary
        print(data['description'])

        # Save the data with the generated summary
        with open(file_path, "w", encoding="utf-8") as file:
            json_str = ujson.dumps(data)
            file.write(json_str)
        
        return json_str
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        

def get_executive_orders():
    url = "https://www.whitehouse.gov/presidential-actions/"
    
    # Set up headless Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get(url)

    try:
        # Wait for executive orders list to load
        wait = WebDriverWait(driver, 10)
        orders = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.wp-block-post-template > li")))

        executive_orders = []
        
        # First pass to collect basic information
        for order in orders:
            try:
                title_element = order.find_element(By.CSS_SELECTOR, "h2.wp-block-post-title a")
                title = title_element.text.strip()
                link = title_element.get_attribute("href")

                date_element = order.find_element(By.CSS_SELECTOR, "div.wp-block-post-date time")
                date_raw = date_element.get_attribute("datetime").split("T")[0]
                date_formatted = datetime.strptime(date_raw, "%Y-%m-%d").strftime("%Y-%m-%d")

                executive_orders.append({
                    "title": title,
                    "date": date_formatted,
                    "link": link,
                    "description": None  # Initialize description field
                })
            except Exception as e:
                print(f"Error processing an executive order: {e}")

        # Second pass to collect descriptions
        for eo in executive_orders:
            try:
                driver.get(eo['link'])
                
                # Wait for description content to load
                desc_wait = WebDriverWait(driver, 10)
                description_element = desc_wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.entry-content.wp-block-post-content"))
                )
                
                # Extract and clean text
                eo['description'] = description_element.text.strip()
                
            except Exception as e:
                print(f"Error fetching description for {eo['link']}: {e}")
                eo['description'] = "Description unavailable"

        return executive_orders

    finally:
        driver.quit()



async def get_historical_sector():
    sector_list = ["SPY","XLB", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLRE", "XLK", "XLU"]
    res_dict = {}
    
    def calculate_percentage_change(current_price, previous_price):
        if previous_price == 0:
            return 0
        return ((current_price - previous_price) / previous_price) * 100
    
    def find_closest_date(data, target_date):
        # Find the closest date entry equal to or before the target date
        target_date = datetime.strptime(target_date, '%Y-%m-%d')
        for entry in reversed(data):  # Reverse to search from newest to oldest
            entry_date = datetime.strptime(entry['time'], '%Y-%m-%d')
            if entry_date <= target_date:
                return entry
        return None

    for symbol in sector_list:
        try:
            # Load historical data
            with open(f"json/historical-price/max/{symbol}.json", "r") as file:
                data = orjson.loads(file.read())
            
            # Load current data for 1D change
            with open(f"json/quote/{symbol}.json", "r") as file:
                current_data = round(orjson.loads(file.read()).get('changesPercentage', 0),2)
            
            if not data:
                continue
                
            # Get the latest price (last item in the list)
            latest_price = data[-1]['close']
            
            # Calculate dates for different periods
            today = datetime.strptime(data[-1]['time'], '%Y-%m-%d')
            dates = {
                '1W': (today - timedelta(days=7)).strftime('%Y-%m-%d'),
                '1M': (today - timedelta(days=30)).strftime('%Y-%m-%d'),
                '3M': (today - timedelta(days=90)).strftime('%Y-%m-%d'),
                '6M': (today - timedelta(days=180)).strftime('%Y-%m-%d'),
                'Inauguration': '2025-01-20'
            }
            
            changes = {'1D': current_data}
            
            # Calculate percentage changes for each period
            for period, target_date in dates.items():
                historical_entry = find_closest_date(data, target_date)
                if historical_entry:
                    change = calculate_percentage_change(latest_price, historical_entry['close'])
                    changes[period] = round(change, 2)
                else:
                    changes[period] = 0
            
            res_dict[symbol] = changes
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    return res_dict

async def get_truth_social_post():
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get("https://trumpstruth.org/?per_page=40")

    wait = WebDriverWait(driver, 20)
    statuses = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'status')))

    posts_data = []

    for status in statuses:
        try:
            # Extract username
            username = status.find_element(By.CLASS_NAME, 'status-info__account-name').text
        except NoSuchElementException:
            username = "N/A"
        
        try:
            # Extract date (second meta item)
            meta_items = status.find_elements(By.CLASS_NAME, 'status-info__meta-item')
            date = meta_items[1].text if len(meta_items) >= 2 else "N/A"
        except (NoSuchElementException, IndexError):
            date = "N/A"
        
        try:
            # Extract content text
            content_element = status.find_element(By.CLASS_NAME, 'status__content')
            content = content_element.text.strip()
        except NoSuchElementException:
            content = ""
        
        # Extract video URL if present
        video_url = ""
        try:
            video_element = status.find_element(By.CSS_SELECTOR, '.status-attachment--video video')
            video_url = video_element.get_attribute('src')
        except NoSuchElementException:
            pass
        
        # Extract external link details if present
        external_link = ""
        link_title = ""
        link_description = ""
        try:
            card = status.find_element(By.CLASS_NAME, 'status-card')
            external_link = card.get_attribute('href')
            link_title = card.find_element(By.CLASS_NAME, 'status-card__title').text
            link_description = card.find_element(By.CLASS_NAME, 'status-card__description').text
        except NoSuchElementException:
            pass
        
        # Extract original post URL
        try:
            original_post_url = status.find_element(By.CLASS_NAME, 'status__external-link').get_attribute('href')
        except NoSuchElementException:
            original_post_url = ""

        posts_data.append({
            'date': date,
            'content': content,
            'videoUrl': video_url,
            'externalLink': external_link,
            'title': link_title,
            'source': original_post_url
        })

    posts_data = [item for item in posts_data if item['videoUrl'] == "" and "youtube" not in item['content'] and item['content'] != ""]
    return posts_data



async def get_data():

    post_list = await get_truth_social_post()
    market_dict = await get_historical_sector()
    
    executive_orders = get_executive_orders()

    executive_orders_summary = []

    for item in executive_orders:
        try:
            data = await get_summary(item)
        except Exception as e:
            print(e)

    for item in executive_orders:
        try:
            unique_id = generate_unique_id(item)
            
            # Open and read the JSON file
            with open(f"json/executive-orders/{unique_id}.json", "r") as file:
                data = orjson.loads(file.read())
                
                # Assign sentiment based on words in the description
                if 'bullish' in data['description']:
                    data['sentiment'] = 'Bullish'
                elif 'bearish' in data['description']:
                    data['sentiment'] = 'Bearish'
                else:
                    data['sentiment'] = 'Neutral'
            
            executive_orders_summary.append(data)
        except Exception as e:
            print(f"Error processing item {item}: {e}")

    
    query = query_template.format(symbol='SPY')

    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")

    df = pd.read_sql_query(query, etf_con, params=("2025-01-20", datetime.today().strftime("%Y-%m-%d")))
    if not df.empty:
        df['changesPercentage'] = (df['close'].pct_change() * 100).round(2)
        sp500_list = df.dropna().to_dict(orient="records")   # Drop NaN values and convert to list
    etf_con.close()


    url = "https://media-cdn.factba.se/rss/json/trump/calendar-full.json"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                # Filter out items with None for date or time, then sort
                data = [
                    e for e in data
                    if datetime.strptime(e['date'], '%Y-%m-%d').date() >= cutoff
                ]

                keys_to_remove = ['lastdaily', 'video_url', 'daycount', 'newmonth', 'url']

                # produce a new list without those keys
                data = [
                    {k: v for k, v in entry.items() if k not in keys_to_remove}
                    for entry in data
                ]

                data = sorted(
                    (item for item in data if item['date'] is not None and item['time'] is not None),
                    key=lambda x: (x['date'], x['time']),
                    reverse=True
                )

            else:
                print(f"Failed to fetch data. HTTP status code: {response.status}")

    if len(data) > 0 and len(executive_orders_summary) > 0:
        
        for item in data:
            for price_item in sp500_list:
                if item['date'] == price_item['date']:
                    item['changesPercentage'] = price_item['changesPercentage']
                    break
        res_dict = {'posts': post_list, 'marketPerformance': market_dict, 'history': data, 'executiveOrders': executive_orders_summary}
        save_json(res_dict)

    
    
asyncio.run(get_data())