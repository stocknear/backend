from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from geopy.geocoders import Nominatim
import aiohttp
import asyncio
import orjson
import ujson
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
from openai import OpenAI
from datetime import datetime
import hashlib

def generate_unique_id(data):
    # Concatenate the title and date to form a string
    unique_str = f"{data['title']}-{data['date']}"
    
    # Hash the concatenated string to ensure uniqueness
    unique_id = hashlib.md5(unique_str.encode()).hexdigest()
    
    return unique_id


load_dotenv()
geolocator = Nominatim(user_agent="myGeocodingApp/1.0 (your-email@example.com)")

openai_api_key = os.getenv('OPENAI_API_KEY')
org_id = os.getenv('OPENAI_ORG')
client = OpenAI(
    api_key=openai_api_key,
    organization=org_id,
)


query_template = """
    SELECT
        date, close
    FROM
        "{symbol}"
    WHERE
        date BETWEEN ? AND ?
"""


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


def get_summary(data):
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Don't use quotes or titles or bullet points. Provide a clear and concise summary of the US president's executive order. To break the section use <br> to make it html compatible. Explain its potential impact on the stock market, indicating whether it is likely to be bullish, bearish, or neutral, and justify your reasoning based on key aspects of the order. Keep it under 600 characters."
                },
                {"role": "user", "content": data_string}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content
        data['description'] = summary
        
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

async def get_data():
    executive_orders = get_executive_orders()

    executive_orders_summary = []

    for item in executive_orders:
        try:
            data = get_summary(item)
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

    return_since = round((sp500_list[-1]['close']/sp500_list[0]['close']-1)*100,2)

    url = "https://media-cdn.factba.se/rss/json/trump/calendar-full.json"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                # Filter out items with None for date or time, then sort
                data = sorted(
                    (item for item in data if item['date'] is not None and item['time'] is not None),
                    key=lambda x: (x['date'], x['time']),
                    reverse=True
                )

            else:
                print(f"Failed to fetch data. HTTP status code: {response.status}")

    if len(data) > 0 and len(executive_orders_summary) > 0:
        # Latest location
        details = data[0]['details']
        location = data[0]['location']

        
        for address in [details, location]:
            try:
                if any(place in address for place in ["White House", "Blair House","Washington DC", "East Room"]):
                    location = "Washington, DC"
                else:
                    location = address  # Otherwise, use the full address string

                # Geocode the processed address
                location_data = geolocator.geocode(location)
                city = location_data.address.split(',', 1)[0]
                if location_data:
                    
                    # Extract city from the address components
                    address_components = location_data.raw.get('address', {})
                    
                   
                    # Extract latitude and longitude
                    latitude = location_data.latitude
                    longitude = location_data.longitude
                    print(f"Latitude: {latitude}, Longitude: {longitude}")
                    break
            except:
                pass

        for item in data:
            for price_item in sp500_list:
                if item['date'] == price_item['date']:
                    item['changesPercentage'] = price_item['changesPercentage']
                    break
        res_dict = {'returnSince': return_since,'city': city, 'lon': longitude, 'lat': latitude, 'history': data, 'executiveOrders': executive_orders_summary}
        save_json(res_dict)
    
    
asyncio.run(get_data())