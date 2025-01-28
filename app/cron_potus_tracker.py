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
from dotenv import load_dotenv
import os

load_dotenv()
geolocator = Nominatim(user_agent="myGeocodingApp/1.0 (your-email@example.com)")

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

# Open the URL
url = os.getenv('POTUS_TRACKER')
driver.get(url)

def get_bills():
    try:
        # Wait for the page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "legislation-container"))
        )

        # Locate the legislation container
        legislation_container = driver.find_element(By.ID, "legislation-container")
        legislation_items = legislation_container.find_elements(By.CLASS_NAME, "legislation-item")

        # Extract data
        data = []
        for item in legislation_items:
            # Badge
            badge = item.find_element(By.CLASS_NAME, "badge").text

            # Header (Title)
            header = item.find_element(By.CLASS_NAME, "legislation-header").text

            # Description
            description = item.find_element(By.CLASS_NAME, "legislation-description").text

            # Time Ago (if present)
            time_ago_element = item.find_elements(By.CLASS_NAME, "datetime-ago")
            time_ago = time_ago_element[0].text if time_ago_element else None

            # Meta Info (e.g., status)
            meta_info_elements = item.find_elements(By.CLASS_NAME, "legislation-meta")
            meta_info = []
            if meta_info_elements:
                for meta_item in meta_info_elements[0].find_elements(By.TAG_NAME, "div"):
                    meta_info.append(meta_item.text.strip())

            # Check if there's a "Read More" button to click
            read_more_buttons = item.find_elements(By.CLASS_NAME, "read-more-btn")  # Now using correct class
            if read_more_buttons:
                print("Found 'Read More' button, clicking it...")
                # Click the "Read More" button
                read_more_buttons[0].click()

                # Wait for the popup to become visible
                print("Waiting for the popup to appear...")
                WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located((By.ID, "popup-container"))  # Wait until popup is visible
                )

                # Extract content from the popup
                print("Popup appeared, extracting content...")
                popup_title = driver.find_element(By.ID, "popup-title").text
                popup_content = driver.find_element(By.ID, "popup-content").text

           
                # Add the popup content and URL to the description (optional)
                description = f"{popup_content}"

                # Close the popup (optional)
                close_button = driver.find_element(By.ID, "popup-close-button")
                close_button.click()
                print("Popup closed.")

            # Append data to list
            data.append({
                "badge": badge,
                "title": header,
                "description": description,
                "time": time_ago,
            })

        # Print scraped data
        
        return data
    finally:
        # Close the driver
        driver.quit()



async def get_data():
    bill_data = get_bills()

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

    if len(data) > 0 and len(bill_data) > 0:
        # Latest location
        details = data[0]['details']
        location = data[0]['location']

        
        for address in [details, location]:
            if any(place in address for place in ["White House", "Blair House","Washington DC"]):
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

        res_dict = {'city': city, 'lon': longitude, 'lat': latitude, 'history': data, 'billData': bill_data}
        save_json(res_dict)


asyncio.run(get_data())