import os
import pandas as pd
import sqlite3
import requests
from bs4 import BeautifulSoup
import re
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
    url = os.getenv('IPO_NEWS')

    json_file_path = 'json/market-news/ipo-news.json'

    try:
        # Set up headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Fetch the website
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find news items - they are in divs with specific classes
        # Look for news containers with the pattern matching your HTML structure
        news_items = soup.find_all('div', class_=re.compile(r'gap-4.*border-gray-300.*bg-default'))
        
        # Also find video news items (different structure)
        video_items = soup.find_all('div', class_=re.compile(r'flex flex-col.*border-gray-300.*bg-default'))

        news_data = []
        
        # Process regular news items
        for item in news_items:
            try:
                # Extract title and link
                title_elem = item.find('h3')
                if not title_elem:
                    continue
                    
                link_elem = title_elem.find('a')
                if not link_elem:
                    continue
                    
                title = link_elem.get_text(strip=True)
                link = link_elem.get('href', '')
                
                # Extract description
                desc_elem = item.find('p', class_=re.compile(r'text-\[0\.95rem\]'))
                description = desc_elem.get_text(strip=True) if desc_elem else ''
                
                # Extract timestamp
                time_elem = item.find('div', class_=re.compile(r'text-sm text-faded'))
                timestamp = time_elem.get_text(strip=True) if time_elem else ''
                
                # Extract image URL
                img_elem = item.find('img')
                img_link = img_elem.get('src', '') if img_elem else ''
                
                # Extract stock tickers
                ticker_links = item.find_all('a', class_='ticker')
                stocks = [ticker.get_text(strip=True) for ticker in ticker_links]
                
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
        
        # Process video items if they have a different structure
        for item in video_items:
            try:
                # Check if this is actually a video item
                if not item.find('div', class_=re.compile(r'bg-black.*cursor-pointer')):
                    continue
                
                # Extract title
                title_elem = item.find('h3')
                if not title_elem:
                    continue
                    
                title = title_elem.get_text(strip=True)
                
                # Extract description
                desc_elem = item.find('p', class_=re.compile(r'text-\[0\.95rem\]'))
                description = desc_elem.get_text(strip=True) if desc_elem else ''
                
                # Extract timestamp
                time_elem = item.find('div', class_=re.compile(r'text-sm text-faded'))
                timestamp = time_elem.get_text(strip=True) if time_elem else ''
                
                # Extract stock tickers
                ticker_links = item.find_all('a', class_='ticker')
                stocks = [ticker.get_text(strip=True) for ticker in ticker_links]
                
                # Filter stocks that exist in your database
                stock_list = [symbol for symbol in stocks if symbol in stock_symbols]
                
                # For video items, we might not have a direct link
                link = ''
                img_link = ''
                
                news_data.append({
                    "title": title,
                    "description": description,
                    "timestamp": timestamp,
                    "link": link,
                    "stocks": stock_list,
                    "img": img_link
                })

            except Exception as e:
                print(f"Error extracting video news item: {e}")

        # Convert the collected data into a DataFrame and save it to a JSON file if not empty
        df = pd.DataFrame(news_data)
        if not df.empty:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
            df.to_json(json_file_path, orient='records', indent=2)
            print(f"Successfully scraped {len(news_data)} news items")
        else:
            print("No news items were found.")
            
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
