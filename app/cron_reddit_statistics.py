import json
import re
import requests
import praw
from datetime import datetime
from collections import defaultdict

import os
from dotenv import load_dotenv

load_dotenv()
client_key = os.getenv('REDDIT_API_KEY')
client_secret = os.getenv('REDDIT_API_SECRET')
user_agent = os.getenv('REDDIT_USER_AGENT')


# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=client_key,
    client_secret=client_secret,
    user_agent=user_agent
)

# Function to save data
def save_data(data):
    with open('json/reddit-tracker/wallstreetbets/stats.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_daily_statistics(file_path):
    # Load the data from the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize a defaultdict to store daily statistics
    daily_stats = defaultdict(lambda: {
        'post_count': 0, 
        'total_comments': 0, 
        'ticker_mentions': defaultdict(int),
        'unique_tickers': set()
    })
    
    # Compile regex pattern for finding tickers
    ticker_pattern = re.compile(r'\$([A-Z]+)')
    
    # Process each post
    for post in data:
        # Convert UTC timestamp to datetime object
        post_date = datetime.utcfromtimestamp(post['created_utc']).date()
        
        # Update statistics for this day
        daily_stats[post_date]['post_count'] += 1
        daily_stats[post_date]['total_comments'] += post['num_comments']
        
        # Find ticker mentions in title and selftext
        text_to_search = post['title'] + ' ' + post['selftext']
        tickers = ticker_pattern.findall(text_to_search)
        
        for ticker in tickers:
            daily_stats[post_date]['ticker_mentions'][ticker] += 1
            daily_stats[post_date]['unique_tickers'].add(ticker)
    
    # Calculate averages and format the results
    formatted_stats = []
    for date, stats in sorted(daily_stats.items(), reverse=True):
        formatted_stats.append({
            'date': date.isoformat(),
            'totalPosts': stats['post_count'],
            'totalComments': stats['total_comments'],
            'totalMentions': sum(stats['ticker_mentions'].values()),
            'companySpread': len(stats['unique_tickers']),
            'tickerMentions': dict(stats['ticker_mentions'])  # Optional: include detailed ticker mentions
        })
    
    return formatted_stats

# Usage
file_path = 'json/reddit-tracker/wallstreetbets/data.json'
daily_statistics = compute_daily_statistics(file_path)
save_data(daily_statistics)