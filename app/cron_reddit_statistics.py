import json
import re
import requests
from datetime import datetime
from collections import defaultdict

def get_subscriber_count():
    url = "https://www.reddit.com/r/wallstreetbets/new.json"
    headers = {'User-agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data['data']['children'][0]['data']['subreddit_subscribers']
    return None

def compute_daily_statistics(file_path):
    # Load the data from the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get current subscriber count
    subscriber_count = get_subscriber_count()
    
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
            'subscribersCount': subscriber_count,
            'totalMentions': sum(stats['ticker_mentions'].values()),
            'companySpread': len(stats['unique_tickers']),
            'tickerMentions': dict(stats['ticker_mentions'])  # Optional: include detailed ticker mentions
        })
    
    return formatted_stats

# Usage
file_path = 'json/reddit-tracker/wallstreetbets/data.json'
daily_statistics = compute_daily_statistics(file_path)
print(json.dumps(daily_statistics, indent=2))