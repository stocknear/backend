import praw
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import time

load_dotenv()
client_key = os.getenv('REDDIT_API_KEY')
client_secret = os.getenv('REDDIT_API_SECRET')
user_agent = os.getenv('REDDIT_USER_AGENT')

# File path for the JSON data
file_path = 'json/reddit-tracker/wallstreetbets/data.json'

# Ensure the directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Function to load existing data
def load_existing_data():
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# Function to save data
def save_data(data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=client_key,
    client_secret=client_secret,
    user_agent=user_agent
)

# Load existing data
existing_data = load_existing_data()

# Create a dictionary of existing posts for faster lookup and update
existing_posts = {post['id']: post for post in existing_data}

# Flag to check if any data was added or updated
data_changed = False

# Get the subreddit
subreddit = reddit.subreddit("wallstreetbets")

# Iterate through new submissions
for submission in subreddit.new(limit=1000):
    post_id = submission.id
    # Check if this post is already in our data
    if post_id in existing_posts:
        # Update existing post
        existing_posts[post_id]['upvote_ratio'] = submission.upvote_ratio
        existing_posts[post_id]['num_comments'] = submission.num_comments
        data_changed = True
    else:
        # Extract the required fields for new post
        extracted_post = {
            "id": post_id,
            "permalink": submission.permalink,
            "title": submission.title,
            "selftext": submission.selftext,
            "created_utc": int(submission.created_utc),
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "link_flair_text": submission.link_flair_text,
            "author": str(submission.author),
        }
        
        # Add the new post to the existing data
        existing_posts[post_id] = extracted_post
        data_changed = True

    time.sleep(1)  # Add a 1-second delay between processing submissions

if data_changed:
    # Convert the dictionary back to a list and sort by created_utc
    updated_data = list(existing_posts.values())
    updated_data.sort(key=lambda x: x['created_utc'], reverse=True)
    
    # Save the updated data
    save_data(updated_data)
    print(f"Data updated and saved to {file_path}")
else:
    print("No new data to add or update.")