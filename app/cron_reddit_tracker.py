import praw
import orjson
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
        with open(file_path, 'r', encoding='utf-8') as file:
            return orjson.loads(file.read())
    return []

# Function to save data
def save_data(data):
    with open(file_path, 'w', encoding='utf-8') as f:
        file.write(orjson.dumps(data, f, ensure_ascii=False, indent=4).decode("utf-8"))

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=client_key,
    client_secret=client_secret,
    user_agent=user_agent
)

# Load existing data
existing_data = load_existing_data()
existing_data = [
    {**item, 'upvote_ratio': round(item['upvote_ratio'] * 100,2) if item['upvote_ratio'] < 1 else item['upvote_ratio']}
    for item in existing_data
    if item['num_comments'] >= 50
]




# Create a dictionary of existing posts for faster lookup and update
existing_posts = {post['id']: post for post in existing_data}

# Flag to check if any data was added or updated
data_changed = False

# Get the subreddit
subreddit = reddit.subreddit("wallstreetbets")

# Iterate through new submissions
for submission in subreddit.hot(limit=1000):
    post_id = submission.id
    
    # Check if the post was deleted by moderators
    if submission.removed_by_category == "mod":
        # Remove post from existing data if it was deleted by moderators
        if post_id in existing_posts:
            del existing_posts[post_id]
            data_changed = True
            print('deleted')
        continue  # Skip this post

    if submission.num_comments < 50:
        # Remove post from existing data if it was deleted by moderators
        if post_id in existing_posts:
            del existing_posts[post_id]
            data_changed = True
            print('deleted')
        continue  # Skip this post

    # Check if this post is already in our data
    if post_id in existing_posts:
        # Update existing post
        existing_posts[post_id]['upvote_ratio'] = round(submission.upvote_ratio * 100, 2)
        existing_posts[post_id]['num_comments'] = submission.num_comments
        data_changed = True
    else:
        if submission.num_comments < 50:
            continue  # Skip this post
        
        # Try to get a high-quality thumbnail URL
        thumbnail = None
        if hasattr(submission, 'preview'):
            thumbnail = submission.preview['images'][0]['source']['url']
        
        # Extract the required fields for new post
        extracted_post = {
            "id": post_id,
            "permalink": submission.permalink,
            "title": submission.title,
            "thumbnail": thumbnail,
            "selftext": submission.selftext,
            "created_utc": int(submission.created_utc),
            "upvote_ratio": round(submission.upvote_ratio * 100, 2),
            "num_comments": submission.num_comments,
            "link_flair_text": submission.link_flair_text,
            "author": str(submission.author),
        }
        
        # Add the new post to the existing data
        existing_posts[post_id] = extracted_post
        data_changed = True

if data_changed:
    # Convert the dictionary back to a list and sort by created_utc
    updated_data = list(existing_posts.values())
    updated_data.sort(key=lambda x: x['created_utc'], reverse=True)
    
    # Save the updated data
    save_data(updated_data)
    print(f"Data updated and saved to {file_path}")
else:
    print("No new data to add or update.")
