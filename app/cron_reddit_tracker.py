import praw
import orjson
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

load_dotenv()
client_key = os.getenv('REDDIT_API_KEY')
client_secret = os.getenv('REDDIT_API_SECRET')
user_agent = os.getenv('REDDIT_USER_AGENT')


run_all = False

# File path for the JSON data
file_path = 'json/reddit-tracker/wallstreetbets/data.json'

# Ensure the directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

def load_existing_data():
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as file:
                return orjson.loads(file.read())
        except:
            return []
    return []

def save_data(data):
    with open(file_path, 'wb') as file:
        file.write(orjson.dumps(data))

def is_within_last_N_days(timestamp):
    current_time = datetime.now()
    post_time = datetime.fromtimestamp(timestamp)
    three_months_ago = current_time - timedelta(days=180)
    return post_time >= three_months_ago

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=client_key,
    client_secret=client_secret,
    user_agent=user_agent
)

# Load existing data
existing_data = load_existing_data()
existing_data = [
    {**item, 'upvote_ratio': round(item['upvote_ratio'] * 100, 2) if item['upvote_ratio'] < 1 else item['upvote_ratio']}
    for item in existing_data if item['num_comments'] >= 50]

# Create a dictionary of existing posts for faster lookup and update
existing_posts = {post['id']: post for post in existing_data}

# Flag to check if any data was added or updated
data_changed = False

# Get the subreddit
subreddit = reddit.subreddit("wallstreetbets")

# Different methods to get posts

#Run once
if run_all == True:
    methods = [
        subreddit.hot(limit=5000),
        subreddit.new(limit=5000),
        subreddit.top(time_filter='month', limit=5000),
        subreddit.top(time_filter='week', limit=5000),
        subreddit.top(time_filter='year', limit=5000),
    ]
else:
    methods = [
        subreddit.hot(limit=1000),
        subreddit.new(limit=1000),
    ]

processed_ids = set()

for submission_stream in methods:
    try:
        for submission in submission_stream:
            post_id = submission.id
            
            # Skip if we've already processed this post
            if post_id in processed_ids:
                continue
            
            processed_ids.add(post_id)
            
            # Check if the post is within the last 3 months
            if not is_within_last_N_days(submission.created_utc):
                continue
            
            # Check if the post was deleted by moderators
            if submission.removed_by_category == "mod":
                if post_id in existing_posts:
                    del existing_posts[post_id]
                    data_changed = True
                    print(f'Deleted post: {post_id}')
                continue
            
            if submission.num_comments < 50:
                if post_id in existing_posts:
                    del existing_posts[post_id]
                    data_changed = True
                    print(f'Removed low-comment post: {post_id}')
                continue
            
            # Check if this post is already in our data
            if post_id in existing_posts:
                # Update existing post
                existing_posts[post_id]['upvote_ratio'] = round(submission.upvote_ratio * 100, 2)
                existing_posts[post_id]['num_comments'] = submission.num_comments
                data_changed = True
            else:
                # Try to get a high-quality thumbnail URL
                thumbnail = None
                if hasattr(submission, 'preview'):
                    try:
                        thumbnail = submission.preview['images'][0]['source']['url']
                    except (KeyError, IndexError):
                        pass
                
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
                print(f'Added new post: {post_id}')
            
            # Sleep briefly to avoid hitting rate limits
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error processing submission stream: {e}")
        continue

if data_changed:
    # Convert the dictionary back to a list and sort by created_utc
    updated_data = list(existing_posts.values())
    updated_data.sort(key=lambda x: x['created_utc'], reverse=True)
    
    # Save the updated data
    save_data(updated_data)
    print(f"Data updated and saved to {file_path}")
    print(f"Total posts in database: {len(updated_data)}")
else:
    print("No new data to add or update.")