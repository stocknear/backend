import requests
import json
from datetime import datetime
import os

# URL of the Reddit API endpoint
url = "https://www.reddit.com/r/wallstreetbets/new.json"
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

# Function to get updated post data
def get_updated_post_data(permalink):
    post_url = f"https://www.reddit.com{permalink}.json"
    response = requests.get(post_url, headers={'User-agent': 'Mozilla/5.0'})
    if response.status_code == 200:
        post_data = response.json()[0]['data']['children'][0]['data']
        return post_data
    return None

# Load existing data
existing_data = load_existing_data()

# Create a dictionary of existing posts for faster lookup and update
existing_posts = {post['id']: post for post in existing_data}

# Send a GET request to the API
response = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})

counter = 0
# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON data
    data = response.json()
    
    # Flag to check if any data was added or updated
    data_changed = False
    
    # Iterate through each post in the 'children' list
    for post in data['data']['children']:
        post_data = post['data']
        post_id = post_data.get('id', '')
        
        # Check if this post is already in our data
        if post_id in existing_posts:
            # Update existing post
            if counter < 25: #Only update the latest 25 posts to not overload the reddit server
	            updated_data = get_updated_post_data(post_data['permalink'])
	            if updated_data:
	                existing_posts[post_id]['upvote_ratio'] = updated_data.get('upvote_ratio', existing_posts[post_id]['upvote_ratio'])
	                existing_posts[post_id]['num_comments'] = updated_data.get('num_comments', existing_posts[post_id]['num_comments'])
	                data_changed = True
	                counter +=1
	                print(counter)
        else:
            # Extract the required fields for new post
            extracted_post = {
                "id": post_id,
                "permalink": post_data.get('permalink', ''),
                "title": post_data.get('title', ''),
                "selftext": post_data.get('selftext', ''),
                "created_utc": post_data.get('created_utc', ''),
                "upvote_ratio": post_data.get('upvote_ratio', ''),
                "num_comments": post_data.get('num_comments', ''),
                "link_flair_text": post_data.get('link_flair_text', ''),
                "author": post_data.get('author', ''),
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
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")