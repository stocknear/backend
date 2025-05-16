import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import time

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Thread ID from which to retrieve messages
thread_id = "thread_zMh6Vi3GvgLwHx36XgEAfk9w"

# Get all messages from the thread
messages = client.beta.threads.messages.list(
    thread_id=thread_id,
    order="asc",  # Get messages in ascending order (oldest first)
    limit=100     # Adjust this based on your needs
)

# Process and format the conversation
conversation_history = []

for msg in messages.data:
    # Extract relevant information
    role = msg.role  # 'user' or 'assistant'
    content = [content_block.text.value for content_block in msg.content if content_block.type == 'text']
    content_text = "\n".join(content)
    created_at = msg.created_at
    
    # Format timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at))
    
    # Add to conversation history
    conversation_history.append({
        "role": role,
        "message": content_text,
        "timestamp": timestamp,
        "message_id": msg.id
    })

# Print or save the conversation history
print(json.dumps(conversation_history, indent=2))
