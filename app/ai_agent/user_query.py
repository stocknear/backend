import os
from dotenv import load_dotenv
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
import time
import re

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_store_id = os.getenv("VECTOR_STORE_ID")
CHAT_MODEL = os.getenv("CHAT_MODEL")
instructions = os.getenv("INSTRUCTIONS", "").replace("\\n", "\n")

# Create or retrieve the assistant
assistant = client.beta.assistants.create(
    name="Financial Analyst Assistant",
    model=CHAT_MODEL,
    instructions=(instructions),
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
)

# Global conversation history list
conversation_history = []

# Function to load initial conversation history from a thread
def load_conversation_history(thread_id):
    global conversation_history
    conversation_history = []
    
    messages = client.beta.threads.messages.list(
        thread_id=thread_id,
        order="asc"
    )
    
    for msg in messages.data:
        role = msg.role
        content = [content_block.text.value for content_block in msg.content if content_block.type == 'text']
        content_text = "\n".join(content)
        created_at = msg.created_at
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at))
        
        conversation_history.append({
            "role": role,
            "message": content_text,
            "timestamp": timestamp,
            "message_id": msg.id
        })
    
    return conversation_history

# Use existing thread ID if provided, otherwise create a new one
def get_thread(thread_id=None):
    if thread_id:
        # Retrieve existing thread
        try:
            thread = client.beta.threads.retrieve(thread_id)
            print(f"Using existing thread: {thread.id}")
            # Load conversation history for existing thread
            load_conversation_history(thread.id)
            return thread
        except Exception as e:
            print(f"Error retrieving thread: {e}")
            print("Creating a new thread instead.")
            thread = client.beta.threads.create()
            print(f"New thread created: {thread.id}")
            return thread
    else:
        # Create a new thread
        thread = client.beta.threads.create()
        print(f"New thread created: {thread.id}")
        return thread

# Define event handler - this is the fixed version
class CustomEventHandler(AssistantEventHandler):
    def __init__(self, thread_id):
        super().__init__()  # Call the parent class constructor
        self.thread_id = thread_id
        self.current_message_id = None
        self.current_message_content = ""
    
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)
        self.current_message_content = ""
    
    @override
    def on_text_delta(self, delta, snapshot) -> None:
        # Clean any citation markers or file references from the text
        clean_text = delta.value
        # Remove citation patterns like 【4:2†AAPL.json】
        clean_text = re.sub(r'【[^】]*】', '', clean_text)
        # Remove citation patterns like [1], [2], etc.
        clean_text = re.sub(r'\[\d+\]', '', clean_text)
        print(clean_text, end="", flush=True)
        self.current_message_content += clean_text
    
    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)
    
    @override
    def on_message_done(self, message) -> None:
        global conversation_history
        # Add assistant's response to conversation history
        created_at = message.created_at
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at))
        
        conversation_history.append({
            "role": "assistant",
            "message": self.current_message_content,
            "timestamp": timestamp,
            "message_id": message.id
        })
        
        # Debug: print current conversation history length
        print(f"\n[Debug] Conversation history now has {len(conversation_history)} messages")

def process_user_query(query, thread):
    """Process a user query about stock information."""
    global conversation_history
    
    # Add the user's query to the thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query
    )
    
    # Add user message to conversation history
    created_at = message.created_at
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at))
    
    conversation_history.append({
        "role": "user",
        "message": query,
        "timestamp": timestamp,
        "message_id": message.id
    })
    
    # Create a run with the assistant and stream the response
    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        event_handler=CustomEventHandler(thread.id)
    ) as stream:
        stream.until_done()

def run_interactive_session(thread_id=None):
    """Run an interactive session where users can query stock information."""
    # Get thread - either existing or new
    thread = get_thread(thread_id)
    
    print("=== Stocknear LLM Financial Assistant ===")
    print("Ask questions about stock prices and financial information.")
    print("Type 'exit', 'quit' to end the session or 'history' to see conversation history.\n")
    print(f"Current thread ID: {thread.id}")
    
    while True:
        user_input = input("\nYour query: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("\nEnding session. Thanks for using Stocknear LLM!")
            print(f"Thread ID for future reference: {thread.id}")
            break
        
        if user_input.lower() == 'history':
            print("\n=== Conversation History ===")
            print(conversation_history)
            for i, msg in enumerate(conversation_history):
                print(f"[{msg['timestamp']}] {msg['role']}: {msg['message'][:100]}...")
            print("===========================\n")
            continue
            
        process_user_query(user_input, thread)

if __name__ == "__main__":
    # You can provide a thread ID here to continue an existing conversation
    existing_thread_id = "thread_zMh6Vi3GvgLwHx36XgEAfk9w"
    if not existing_thread_id:
        existing_thread_id = None
    
    run_interactive_session(existing_thread_id)