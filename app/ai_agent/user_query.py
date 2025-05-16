import os
from dotenv import load_dotenv
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
import time

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_store_id = os.getenv("VECTOR_STORE_ID")
CHAT_MODEL = os.getenv("CHAT_MODEL")

instructions = os.getenv("INSTRUCTIONS", "").replace("\\n", "\n")


assistant = client.beta.assistants.create(
    name="Financial Analyst Assistant",
    model=CHAT_MODEL,
    instructions=(instructions),
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
)

# Create a thread
thread = client.beta.threads.create()

# Define event handler
class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)
    
    @override
    def on_text_delta(self, delta, snapshot) -> None:
        # Clean any citation markers or file references from the text
        clean_text = delta.value
        # Remove citation patterns like 【4:2†AAPL.json】
        import re
        clean_text = re.sub(r'【[^】]*】', '', clean_text)
        # Remove citation patterns like [1], [2], etc.
        clean_text = re.sub(r'\[\d+\]', '', clean_text)
        print(clean_text, end="", flush=True)
    
    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)
    
    @override
    def on_message_done(self, message) -> None:
        # No additional processing needed at end of message
        pass


def process_user_query(query):
    """Process a user query about stock information."""
    #print(f"\nUser Query: {query}")
    
    # Add the user's query to the thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query
    )
    
    # Create a run with the assistant and stream the response
    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        event_handler=EventHandler()
    ) as stream:
        stream.until_done()


def run_interactive_session():
    """Run an interactive session where users can query stock information."""
    print("=== Stocknear LLM Financial Assistant ===")
    print("Ask questions about stock prices and financial information.")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        user_input = input("\nYour query: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("\nEnding session. Thanks for using Stocknear LLM!")
            break
        
        process_user_query(user_input)


if __name__ == "__main__":
    run_interactive_session()