import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from functions import *  # Your function implementations

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
STOCKNEAR_API_KEY = os.getenv("STOCKNEAR_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")
API_URL = "http://localhost:8000"
instructions = 'Retrieve data by using functions based on the correct description. Always interpret and validate user metrics, default to descending sort, fall back to the last-mentioned metric if unspecified, invoke the correct data functions, and verify results before returning.' #os.getenv("INSTRUCTIONS").replace("\\n", "\n")

# Dynamically gather function definitions and map names to callables
function_definitions = get_function_definitions()
function_map = {fn["name"]: globals()[fn["name"]] for fn in function_definitions}

# Initialize chat history
messages = [{"role": "system", "content": instructions}]

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    # Ask GPT which function to call, if any
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=[{"type": "function", "function": fn} for fn in function_definitions],
        tool_choice="auto"
    )

    message = response.choices[0].message
    tool_calls = message.tool_calls or []

    if tool_calls:
        # Process each tool call
        for call in tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            if name in function_map:
                # Execute the corresponding Python function
                result = function_map[name](**args)
            else:
                result = {"error": f"Unknown function: {name}"}

            # Append function call and result to messages
            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(result)
            })

        # Stream the final assistant response
        print("\nAssistant:", end=" ", flush=True)
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
        print()

    else:
        # No function call: just stream the response
        print("\nAssistant:", end=" ", flush=True)
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
        print()
