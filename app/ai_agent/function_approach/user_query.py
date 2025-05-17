import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
from functions import *

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

STOCKNEAR_API_KEY = os.getenv("STOCKNEAR_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")
API_URL = "http://localhost:8000"
instructions = os.getenv("INSTRUCTIONS").replace("\\n", "\n")

function_definitions = get_function_definitions()


# Initialize chat history
messages = [{"role": "system", "content": instructions}]

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    # First GPT call to decide on function usage
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=[{"type": "function", "function": function_definitions[0]}],
        tool_choice="auto"
    )

    tool_call = response.choices[0].message.tool_calls
    if tool_call:
        function_name = tool_call[0].function.name
        arguments = json.loads(tool_call[0].function.arguments)
        if function_name == "get_financial_statements":
            tool_output = get_financial_statements(arguments["ticker"], arguments["statement"])

            # Add function call and tool response to messages
            messages.append(response.choices[0].message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call[0].id,
                "name": function_name,
                "content": json.dumps(tool_output)
            })

            # Second GPT call with tool result — stream this part
            print("\nAssistant:", end=" ", flush=True)
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                try:
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                except:
                    pass
            print()
    else:
        # No function call — just stream the answer
        print("\nAssistant:", end=" ", flush=True)
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            try:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
            except:
                pass
        print()
