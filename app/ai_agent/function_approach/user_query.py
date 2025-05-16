import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

STOCKNEAR_API_KEY = os.getenv("STOCKNEAR_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
API_URL = "http://localhost:8000"
instructions = os.getenv("INSTRUCTIONS", "You are a helpful financial assistant.").replace("\\n", "\n")

def get_income_data(ticker):
    try:
        response = requests.post(
            f"{API_URL}/financial-statement",
            json={"ticker": ticker, "statement": "income-statement"},
            headers={"X-API-KEY": STOCKNEAR_API_KEY}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

function_definitions = [
    {
        "name": "get_income_data",
        "description": "Retrieve raw income statement data for a stock ticker",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol, like 'TSLA'"
                }
            },
            "required": ["ticker"]
        },
    }
]

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
        if function_name == "get_income_data":
            tool_output = get_income_data(arguments["ticker"])

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
