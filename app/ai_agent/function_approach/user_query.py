import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from functions import * # Your function implementations
import asyncio
import aiohttp

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
STOCKNEAR_API_KEY = os.getenv("STOCKNEAR_API_KEY")
CHAT_MODEL = 'gpt-4.1-mini-2025-04-14' #os.getenv("CHAT_MODEL")
API_URL = "http://localhost:8000"
instructions = 'Retrieve data by using functions based on the correct description. Answer only based on the data from the functions. Always interpret and validate user metrics, default to descending sort, fall back to the last-mentioned metric if unspecified, invoke the correct data functions, and verify results before returning.' #os.getenv("INSTRUCTIONS").replace("\\n", "\n")

# Dynamically gather function definitions and map names to callables
function_definitions = get_function_definitions()
function_map = {fn["name"]: globals()[fn["name"]] for fn in function_definitions}

# Initialize chat history
messages = [{"role": "system", "content": instructions}]


async def main():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})

        # Ask GPT which function to call, if any
        # This is the FIRST API call
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=[{"type": "function", "function": fn} for fn in function_definitions],
                tool_choice="auto"
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

        except Exception as e:
            print(f"API Error during initial call: {e}")
            messages.pop() # Remove user message if API call failed
            continue # Skip to next loop iteration

        if tool_calls:
            # Append the assistant message with tool_calls ONCE
            messages.append(message)

            # Process each tool call and collect results
            tool_outputs = []
            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    print(f"Error decoding function arguments for {name}")
                    # Handle invalid arguments, maybe append an error tool message
                    result_content = {"error": f"Invalid JSON arguments for function {name}"}
                    tool_outputs.append({
                        "tool_call_id": call.id,
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result_content)
                    })
                    continue # Skip to next tool call

                if name in function_map:
                    func = function_map[name]
                    try:
                        # Execute the function (await since functions are async)
                        result_content = await func(**args)
                    except Exception as e:
                        print(f"Error executing function {name}: {e}")
                        result_content = {"error": f"Function execution failed for {name}"}
                else:
                    print(f"Unknown function requested: {name}")
                    result_content = {"error": f"Unknown function: {name}"}

                # Store the result along with the tool_call_id to append later
                tool_outputs.append({
                    "tool_call_id": call.id,
                    "role": "tool",
                    "name": name, # The name is also needed for the tool message
                    "content": json.dumps(result_content)
                })

            # Append all tool output messages AFTER the assistant message
            messages.extend(tool_outputs) # Use extend to add all tool outputs at once

            # Now, send the updated messages list back to the API for the final response
            # This is the SECOND API call
            print("\nAssistant:", end=" ", flush=True)
            try:
                stream = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages, # messages now contains user, assistant (with tool_calls), and tool messages
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                print()
                # Note: The final assistant response from the stream is implicitly added
                # by the OpenAI library's handling of streaming responses when you
                # process the chunks. You typically don't need to manually append it
                # here if you are just printing. If you wanted to store the *full*
                # final message in the messages list, you'd accumulate content
                # and append a single message dictionary after the loop.

            except Exception as e:
                print(f"API Error during final response call: {e}")
                # Decide how to handle this error - maybe remove the tool messages
                # that were just added? For simplicity, we'll just print the error.


        else:
            # No function call: just stream the response
            # Append the assistant message before streaming
            messages.append(message)
            print("\nAssistant:", end=" ", flush=True)
            try:
                stream = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                print()
                 # Similar note as above regarding appending the streamed message

            except Exception as e:
                print(f"API Error during non-tool response call: {e}")
                # Handle error as needed

if __name__ == "__main__":
    asyncio.run(main())