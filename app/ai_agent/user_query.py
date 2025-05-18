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
CHAT_MODEL = os.getenv("CHAT_MODEL")
instructions = 'Retrieve data by using functions based on the correct description. Answer only based on the data from the functions. Always interpret and validate user metrics, default to descending sort, fall back to the last-mentioned metric if unspecified, invoke the correct data functions, and verify results before returning.'

# Dynamically gather function definitions and map names to callables
function_definitions = get_function_definitions()
function_map = {fn["name"]: globals()[fn["name"]] for fn in function_definitions}

# Keep the system instruction separate
system_message = {"role": "system", "content": instructions}


async def main():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # --- Start of Modification ---
        # Create a new messages list for the current turn
        # Only include the system instruction and the current user message
        current_turn_messages = [system_message, {"role": "user", "content": user_input}]
        # --- End of Modification ---


        # Ask GPT which function to call, if any
        # This is the FIRST API call, using only the current turn's context
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=current_turn_messages, # Use the current turn's messages
                tools=[{"type": "function", "function": fn} for fn in function_definitions],
                tool_choice="auto"
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

        except Exception as e:
            print(f"API Error during initial call: {e}")
            # No need to pop from current_turn_messages, it's rebuilt each loop
            continue # Skip to next loop iteration

        if tool_calls:
            # Append the assistant message with tool_calls to the current turn's messages
            current_turn_messages.append(message)

            # Process each tool call and collect results
            tool_outputs = []
            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    print(f"Error decoding function arguments for {name}")
                    result_content = {"error": f"Invalid JSON arguments for function {name}"}
                    tool_outputs.append({
                        "tool_call_id": call.id,
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result_content)
                    })
                    continue

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
                    "name": name,
                    "content": json.dumps(result_content)
                })

            # Append all tool output messages to the current turn's messages
            current_turn_messages.extend(tool_outputs)

            # Now, send the updated messages list (system, user, assistant tool_calls, tool_outputs)
            # back to the API for the final response. This is the SECOND API call.
            print("\nAssistant:", end=" ", flush=True)
            try:
                stream = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=current_turn_messages, # Use the current turn's messages including tool results
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                print()

            except Exception as e:
                print(f"API Error during final response call: {e}")

        else:
            # No function call: just stream the response
            # Append the assistant message to the current turn's messages
            current_turn_messages.append(message)
            print("\nAssistant:", end=" ", flush=True)
            try:
                stream = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=current_turn_messages, # Use the current turn's messages including assistant response
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                print()

            except Exception as e:
                print(f"API Error during non-tool response call: {e}")


if __name__ == "__main__":
    asyncio.run(main())