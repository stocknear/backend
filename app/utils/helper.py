from datetime import datetime, timedelta, time, date
import os
import orjson
import pytz
import math
import json
import re
import asyncio

def check_market_hours():

    holidays = ['2025-01-01', '2025-01-09','2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25']

    
    # Get the current date and time in ET (Eastern Time)
    et_timezone = pytz.timezone('America/New_York')
    current_time = datetime.now(et_timezone)
    current_date_str = current_time.strftime('%Y-%m-%d')
    current_hour = current_time.hour
    current_minute = current_time.minute
    current_day = current_time.weekday()  # Monday is 0, Sunday is 6

    # Check if the current date is a holiday or weekend
    is_weekend = current_day >= 5  # Saturday (5) or Sunday (6)
    is_holiday = current_date_str in holidays

    # Determine the market status
    if is_weekend or is_holiday:
        return False #"Market is closed."
    elif (current_hour == 16 and current_minute == 10) or 9 <= current_hour < 16:
        return True #"Market hours."
    else:
        return False #"Market is closed."


def load_latest_json(directory: str, find=True):
    """
    Load the JSON file corresponding to today's date (New York time) or the last Friday if today is a weekend.
    If `find` is True, try going back one day up to 10 times until a JSON file is found.
    If `find` is False, only check the current date (or adjusted Friday for weekends).
    """
    try:
        # Get today's date in New York timezone
        ny_tz = pytz.timezone("America/New_York")
        today_ny = datetime.now(ny_tz).date()

        # Adjust to Friday if today is Saturday or Sunday
        if today_ny.weekday() == 5:  # Saturday
            today_ny -= timedelta(days=1)
        elif today_ny.weekday() == 6:  # Sunday
            today_ny -= timedelta(days=2)

        attempts = 0

        # Loop to find the JSON file
        while True:
            # Construct the filename based on the adjusted date
            target_filename = f"{today_ny}.json"
            target_file_path = os.path.join(directory, target_filename)

            # Check if the file exists and load it
            if os.path.exists(target_file_path):
                with open(target_file_path, 'rb') as file:
                    print(f"JSON file found for date: {today_ny}")
                    return orjson.loads(file.read())

            # If find is False, only check the current date and exit
            if not find:
                print(f"No JSON file found for date: {today_ny}. Exiting as `find` is set to False.")
                break

            # Increment attempts and move to the previous day
            attempts += 1
            if attempts >= 10:
                print("No JSON file found after 10 attempts.")
                break
            today_ny -= timedelta(days=1)

        # Return an empty list if no file is found
        return []

    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

'''
def get_last_completed_quarter():
    today = datetime.today()
    year = today.year
    month = today.month
    # Calculate the current quarter (1 to 4)
    current_quarter = (month - 1) // 3 + 1

    # The previous quarter is the last completed quarter.
    # If we're in Q1, the previous quarter is Q4 of last year.
    if current_quarter == 1:
        return 4, year - 1
    else:
        return current_quarter - 1, year
'''

def get_last_completed_quarter():
    #return last two quarters ago
    today = datetime.today()
    year = today.year
    month = today.month
    # Calculate the current quarter (1 to 4)
    current_quarter = (month - 1) // 3 + 1

    # Determine the quarter that is two quarters ago.
    target_quarter = current_quarter - 2
    if target_quarter < 1:
        target_quarter += 4
        year -= 1

    return target_quarter, year



def replace_representative(office):
    replacements = {
        'Banks, James E. (Senator)': 'James Banks',
        'Banks, James (Senator)': 'James Banks',
        'James E Hon Banks': 'James Banks',
        'Knott, Brad (Senator)': 'Brad Knott',
        'Moody, Ashley B. (Senator)': 'Ashley Moody',
        'McCormick, David H. (Senator)': 'Dave McCormick',
        'McCormick, David H.': 'Dave McCormick',
        'Carper, Thomas R. (Senator)': 'Tom Carper',
        'Thomas R. Carper': 'Tom Carper',
        'Tuberville, Tommy (Senator)': 'Tommy Tuberville',
        'Ricketts, Pete (Senator)': 'John Ricketts',
        'Pete Ricketts': 'John Ricketts',
        'Moran, Jerry (Senator)': 'Jerry Moran',
        'Fischer, Deb (Senator)': 'Deb Fischer',
        'Mullin, Markwayne (Senator)': 'Markwayne Mullin',
        'Whitehouse, Sheldon (Senator)': 'Sheldon Whitehouse',
        'Toomey, Pat (Senator)': 'Pat Toomey',
        'Sullivan, Dan (Senator)': 'Dan Sullivan',
        'Capito, Shelley Moore (Senator)': 'Shelley Moore Capito',
        'Roberts, Pat (Senator)': 'Pat Roberts',
        'King, Angus (Senator)': 'Angus King',
        'Hoeven, John (Senator)': 'John Hoeven',
        'Duckworth, Tammy (Senator)': 'Tammy Duckworth',
        'Perdue, David (Senator)': 'David Perdue',
        'Inhofe, James M. (Senator)': 'James M. Inhofe',
        'Murray, Patty (Senator)': 'Patty Murray',
        'Boozman, John (Senator)': 'John Boozman',
        'Loeffler, Kelly (Senator)': 'Kelly Loeffler',
        'Reed, John F. (Senator)': 'John F. Reed',
        'Collins, Susan M. (Senator)': 'Susan M. Collins',
        'Cassidy, Bill (Senator)': 'Bill Cassidy',
        'Wyden, Ron (Senator)': 'Ron Wyden',
        'Hickenlooper, John (Senator)': 'John Hickenlooper',
        'Booker, Cory (Senator)': 'Cory Booker',
        'Donald Beyer, (Senator).': 'Donald Sternoff Beyer',
        'Peters, Gary (Senator)': 'Gary Peters',
        'Donald Sternoff Beyer, (Senator).': 'Donald Sternoff Beyer',
        'Donald S. Beyer, Jr.': 'Donald Sternoff Beyer',
        'Donald Sternoff Honorable Beyer': 'Donald Sternoff Beyer',
        'K. Michael Conaway': 'Michael Conaway',
        'C. Scott Franklin': 'Scott Franklin',
        'Scott Scott Franklin': 'Scott Franklin',
        'Robert C. "Bobby" Scott': 'Bobby Scott',
        'Kelly Louise Morrison': 'Kelly Morrison',
        'Madison Cawthorn': 'David Madison Cawthorn',
        'Cruz, Ted (Senator)': 'Ted Cruz',
        'Smith, Tina (Senator)': 'Tina Smith',
        'Graham, Lindsey (Senator)': 'Lindsey Graham',
        'Hagerty, Bill (Senator)': 'Bill Hagerty',
        'Scott, Rick (Senator)': 'Rick Scott',
        'Warner, Mark (Senator)': 'Mark Warner',
        'McConnell, A. Mitchell Jr. (Senator)': 'Mitch McConnell',
        'Mitchell McConnell': 'Mitch McConnell',
        'Charles J. "Chuck" Fleischmann': 'Chuck Fleischmann',
        'Vance, J.D. (Senator)': 'James Vance',
        'Neal Patrick MD, Facs Dunn': 'Neal Dunn',
        'Neal Patrick MD, Facs Dunn (Senator)': 'Neal Dunn',
        'Neal Patrick Dunn, MD, FACS': 'Neal Dunn',
        'Neal P. Dunn': 'Neal Dunn',
        'Tillis, Thom (Senator)': 'Thom Tillis',
        'W. Gregory Steube': 'Greg Steube',
        'W. Grego Steube': 'Greg Steube',
        'W. Greg Steube': 'Greg Steube',
        'David David Madison Cawthorn': 'David Madison Cawthorn',
        'Blunt, Roy (Senator)': 'Roy Blunt',
        'Thune, John (Senator)': 'John Thune',
        'Rosen, Jacky (Senator)': 'Jacky Rosen',
        'Britt, Katie (Senator)': 'Katie Britt',
        'Britt, Katie': 'Katie Britt',
        'James Costa': 'Jim Costa',
        'Lummis, Cynthia (Senator)': 'Cynthia Lummis',
        'Coons, Chris (Senator)': 'Chris Coons',
        'Udall, Tom (Senator)': 'Tom Udall',
        'Kennedy, John (Senator)': 'John Kennedy',
        'Bennet, Michael (Senator)': 'Michael Bennet',
        'Casey, Robert P. Jr. (Senator)': 'Robert Casey',
        'Van Hollen, Chris (Senator)': 'Chris Van Hollen',
        'Manchin, Joe (Senator)': 'Joe Manchin',
        'Cornyn, John (Senator)': 'John Cornyn',
        'Enzy, Michael (Senator)': 'Michael Enzy',
        'Cardin, Benjamin (Senator)': 'Benjamin Cardin',
        'Kaine, Tim (Senator)': 'Tim Kaine',
        'Joseph P. Kennedy III': 'Joe Kennedy',
        'James E Hon Banks': 'Jim Banks',
        'Michael F. Q. San Nicolas': 'Michael San Nicolas',
        'Barbara J Honorable Comstock': 'Barbara Comstock',
        'Darin McKay LaHood': 'Darin LaHood',
        'Harold Dallas Rogers': 'Hal Rogers',
        'April McClain Delaney': 'April Delaney',
        'Mr ': '',
        'Mr. ': '',
        'Dr ': '',
        'Dr. ': '',
        'Mrs ': '',
        'Mrs. ': '',
        '(Senator)': '',
    }

    for old, new in replacements.items():
        office = office.replace(old, new)
        office = ' '.join(office.split())
    return office


def compute_option_return(option: dict, current_price: float) -> float:
   
    try:
        # --- Parse and validate basic fields ---
        pc = option.get("put_call")

    
        strike = float(option["strike_price"])
        sentiment = option.get("sentiment")
        if sentiment is None:
            return None
        sentiment = str(sentiment).strip().capitalize()

        # Determine long/short from sentiment
        if pc == "Calls":
            is_long = sentiment in ("Bullish", "Neutral")
        else:  # PUT
            is_long = sentiment in ("Bearish", "Neutral")

        # --- Cost basis ---
        # If provided, use it; else calculate
        cost_basis = option.get("cost_basis")
        size = option.get('size',0)

        multiplier = 100

        intrinsic = 0.0
        if pc == "Calls":
            intrinsic = max(current_price - strike, 0.0)
        else:
            intrinsic = max(strike - current_price, 0.0)

        current_premium = intrinsic

        # --- Mark-to-market P/L ---
        current_value = current_premium * size * multiplier

        if is_long:
            profit = current_value - cost_basis
        else:
            profit = cost_basis - current_value

        pct_return = (profit / cost_basis) * 100.0

        if not math.isfinite(pct_return):
            return None

        return round(pct_return, 2)

    except Exception:
        return None

def json_to_string(json_data):
    try:
        # Use json.dumps() for a more robust and readable conversion
        formatted_string = json.dumps(json_data, indent=4)  # Indent for better readability
        return formatted_string
    except TypeError as e:
        return f"Error: Invalid JSON data.  Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"



# --- Configuration for Trigger Phrases ---
TRIGGER_CONFIG = {
    "@Analyst": {
        "description": "Handles analyst-related queries by forcing specific financial tool calls.",
        "parameter_extraction": {
            "prompt_template": "First identify the stock ticker symbols mentioned in the user's query: '{query}'. If no specific tickers are mentioned, identify which companies the user is likely interested in and determine their ticker symbols. Return ONLY the ticker symbols as a comma-separated list without any explanation or additional text. Example response format: 'AAPL,MSFT,GOOG'",
            "regex_pattern": r'\$?([A-Z]{1,5})\b', # Regex to find ticker symbols
            "default_value": ["AAPL"], # Default if no tickers found
            "param_name": "tickers_list"  # Key for storing extracted tickers
        },
        "perform_initial_llm_call": True,
        "pre_forced_tools_assistant_message_template": "Let me check the analyst information for {params}.",
        "forced_tool_calls": [
            {
                "id_template": "afunc1",
                "function_name": "get_analyst_estimate",
                # Maps function argument "tickers" to the extracted "tickers_list"
                "arguments_mapping": {"tickers": "tickers_list"}
            },
            {
                "id_template": "afunc2",
                "function_name": "get_analyst_ratings",
                "arguments_mapping": {"tickers": "tickers_list"}
            }
        ],
    },
    "@OptionsFlow": {
        "description": "Handles options flow order related queries by forcing specific financial tool calls.",
        "parameter_extraction": {
            "prompt_template": "First identify the stock ticker symbols mentioned in the user's query: '{query}'. If no specific tickers are mentioned, identify which companies the user is likely interested in and determine their ticker symbols. Return ONLY the ticker symbols as a comma-separated list without any explanation or additional text. Example response format: 'AAPL,MSFT,GOOG'",
            "regex_pattern": r'\$?([A-Z]{1,5})\b',
            "default_value": ["AAPL"],
            "param_name": "tickers_list"
        },
        "perform_initial_llm_call": True,
        "pre_forced_tools_assistant_message_template": "Let me check the latest options flow orders information for {params}.",
        "forced_tool_calls": [
            {
                "id_template": "ofeed",
                "function_name": "get_latest_options_flow_feed",
                "arguments_mapping": {"tickers": "tickers_list"}
            },
        ],
    },
    "@DarkPoolFlow": {
        "description": "Handles dark pool flow order related queries by forcing specific financial tool calls.",
        "parameter_extraction": {
            "prompt_template": "First identify the stock ticker symbols mentioned in the user's query: '{query}'. If no specific tickers are mentioned, identify which companies the user is likely interested in and determine their ticker symbols. Return ONLY the ticker symbols as a comma-separated list without any explanation or additional text. Example response format: 'AAPL,MSFT,GOOG'",
            "regex_pattern": r'\$?([A-Z]{1,5})\b',
            "default_value": ["AAPL"],
            "param_name": "tickers_list"
        },
        "perform_initial_llm_call": True,
        "pre_forced_tools_assistant_message_template": "Let me check the latest dark pool flow orders information for {params}.",
        "forced_tool_calls": [
            {
                "id_template": "dark_pool_feed",
                "function_name": "get_latest_dark_pool_feed",
                "arguments_mapping": {"tickers": "tickers_list"}
            },
        ],
    },
    "@News": {
        "description": "Handles news-related queries by forcing specific financial tool calls.",
        "parameter_extraction": {
            "prompt_template": "First identify the stock ticker symbols mentioned in the user's query: '{query}'. If no specific tickers are mentioned, identify which companies the user is likely interested in and determine their ticker symbols. Return ONLY the ticker symbols as a comma-separated list without any explanation or additional text. Example response format: 'AAPL,MSFT,GOOG'",
            "regex_pattern": r'\$?([A-Z]{1,5})\b', # Regex to find ticker symbols
            "default_value": ["AAPL"], # Default if no tickers found
            "param_name": "tickers_list"  # Key for storing extracted tickers
        },
        "perform_initial_llm_call": True,
        "pre_forced_tools_assistant_message_template": "Let me check the latest news information for {params}.",
        "forced_tool_calls": [
            {
                "id_template": "wiim",
                "function_name": "get_why_priced_moved",
                "arguments_mapping": {"tickers": "tickers_list"}
            },
            {
                "id_template": "marketNews",
                "function_name": "get_market_news",
                "arguments_mapping": {"tickers": "tickers_list"}
            }
        ],
    },
    # Add more trigger configurations here
}

# --- Helper Functions ---


async def _extract_parameters(user_query, extraction_config, async_client, model, max_tokens, semaphore, context_messages):
    """Extracts parameters based on the provided configuration."""
    extracted_values = []
    param_name = extraction_config["param_name"]

    # 1. LLM-based extraction
    if extraction_config.get("prompt_template"):
        extraction_prompt_content = extraction_config["prompt_template"].format(query=user_query)
        
        # Use a copy of context messages, add the specific extraction system prompt
        llm_extraction_messages = context_messages.copy()
        llm_extraction_messages.append({"role": "system", "content": extraction_prompt_content})
        
        async with semaphore:
            response = await async_client.chat.completions.create(
                model=model,
                messages=llm_extraction_messages,
                max_tokens=max_tokens
            )
        params_str = response.choices[0].message.content.strip()
        if params_str:
            extracted_values = [p.strip() for p in params_str.split(',') if p.strip()]

    # 2. Regex fallback (if LLM failed or not configured for LLM)
    if not extracted_values and extraction_config.get("regex_pattern"):
        matches = re.findall(extraction_config["regex_pattern"], user_query)
        if matches:
            # Handle cases where findall returns list of strings or list of tuples (from capture groups)
            if isinstance(matches[0], tuple):
                extracted_values = [m[0].strip() for m in matches if m[0].strip()]
            else:
                extracted_values = [m.strip() for m in matches if m.strip()]
    
    # 3. Default fallback
    if not extracted_values and extraction_config.get("default_value"):
        extracted_values = extraction_config["default_value"]

    return {param_name: extracted_values}


async def _execute_and_append_tool_calls(tool_calls_to_process, messages, function_map):
    """
    Executes a list of tool calls (either from LLM or forced) and appends results to messages.
    tool_calls_to_process can be a list of OpenAI tool_call objects or dicts for forced calls.
    """
    
    async def execute_single_tool(call_info):
        is_forced_call_dict = isinstance(call_info, dict)

        tool_call_id = call_info.id if not is_forced_call_dict else call_info["id"]
        fn_name = call_info.function.name if not is_forced_call_dict else call_info["function"]["name"]
        arguments_str = call_info.function.arguments if not is_forced_call_dict else call_info["function"]["arguments"]
        
        result_content_payload = {}
        try:
            args = json.loads(arguments_str)
            if fn_name in function_map:
                # Call the actual function
                result_content_payload = await function_map[fn_name](**args)
            else:
                print(f"Error: Unknown function {fn_name} requested by tool call.")
                result_content_payload = {"error": f"Unknown function {fn_name}"}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON arguments for {fn_name}: {str(e)}. Arguments: '{arguments_str}'")
            result_content_payload = {"error": f"Invalid arguments format for {fn_name}: {str(e)}"}
        except Exception as e:
            print(f"Error executing function {fn_name}: {str(e)}")
            result_content_payload = {"error": str(e)}

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": fn_name,
            "content": json.dumps(result_content_payload) # Ensure content is JSON string
        }

    tasks = [execute_single_tool(tc) for tc in tool_calls_to_process]
    tool_results_messages = await asyncio.gather(*tasks)

    for res_msg in tool_results_messages:
        messages.append(res_msg)


async def _handle_configured_case(data, base_messages, config, user_query, 
                                  async_client, chat_model, max_tokens, semaphore, 
                                  function_map, system_message, tools_payload):
    """Handles request processing for a specifically configured trigger case."""
    messages = base_messages.copy() # Start with system_message + history + current_user_query
    extracted_data = {}

    # 1. Parameter Extraction
    if "parameter_extraction" in config:
        # For parameter extraction, we might not need full history, just user query and a system prompt.
        # Or, use a minimal context. The original Analyst code used messages.copy() for ticker_extraction_messages.
        # Let's pass the current `messages` as context for extraction LLM.
        context_for_extraction = messages.copy() # Contains system_msg + history + user_query
        extracted_data = await _extract_parameters(
            user_query,
            config["parameter_extraction"],
            async_client,
            chat_model,
            max_tokens,
            semaphore,
            context_for_extraction
        )
        print(f"Extracted parameters for '{config['description']}': {extracted_data}")
        # Ensure default is applied if extraction yields nothing and default exists
        param_conf = config["parameter_extraction"]
        if not extracted_data.get(param_conf["param_name"]) and param_conf.get("default_value"):
           extracted_data[param_conf["param_name"]] = param_conf["default_value"]


    # 2. Optional: Initial LLM Call (before forced tools)
    # This mimics the original @Analyst behavior where an initial response is generated.
    if config.get("perform_initial_llm_call", False):
        async with semaphore:
            # This call uses the main message history
            initial_response = await async_client.chat.completions.create(
                model=chat_model,
                messages=messages, # These are system_msg + history + user_query
                max_tokens=max_tokens,
                # Decide if this initial call can also use tools or not.
                # Original Analyst flow didn't seem to process tools from this specific call.
                # tools=tools_payload if config.get("initial_call_can_use_tools") else None,
                # tool_choice="auto" if config.get("initial_call_can_use_tools") else None
            )
        assistant_msg_before_forced = initial_response.choices[0].message
        # Convert to dict if it's an OpenAI object and your messages list expects dicts
        # For simplicity, assuming it can be appended directly or is converted by your framework.
        messages.append(assistant_msg_before_forced)
        # If assistant_msg_before_forced has tool_calls, they would be handled by the standard loop if we returned here.
        # However, the purpose of a forced_tool_call config is usually to override/direct.

    # 3. Prepare and Execute Forced Tool Calls
    if "forced_tool_calls" in config and config["forced_tool_calls"]:
        forced_tool_call_objects_for_api = []
        
        # Get the primary parameter value list (e.g., list of tickers or locations)
        # This is used for the pre_forced_tools_assistant_message_template's {params}
        param_values_for_template = []
        if "parameter_extraction" in config:
            param_values_for_template = extracted_data.get(config["parameter_extraction"]["param_name"], [])

        for i, tool_conf in enumerate(config["forced_tool_calls"]):
            function_args = {}
            if "arguments_mapping" in tool_conf:
                for func_arg_name, source_param_key in tool_conf["arguments_mapping"].items():
                    if source_param_key in extracted_data:
                        function_args[func_arg_name] = extracted_data[source_param_key]
                    else:
                        print(f"Warning: Source parameter '{source_param_key}' not found in extracted_data for function '{tool_conf['function_name']}'. Using empty list as fallback.")
                        function_args[func_arg_name] = [] # Sensible fallback, might need to be configurable
            
            forced_tool_call_objects_for_api.append({
                "id": tool_conf["id_template"].format(index=i), # Simpler ID, ensure uniqueness if needed
                "type": "function",
                "function": {
                    "name": tool_conf["function_name"],
                    "arguments": json.dumps(function_args)
                }
            })

        if forced_tool_call_objects_for_api:
            # Create a synthetic assistant message to carry these forced tool calls
            assistant_content = "Proceeding with required actions..." # Default content
            if config.get("pre_forced_tools_assistant_message_template"):
                assistant_content = config["pre_forced_tools_assistant_message_template"].format(
                    params=", ".join(map(str, param_values_for_template)) if param_values_for_template else "the relevant items"
                )
            
            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": forced_tool_call_objects_for_api
            })
            
            # Execute these forced tools
            await _execute_and_append_tool_calls(forced_tool_call_objects_for_api, messages, function_map)

    # 4. Get Final Response from LLM
    # This call happens after parameter extraction, optional initial call, and any forced tool calls + results.
    final_llm_messages = messages.copy()
    
    async with semaphore:
        final_response = await async_client.chat.completions.create(
            model=chat_model,
            messages=final_llm_messages,
            max_tokens=max_tokens
            # Typically, no 'tools' or 'tool_choice' here, as this is the concluding text response.
        )
    final_assistant_msg = final_response.choices[0].message
    messages.append(final_assistant_msg) # Append the final message object

    return messages


async def process_request(data, async_client, function_map, request_semaphore, system_message, CHAT_MODEL, MAX_TOKENS, tools_payload):
    user_query = data.query.lower()
    current_messages_history = list(data.messages) 

    prepared_initial_messages = [system_message] + current_messages_history + [{"role": "user", "content": user_query}]

    active_config = None
    trigger_phrase_found = None

    for trigger, config_item in TRIGGER_CONFIG.items():
        if trigger.lower() in user_query:
            active_config = config_item
            trigger_phrase_found = trigger
            print(f"Detected trigger: {trigger_phrase_found}")
            break
            
    try:
        if active_config:
            # Handle request using the specific configuration for the detected trigger
            return await _handle_configured_case(
                data, prepared_initial_messages, active_config, user_query,
                async_client, CHAT_MODEL, MAX_TOKENS, request_semaphore,
                function_map, system_message, tools_payload
            )
        else:
            # Standard flow: LLM decides on tool usage based on general prompt and available tools
            messages = prepared_initial_messages.copy()

            async with request_semaphore:
                response = await async_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    tools=tools_payload,
                    tool_choice="auto" if tools_payload else "none"
                )
            
            assistant_msg = response.choices[0].message
            messages.append(assistant_msg) # Append OpenAI message object

            # Loop to handle multiple rounds of tool calls if initiated by the LLM
            while hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
                await _execute_and_append_tool_calls(assistant_msg.tool_calls, messages, function_map)
                
                # Get model response after processing tool calls
                async with request_semaphore:
                    followup_response = await async_client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        tools=tools_payload, # Allow further tool use
                        tool_choice="auto" if tools_payload else "none" 
                    )
                assistant_msg = followup_response.choices[0].message
                messages.append(assistant_msg)
            
            return messages

    except Exception as e:
        print(f"Request processing failed: {e}")
        raise
        
