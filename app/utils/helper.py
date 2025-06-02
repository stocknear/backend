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

def load_congress_db():
    data = {}
    directory = "./json/congress-trading/politician-db/"
    
    try:
        files = os.listdir(directory)
        json_files = [f for f in files if f.endswith('.json')]
        
        for filename in json_files:
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "rb") as file:
                    file_data = orjson.loads(file.read())
                    
                    if 'history' in file_data and len(file_data['history']) > 0:
                        politician_id = file_data['history'][0]['id']
                        name = file_data['history'][0]['office']
                        data[name] = politician_id
                        
            except (KeyError, IndexError, orjson.JSONDecodeError) as e:
                print(f"Error processing {filename}: {e}")
                continue
                
    except FileNotFoundError:
        print(f"Directory {directory} not found")
    return data


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


key_congress_db = load_congress_db()

# --- Enhanced Configuration for Trigger Phrases ---
TRIGGER_CONFIG = {
    "@Analyst": {
        "description": "Handles analyst-related queries by forcing specific financial tool calls.",
        "parameter_extraction": {
            "prompt_template": "First identify the stock ticker symbols mentioned in the user's query: '{query}'. If no specific tickers are mentioned, identify which companies the user is likely interested in and determine their ticker symbols. Return ONLY the ticker symbols as a comma-separated list without any explanation or additional text. Example response format: 'AAPL,MSFT,GOOG'",
            "regex_pattern": r'\$?([A-Z]{1,5})\b',
            "default_value": ["AAPL"],
            "param_name": "ticker_list"
        },
        "perform_initial_llm_call": True,
        "pre_forced_tools_assistant_message_template": "Let me check the analyst information for {params}.",
        "forced_tool_calls": [
            {
                "id_template": "afunc1_{index}",
                "function_name": "get_ticker_analyst_estimate",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True  # Ensures this function MUST be called
            },
            {
                "id_template": "afunc2_{index}",
                "function_name": "get_ticker_analyst_rating",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            }
        ],
        "validate_all_calls_executed": True,  # New flag to ensure all functions are called
    },
    "@DarkPoolData": {
        "description": "Retrieves all the dark pool data for the company to find the sentiment.",
        "parameter_extraction": {
            "prompt_template": "Identify the stock ticker symbols mentioned in the user's query: '{query}'. If no explicit symbols are provided, infer which companies the user is likely referring to and return their ticker symbols. Output only the symbols as a comma-separated list with no additional text. Example: 'AAPL,MSFT,GOOG'.",
            "regex_pattern": "\\$?([A-Z]{1,5})\\b",
            "default_value": ["AAPL"],
            "param_name": "ticker_list"
        },
        "perform_initial_llm_call": True,
        "pre_forced_tools_assistant_message_template": "Using the provided data, write an overview of each dark pool data section. Provide detailed insights into recent dark pool activity, presenting clear bullish and bearish interpretations. Conclude with a concise summary and a definitive investment signal—Bullish, Neutral, or Bearish. Do not include notes or disclaimers.",
        "forced_tool_calls": [
            {
                "id_template": "dp_latest_flow",
                "function_name": "get_latest_dark_pool_feed",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "dp_overview",
                "function_name": "get_ticker_dark_pool",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "dp_ticker_quote",
                "function_name": "get_ticker_quote",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
        ],
        "validate_all_calls_executed": True,
    },
    "@OptionsData": {
        "description": "Retrieves all the options flow data for the company to find the sentiment.",
        "parameter_extraction": {
            "prompt_template": "Identify the stock ticker symbols mentioned in the user's query: '{query}'. If no explicit symbols are provided, infer which companies the user is likely referring to and return their ticker symbols. Output only the symbols as a comma-separated list with no additional text. Example: 'AAPL,MSFT,GOOG'.",
            "regex_pattern": "\\$?([A-Z]{1,5})\\b",
            "default_value": ["AAPL"],
            "param_name": "ticker_list"
        },
        "perform_initial_llm_call": True,
        "pre_forced_tools_assistant_message_template": "Using the provided data, write an overview of each options flow data section. Provide detailed insights into recent options activity, presenting clear bullish and bearish interpretations. Conclude with a concise summary and a definitive investment signal—Bullish, Neutral, or Bearish. Do not include notes or disclaimers.",
        "forced_tool_calls": [
            {
                "id_template": "of_latest_flow",
                "function_name": "get_latest_options_flow_feed",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "of_overview",
                "function_name": "get_ticker_options_data",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "of_max_pain",
                "function_name": "get_ticker_max_pain",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "of_oi_by_strike_expiry",
                "function_name": "get_ticker_open_interest_by_strike_and_expiry",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "of_hottest_options_contracts",
                "function_name": "get_ticker_hottest_options_contracts",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "of_ticker_quote",
                "function_name": "get_ticker_quote",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
        ],
        "validate_all_calls_executed": True,
    },
    "@BullvsBear": {
        "description": "Retrieves all the data to decide to create a bull case and bear case for the company.",
        "parameter_extraction": {
            "prompt_template": "Identify the stock ticker symbols mentioned in the user's query: '{query}'. If no explicit symbols are provided, infer which companies the user is likely referring to and return their ticker symbols. Output only the symbols as a comma-separated list with no additional text. Example: 'AAPL,MSFT,GOOG'.",
            "regex_pattern": "\\$?([A-Z]{1,5})\\b",
            "default_value": ["AAPL"],
            "param_name": "ticker_list"
        },
        "perform_initial_llm_call": True,
        "pre_forced_tools_assistant_message_template": "Using the provided data, write a short overview what the company does in 1-2 sentences. Construct both a bull case and a bear case for the company, clearly presenting arguments for each perspective. Conclude with a concise summary that includes a definitive investment signal—Buy, Hold, or Sell. Additionally, provide a 12-month price forecast with low, median, and high price targets, including the percentage upside or downside from the current stock price for each scenario. Don't add any Notes or disclaimers.",
        "forced_tool_calls": [
            {
                "id_template": "bvb_bvb",
                "function_name": "get_ticker_bull_vs_bear",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "bvb_wiim",
                "function_name": "get_why_priced_moved",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "bvb_analyst_estimate",
                "function_name": "get_ticker_analyst_estimate",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "bvb_analyst_rating",
                "function_name": "get_ticker_analyst_rating",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
        ],
        "validate_all_calls_executed": True,
    },
    "@CompareStocks": {
        "description": "Retrieves all the data to decide to compare 2 or more companies which one is better",
        "parameter_extraction": {
            "prompt_template": "Identify the stock ticker symbols mentioned in the user's query: '{query}'. If no explicit symbols are provided, infer which companies the user is likely referring to and return their ticker symbols. Output only the symbols as a comma-separated list with no additional text. Example: 'AAPL,MSFT,GOOG'.",
            "regex_pattern": "\\$?([A-Z]{1,5})\\b",
            "default_value": ["AAPL"],
            "param_name": "ticker_list"
        },
        "perform_initial_llm_call": True,
        "pre_forced_tools_assistant_message_template": "Analyze the provided data across all sections—fundamental metrics, statistical indicators, news sentiment, options flow, and any other relevant categories. For each section, provide a structured and conclusive comparison between the companies. Clearly state which company demonstrates stronger performance or outlook in that section. Support your analysis with detailed, data-driven insights, and include both bullish and bearish interpretations where applicable. Ensure the response is insightful, actionable, and free from disclaimers or editorial notes.",
        "forced_tool_calls": [
            {
                "id_template": "compare_wiim",
                "function_name": "get_why_priced_moved",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "compare_company_data",
                "function_name": "get_company_data",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "compare_ticker_news",
                "function_name": "get_ticker_news",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "compare_business_metrics",
                "function_name": "get_ticker_business_metrics",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "compare_analyst_estimate",
                "function_name": "get_ticker_analyst_estimate",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "compare_analyst_rating",
                "function_name": "get_ticker_analyst_rating",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "compare_ticker_statistics",
                "function_name": "get_ticker_statistics",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "compare_ticker_quote",
                "function_name": "get_ticker_quote",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "compare_ticker_shareholders",
                "function_name": "get_ticker_shareholders",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "compare_ticker_insider_trading",
                "function_name": "get_ticker_insider_trading",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
        ],
        "validate_all_calls_executed": True,
    },
    "@FundamentalData": {
        "description": "Retrieves all the fundamental data.",
        "parameter_extraction": {
            "prompt_template": "Identify the stock ticker symbols mentioned in the user's query: '{query}'. If no explicit symbols are provided, infer which companies the user is likely referring to and return their ticker symbols. Output only the symbols as a comma-separated list with no additional text. Example: 'AAPL,MSFT,GOOG'.",
            "regex_pattern": "\\$?([A-Z]{1,5})\\b",
            "default_value": ["AAPL"],
            "param_name": "ticker_list"
        },
        "perform_initial_llm_call": True,
        "pre_forced_tools_assistant_message_template": "Conduct a comprehensive value investor analysis of the provided financial data across all key sections, including fundamental metrics, the income statement, balance sheet, cash flow statement, and financial ratios. For each section, deliver a well-structured and conclusive assessment, clearly determining which company demonstrates stronger performance or a more favorable outlook based on the data. Support your evaluation with specific, data-driven insights, incorporating both bullish and bearish interpretations to reflect the full scope of each company's financial health. Your response should be insightful and actionable, focusing on key indicators such as profitability, sustainability, efficiency, and risk. In the end create a summary with a clear Bullish, Neutral or Bearish Signal. Avoid disclaimers, editorial remarks, or vague commentary—ensure that the analysis is objective, comparative, and aimed at empowering confident decision-making.",
        "forced_tool_calls": [
            {
                "id_template": "fund_ticker_income",
                "function_name": "get_ticker_income_statement",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "fund_ticker_balance_sheet",
                "function_name": "get_ticker_balance_sheet_statement",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "fund_ticker_cash_flow",
                "function_name": "get_ticker_cash_flow_statement",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "fund_ticker_ratios",
                "function_name": "get_ticker_ratios_statement",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "fund_ticker_key_metrics",
                "function_name": "get_ticker_key_metrics",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
            {
                "id_template": "fund_ticker_owner_earnings",
                "function_name": "get_ticker_owner_earnings",
                "arguments_mapping": {"tickers": "ticker_list"},
                "required": True
            },
        
        ],
        "validate_all_calls_executed": True,
    },
    "@Plot": {
        "description": "Analyzes the tickers",
        "parameter_extraction": {
            "prompt_template": "Identify the stock ticker symbols mentioned in the user's query: '{query}'. If no explicit symbols are provided, infer which companies the user is likely referring to and return their ticker symbols. Output only the symbols as a comma-separated list with no additional text. Example: 'AAPL,MSFT,GOOG'.",
            "regex_pattern": "\\$?([A-Z]{1,5})\\b",
            "default_value": [],
            "param_name": "ticker_list"
        },
        "perform_initial_llm_call": True,
        "validate_all_calls_executed": False,
    },
}


UI_COMPONENT_CONFIG = {
    "@plot": {
        "config_param_path": ["parameter_extraction", "param_name"],  # Path to parameter name in config
        "content_template": "Here is the plot for  {tickers}",  # Template for success message
        "no_data_content": "No stocks data available to plot",  # Template for empty data
        "component_name": "plot",  # Component key in callComponent
        "data_key": "tickerList"  # Data key in callComponent
    }
    # Add new components here like:
}

html_formatting_system_prompt = {
    "role": "system",
    "content": (
        "Format your entire response strictly using this HTML structure:\n\n"
        "<h3 class=\"text-lg font-semibold mt-4 mb-2\">Section Title</h3>\n"
        "<h4 class=\"text-md font-medium mt-3 mb-2\">Subsection Title</h4>\n"
        "<ul class=\"list-disc pl-5 space-y-2 mb-4\">\n"
        "  <li><strong>Label:</strong> Value</li>\n"
        "</ul>\n"
        "<p class=\"mt-2 mb-4\">Your text here.</p>\n"
        "**DO NOT** use numbered lists or Markdown bullets. Each metric or value must be formatted as a separate bullet or div. Convert all raw assistant replies into this HTML format. "
        "Ensure your entire output adheres to this HTML structure. If the user asks for something that doesn't fit well (e.g. 'hello'), "
        "still try to use a <p> tag or appropriate HTML. If you are presenting data from tools, format that data using the specified HTML."
    )
} 

# --- Enhanced Helper Functions ---
class ForcedToolCallExecutionError(Exception):
    """Raised when forced tool calls fail to execute properly."""
    pass


async def _extract_parameters(user_query, extraction_config, async_client, model, max_tokens, semaphore, context_messages):
    """Extracts parameters based on the provided configuration."""
    extracted_values = []
    param_name = extraction_config["param_name"]

    # 1. LLM-based extraction
    if extraction_config.get("prompt_template"):
        extraction_prompt_content = extraction_config["prompt_template"].format(query=user_query)
        
        llm_extraction_messages = context_messages.copy()
        llm_extraction_messages.append({"role": "system", "content": extraction_prompt_content})
        
        async with semaphore:
            response = await async_client.chat.completions.create(
                model=model,
                messages=llm_extraction_messages,
                max_tokens=30,
                temperature=0
            )
        params_str = response.choices[0].message.content.strip()
        if params_str:
            extracted_values = [p.strip() for p in params_str.split(',') if p.strip()]

    # 2. Regex fallback
    if not extracted_values and extraction_config.get("regex_pattern"):
        matches = re.findall(extraction_config["regex_pattern"], user_query)
        if matches:
            if isinstance(matches[0], tuple):
                extracted_values = [m[0].strip() for m in matches if m[0].strip()]
            else:
                extracted_values = [m.strip() for m in matches if m.strip()]
    
    # 3. Default fallback
    if not extracted_values and extraction_config.get("default_value"):
        extracted_values = extraction_config["default_value"]

    return {param_name: extracted_values}


async def _execute_and_append_tool_calls(tool_calls_to_process, messages, function_map, validate_execution=False):
    """
    Executes a list of tool calls and appends results to messages.
    
    Args:
        tool_calls_to_process: List of tool calls to execute
        messages: Message list to append results to
        function_map: Available functions
        validate_execution: If True, raises exception if any tool call fails
    """
    execution_results = {"successful": [], "failed": []}
    
    async def execute_single_tool(call_info):
        is_forced_call_dict = isinstance(call_info, dict)
        tool_call_id = call_info.id if not is_forced_call_dict else call_info["id"]
        fn_name = call_info.function.name if not is_forced_call_dict else call_info["function"]["name"]
        arguments_str = call_info.function.arguments if not is_forced_call_dict else call_info["function"]["arguments"]
        
        result_content_payload = {}
        success = False
        
        try:
            args = json.loads(arguments_str)
            if fn_name in function_map:
                result_content_payload = await function_map[fn_name](**args)
                success = True
                print(f"✓ Successfully executed function: {fn_name}")
            else:
                error_msg = f"Unknown function {fn_name} requested by tool call."
                print(f"✗ Error: {error_msg}")
                result_content_payload = {"error": error_msg}
        except json.JSONDecodeError as e:
            error_msg = f"Invalid arguments format for {fn_name}: {str(e)}"
            print(f"✗ JSON Error: {error_msg}. Arguments: '{arguments_str}'")
            result_content_payload = {"error": error_msg}
        except Exception as e:
            error_msg = f"Error executing function {fn_name}: {str(e)}"
            print(f"✗ Execution Error: {error_msg}")
            result_content_payload = {"error": error_msg}

        tool_result = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": fn_name,
            "content": json.dumps(result_content_payload)
        }
        
        if success:
            execution_results["successful"].append(fn_name)
        else:
            execution_results["failed"].append(fn_name)
            
        return tool_result

    # Execute all tool calls concurrently
    tasks = [execute_single_tool(tc) for tc in tool_calls_to_process]
    tool_results_messages = await asyncio.gather(*tasks)

    # Append all results to messages
    for res_msg in tool_results_messages:
        messages.append(res_msg)
    
    # Validate execution if required
    if validate_execution and execution_results["failed"]:
        failed_functions = ", ".join(execution_results["failed"])
        raise ForcedToolCallExecutionError(
            f"Failed to execute required functions: {failed_functions}"
        )
    
    print(f"Tool execution summary - Successful: {len(execution_results['successful'])}, "
          f"Failed: {len(execution_results['failed'])}")
    
    return execution_results


async def _validate_forced_tool_calls_completion(config, execution_results):
    """
    Validates that all required forced tool calls were executed successfully.
    """
    if not config.get("validate_all_calls_executed", False):
        return True
    
    required_functions = [
        tool_call["function_name"] 
        for tool_call in config.get("forced_tool_calls", [])
        if tool_call.get("required", True)
    ]
    
    successful_functions = execution_results.get("successful", [])
    missing_functions = [fn for fn in required_functions if fn not in successful_functions]
    
    if missing_functions:
        raise ForcedToolCallExecutionError(
            f"Required functions were not executed successfully: {', '.join(missing_functions)}"
        )
    
    print(f"✓ All {len(required_functions)} required functions executed successfully")
    return True


def handle_call_component_case(trigger_phrase, extracted_data, ):
    if trigger_phrase.lower() == "@plot":
        param_name = config["parameter_extraction"]["param_name"]
        ticker_list = extracted_data.get(param_name, [])
        # Format ticker list for the response
        if ticker_list:
            ticker_str = ', '.join(ticker_list) if ticker_list else ''
            content = f"Here is the plot  {ticker_str}"
            messages.append({
                "role": "assistant",
                "content": content,
                "callComponent": {"plot": True, "tickerList": ticker_list}
            })
        else:
            content = "No stocks data available to plot"
            messages.append({
                "role": "assistant",
                "content": content,
                "callComponent": {"plot": False, "tickerList": []}
            })

    return messages

async def _handle_configured_case(data, base_messages, config, user_query, 
                                  async_client, chat_model, max_tokens, semaphore, 
                                  function_map, system_message, tools_payload, trigger_phrase=None):
    """Handles request processing for a specifically configured trigger case."""
    messages = base_messages.copy()
    extracted_data = {}

    print(f"Processing configured case: {config['description']}")

    # 1. Parameter Extraction
    if "parameter_extraction" in config:
        context_for_extraction = messages.copy()
        extracted_data = await _extract_parameters(
            user_query,
            config["parameter_extraction"],
            async_client,
            chat_model,
            max_tokens,
            semaphore,
            context_for_extraction
        )
        print(f"Extracted parameters: {extracted_data}")
        
        # Ensure default is applied if extraction yields nothing
        param_conf = config["parameter_extraction"]
        if not extracted_data.get(param_conf["param_name"]) and param_conf.get("default_value"):
           extracted_data[param_conf["param_name"]] = param_conf["default_value"]

    #==========Check to load UI Component Data============
    trigger = trigger_phrase.lower()
    if trigger in UI_COMPONENT_CONFIG:
        comp_config = UI_COMPONENT_CONFIG[trigger]
        
        # Get parameter name from config
        param_name = config
        for key in comp_config["config_param_path"]:
            param_name = param_name[key]
        
        # Get data list from extracted parameters
        data_list = extracted_data.get(param_name, [])
        
        # Prepare response based on data availability
        if data_list:
            content = comp_config["content_template"].format(tickers=', '.join(data_list))
            component_status = True
        else:
            content = comp_config["no_data_content"]
            component_status = False
        
        # Build component payload
        call_component = {
            comp_config["component_name"]: component_status,
            comp_config["data_key"]: data_list if component_status else []
        }
        
        # Add formatted message
        messages.append({
            "role": "assistant",
            "content": content,
            "callComponent": call_component
        })
        return messages



    # 2. Initial LLM Call (optional)
    '''
    if config.get("perform_initial_llm_call", False):
        print("Performing initial LLM call...")
        async with semaphore:
            initial_response = await async_client.chat.completions.create(
                model=chat_model,
                messages=messages,
                max_tokens=max_tokens
            )
        assistant_msg_before_forced = initial_response.choices[0].message
        messages.append(assistant_msg_before_forced)
    '''

    # 3. Execute Forced Tool Calls (GUARANTEED EXECUTION)
    if "forced_tool_calls" in config and config["forced_tool_calls"]:
        print(f"Executing {len(config['forced_tool_calls'])} forced tool calls...")
        
        forced_tool_call_objects_for_api = []
        param_values_for_template = []
        
        if "parameter_extraction" in config:
            param_values_for_template = extracted_data.get(
                config["parameter_extraction"]["param_name"], []
            )

        # Build forced tool call objects
        for i, tool_conf in enumerate(config["forced_tool_calls"]):
            function_args = {}
            if "arguments_mapping" in tool_conf:
                for func_arg_name, source_param_key in tool_conf["arguments_mapping"].items():
                    if source_param_key in extracted_data:
                        function_args[func_arg_name] = extracted_data[source_param_key]
                    else:
                        print(f"Warning: Source parameter '{source_param_key}' not found. Using empty list.")
                        function_args[func_arg_name] = []
            
            forced_tool_call_objects_for_api.append({
                "id": tool_conf["id_template"].format(index=i),
                "type": "function",
                "function": {
                    "name": tool_conf["function_name"],
                    "arguments": json.dumps(function_args)
                }
            })

        if forced_tool_call_objects_for_api:
            # Create assistant message with forced tool calls
            assistant_content = "Proceeding with required actions..."
            if config.get("pre_forced_tools_assistant_message_template"):
                assistant_content = config["pre_forced_tools_assistant_message_template"].format(
                    params=", ".join(map(str, param_values_for_template)) if param_values_for_template else "the relevant items"
                )
            
            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": forced_tool_call_objects_for_api
            })
            
            # Execute forced tools with validation
            execution_results = await _execute_and_append_tool_calls(
                forced_tool_call_objects_for_api, 
                messages, 
                function_map,
                validate_execution=config.get("validate_all_calls_executed", False)
            )
            
            # Additional validation step
            await _validate_forced_tool_calls_completion(config, execution_results)

    # 4. Final Response from LLM
    print("Getting final response from LLM...")
    #Append in the end the html format rules
    messages.append(html_formatting_system_prompt)
    final_llm_messages = messages.copy()
    async with semaphore:
        final_response = await async_client.chat.completions.create(
            model=chat_model,
            messages=final_llm_messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
    final_assistant_msg = final_response.choices[0].message
    '''
    if add_plot: 
        call_component = {'plot': True, "tickerList": extracted_data['ticker_list']}
    else:
        call_component = {}
    messages.append({
            "role": "assistant",
            "content": final_assistant_msg.content,
            "callComponent": call_component
        })
    '''
    messages.append(final_assistant_msg)


    print("✓ Configured case processing completed successfully")
    return messages


async def _handle_output_only_case(messages, async_client, chat_model, max_tokens, semaphore):
    """Direct response for output_only intent"""
    messages.append(html_formatting_system_prompt)
    
    async with semaphore:
        response = await async_client.chat.completions.create(
            model=chat_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
    
    assistant_msg = response.choices[0].message
    print(assistant_msg)
    messages.append(assistant_msg)
    return messages

async def _classify_intent(user_query, async_client, model, semaphore):
    """Classify user intent into one of three categories."""
    intent_prompt = """
    Classify the user's intent into one of these categories:
    1. function_code - Requires function calling + code generation (e.g., "plot the revenue of Tesla"). everything where external dataset is needed and to coding to achieve the goal to the fullest.
    2. function_only - Requires function calling only (e.g., "what is Tesla's revenue?"). everything where external dataset is needed.
    3. output_only - Can be answered directly (e.g., "who are you?")
    
    Respond ONLY with the intent category name (function_code, function_only, or output_only).
    """
    
    messages = [
        {"role": "system", "content": intent_prompt},
        {"role": "user", "content": user_query}
    ]
    
    async with semaphore:
        response = await async_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=20,
            temperature=0
        )
    
    intent = response.choices[0].message.content.strip().lower()
    valid_intents = ["function_code", "function_only", "output_only"]
    return intent if intent in valid_intents else "function_only"  # Default fallback

async def process_request(data, async_client, function_map, request_semaphore, system_message, 
                         CHAT_MODEL, MAX_TOKENS, tools_payload):
    """
    Enhanced process_request function with guaranteed function execution for triggers.
    """
    user_query = data.query.lower()


    # Get the latest 20 messages only
    current_messages_history = list(data.messages)[-20:]

    prepared_initial_messages = [system_message] + current_messages_history + [
        {"role": "user", "content": user_query}
    ]

    # Classify intent
    '''
    intent = await _classify_intent(
        user_query, async_client, CHAT_MODEL, request_semaphore
    )
    print(f"Detected intent: {intent}")

    if intent == 'output_only':
        return await _handle_output_only_case(prepared_initial_messages, async_client, CHAT_MODEL, MAX_TOKENS, request_semaphore)
    '''

    active_config = None
    trigger_phrase_found = None

    # Check for trigger phrases
    for trigger, config_item in TRIGGER_CONFIG.items():
        if trigger.lower() in user_query:
            active_config = config_item
            trigger_phrase_found = trigger
            '''
            if "@comparestocks" in user_query:
                add_plot = True
            '''
            print(f"Detected trigger: {trigger_phrase_found}")
            break
            
    try:
        if active_config:
            # Handle request using the specific configuration for the detected trigger
            # This guarantees all configured functions will be called
            return await _handle_configured_case(
                data, prepared_initial_messages, active_config, user_query,
                async_client, CHAT_MODEL, MAX_TOKENS, request_semaphore,
                function_map, system_message, tools_payload, trigger_phrase=trigger_phrase_found)
        else:
            # Standard flow: LLM decides on tool usage
            print("Processing standard request (no triggers detected)")
            messages = prepared_initial_messages.copy()
            #Append in the end the html format rules
            messages.append(html_formatting_system_prompt)
            async with request_semaphore:
                response = await async_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                    tools=tools_payload,
                    tool_choice="auto" if tools_payload else "none"
                )
            
            assistant_msg = response.choices[0].message
            messages.append(assistant_msg)

            # Handle multiple rounds of tool calls if initiated by the LLM
            while hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
                await _execute_and_append_tool_calls(assistant_msg.tool_calls, messages, function_map)
                
                async with request_semaphore:
                    followup_response = await async_client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        temperature=0.7,
                        tools=tools_payload,
                        tool_choice="auto" if tools_payload else "none" 
                    )
                assistant_msg = followup_response.choices[0].message
                messages.append(assistant_msg)
            
            return messages

    except ForcedToolCallExecutionError as e:
        print(f"Forced tool call execution failed: {e}")
        # You might want to return an error message or retry logic here
        raise
    except Exception as e:
        print(f"Request processing failed: {e}")
        raise


# --- Utility Functions for Monitoring ---

def get_trigger_statistics():
    """Returns statistics about configured triggers."""
    stats = {}
    for trigger, config in TRIGGER_CONFIG.items():
        forced_calls = config.get("forced_tool_calls", [])
        stats[trigger] = {
            "description": config.get("description", ""),
            "total_forced_functions": len(forced_calls),
            "required_functions": len([fc for fc in forced_calls if fc.get("required", True)]),
            "function_names": [fc["function_name"] for fc in forced_calls],
            "validation_enabled": config.get("validate_all_calls_executed", False)
        }
    return stats


def validate_trigger_config():
    """Validates the trigger configuration for consistency."""
    issues = []
    
    for trigger, config in TRIGGER_CONFIG.items():
        # Check for required fields
        if not config.get("description"):
            issues.append(f"{trigger}: Missing description")
        
        # Check forced tool calls
        forced_calls = config.get("forced_tool_calls", [])
        for i, call in enumerate(forced_calls):
            if not call.get("function_name"):
                issues.append(f"{trigger}: Tool call {i} missing function_name")
            if not call.get("id_template"):
                issues.append(f"{trigger}: Tool call {i} missing id_template")
    
    return issues