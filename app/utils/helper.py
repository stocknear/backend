from datetime import datetime, timedelta, time, date
import os
import orjson
import pytz
import math


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
    """
    Compute the return percentage of an option trade, considering sentiment
    to determine long or short position.

    Parameters
    ----------
    option : dict
        A dict containing at least the keys:
          - "put_call" (str): "CALL", "CALLS", "PUT", or "PUTS"
          - "strike_price" (str or float)
          - "price" (str or float): the premium per share (paid for long, received for short)
          - "size" (str or int): number of contracts
          - "sentiment" (str): "Bullish", "Bearish", or "Neutral". This is used
            to infer if the position is long or short.
          - "cost_basis" (str or float, optional): total dollars paid or received.
            If missing or invalid, it's computed as price * size * 100. For short
            positions, this represents the premium received.

    current_price : float
        Current price of the underlying stock.

    Returns
    -------
    float
        Percentage return on cost basis (e.g. 56.8 for +56.8%).
        Returns None if input is invalid or calculation is not possible.
    """
    try:
        # --- Normalize and parse inputs ---
        pc_raw = option.get("put_call")
        if pc_raw is None:
            # put_call is a required field
            return None

        pc = str(pc_raw).strip().upper()
        if pc.endswith("S"):  # handle "CALLS" and "PUTS"
            pc = pc[:-1]
        if pc not in ("CALL", "PUT"):
            # Handle unexpected put_call types
            return None

        # Safely get and convert numeric values, handling potential errors
        try:
            strike = float(option.get("strike_price"))
            premium = float(option.get("price"))
            size = int(option.get("size"))
            if size <= 0 or premium < 0: # Basic validation for size and premium
                 return None
        except (ValueError, TypeError):
            # Handle cases where numeric conversion fails for required fields
            return None

        sentiment_raw = option.get("sentiment")
        if sentiment_raw is None:
            # Sentiment is required to determine long/short
            return None
        sentiment = str(sentiment_raw).strip().capitalize()

        multiplier = 100  # standard options multiplier

        # total cost basis (premium paid for long, premium received for short)
        # Handle potential None or empty string for cost_basis
        cost_basis_raw = option.get("cost_basis")
        cost_basis = None
        if cost_basis_raw is not None and cost_basis_raw != "":
             try:
                 cost_basis = float(cost_basis_raw)
             except (ValueError, TypeError):
                  # If provided cost_basis is invalid, we will calculate it later
                  pass

        # --- Determine if the position is likely long or short based on option type and sentiment ---
        is_long = None # Use None initially to indicate undetermined
        if pc == "CALL":
            if sentiment == "Bullish" or sentiment == "Neutral":
                is_long = True # Assume long call for bullish/neutral sentiment
            elif sentiment == "Bearish":
                is_long = False # Assume short call for bearish sentiment
        elif pc == "PUT":
            if sentiment == "Bearish" or sentiment == "Neutral":
                is_long = True # Assume long put for bearish/neutral sentiment
            elif sentiment == "Bullish":
                is_long = False # Assume short put for bullish sentiment

        if is_long is None:
            # If we couldn't determine long/short based on input, return None
            return None

        # If cost_basis was not provided or was invalid, calculate it
        if cost_basis is None:
             cost_basis = premium * size * multiplier

        # If cost basis is still non-positive (e.g., zero premium for a short), cannot calculate return percentage meaningfully
        if cost_basis <= 0:
             return None

        # --- Calculate intrinsic value per share (value from the perspective of a long holder) ---
        intrinsic_per_share = 0.0
        if pc == "CALL":
            intrinsic_per_share = max(current_price - strike, 0.0)
        elif pc == "PUT":
            intrinsic_per_share = max(strike - current_price, 0.0)

        # --- Calculate the current value of the position and the profit ---
        profit = 0.0
        if is_long:
            # For a long position, current value is the intrinsic value
            total_current_value = intrinsic_per_share * size * multiplier
            profit = total_current_value - cost_basis
        else: # is_short
            # For a short position, the profit is the premium received minus
            # the intrinsic value that would be paid if closed now.
            # The cost_basis here represents the premium *received*.
            total_value_against_short = intrinsic_per_share * size * multiplier
            profit = cost_basis - total_value_against_short

        # --- Calculate return percentage ---
        return_percentage = (profit / cost_basis) * 100.0

        # Return None if the result is not a finite number (e.g., from extreme values)
        if not math.isfinite(return_percentage):
            return None

        return round(return_percentage, 2)

    except Exception as e:
        # Catch any other unexpected errors during processing
        # print(f"An error occurred: {e}") # Optional: log the error for debugging
        return None # Return None in case of any unhandled exception