from datetime import datetime, timedelta, time, date
import os
import orjson
import pytz

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