from datetime import datetime, timedelta, time
import os
import orjson
import pytz

def check_market_hours():

    holidays = [
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
        "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
        "2024-11-28", "2024-12-25"
    ]
    
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
    elif 9 <= current_hour < 16 or (current_hour == 17 and current_minute == 0):
        return True #"Market hours."
    else:
        return False #"Market is closed."


def load_latest_json(directory: str):
    """Load the latest JSON file from a directory based on the filename (assumed to be a date)."""
    try:
        latest_file = None
        latest_date = None
        
        # Iterate over files in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                # Extract date from filename (assumed format 'YYYY-MM-DD.json')
                file_date = filename.split('.')[0]
                
                if latest_date is None or file_date > latest_date:
                    latest_date = file_date
                    latest_file = filename
        
        if not latest_file:
            return []  # No files found
        
        latest_file_path = os.path.join(directory, latest_file)
        with open(latest_file_path, 'rb') as file:
            return orjson.loads(file.read())
    except Exception as e:
        print(f"Error loading latest JSON file: {e}")
        return []