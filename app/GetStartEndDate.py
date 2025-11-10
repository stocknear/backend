from datetime import datetime, timedelta
import pytz


class GetStartEndDate:
    def __init__(self):
        self.new_york_tz = pytz.timezone('America/New_York')
        self.current_datetime = datetime.now(self.new_york_tz)

    def check_if_holiday(self):
        holiday_dates = {
            datetime(2025, 1, 1): 'new_year',

        }

        for date, name in holiday_dates.items():
            if date.date() == self.current_datetime.date():
                return name
        return None

    def correct_1d_interval(self, holiday):
        if holiday in ('new_year', 'new_year+1'):
            start_date_1d = datetime(2024, 12, 31)
        else:
            current_time_new_york = datetime.now(self.new_york_tz)
            current_weekday = current_time_new_york.weekday()
            is_afternoon = current_time_new_york.hour > 9 or (current_time_new_york.hour == 9 and current_time_new_york.minute >= 30)

            if current_weekday == 0:
                start_date_1d = current_time_new_york if is_afternoon else current_time_new_york - timedelta(days=3)
            elif current_weekday in (5, 6):  # Saturday or Sunday
                start_date_1d = current_time_new_york - timedelta(days=current_weekday % 5 + 1)
            else:
                start_date_1d = current_time_new_york if is_afternoon else current_time_new_york - timedelta(days=1)
        return start_date_1d

    def run(self):
        holiday = self.check_if_holiday()
        start_date_1d = self.correct_1d_interval(holiday)

        current_time_new_york = datetime.now(self.new_york_tz)
        is_afternoon = current_time_new_york.hour > 9 or (current_time_new_york.hour == 9 and current_time_new_york.minute >= 30)
        if holiday:
            holiday_dates = {
                'new_year': datetime(2024, 12, 31),
            }

            if holiday in holiday_dates:
                end_date_1d = holiday_dates[holiday]
            elif holiday in ['new_year+1'] and not is_afternoon:
                end_date_1d = holiday_dates[holiday]
            else:
                end_date_1d = self.current_datetime
        elif current_time_new_york.weekday() == 0:
            end_date_1d = current_time_new_york if is_afternoon else current_time_new_york - timedelta(days=3)
        else:
            end_date_1d = current_time_new_york

        return start_date_1d, end_date_1d


#Test Mode
#start, end = GetStartEndDate().run()
#print(start, end)