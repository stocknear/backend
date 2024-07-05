from datetime import datetime, timedelta
import pytz


class GetStartEndDate:
    def __init__(self):
        self.new_york_tz = pytz.timezone('America/New_York')
        self.current_datetime = datetime.now(self.new_york_tz)

    def check_if_holiday(self):
        holiday_dates = {
            datetime(2023, 9, 4): 'labor_day',
            datetime(2023, 9, 5): 'labor_day+1',
            datetime(2023, 11, 23): 'thanks_giving',
            datetime(2023, 12, 25): 'christmas',
            datetime(2024, 1, 1): 'new_year',
            datetime(2024, 1, 15): 'martin_luther_king',
            datetime(2024, 2, 19): 'washington_birthday',
            datetime(2024, 5, 27): 'memorial_day',
            datetime(2024, 7, 4): 'independence_day',
        }

        for date, name in holiday_dates.items():
            if date.date() == self.current_datetime.date():
                return name
        return None

    def correct_1d_interval(self, holiday):
        if holiday in ('labor_day', 'labor_day+1'):
            start_date_1d = datetime(2023, 9, 1)
        elif holiday == 'thanks_giving':
            start_date_1d = datetime(2023, 11, 22)
        elif holiday == 'new_year':
            start_date_1d = datetime(2023, 12, 29)
        elif holiday == 'martin_luther_king':
            start_date_1d = datetime(2023, 1, 12)
        elif holiday == 'washington_birthday':
            start_date_1d = datetime(2024, 2, 16)
        elif holiday == 'memorial_day':
            start_date_1d = datetime(2024, 5, 24)
        elif holiday in ('independence_day', 'independence_day+1'):
            start_date_1d = datetime(2024, 7, 3)
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
                'labor_day': datetime(2023, 9, 1),
                'labor_day+1': datetime(2023, 9, 1),
                'thanks_giving': datetime(2023, 11, 22),
                'christmas': datetime(2023, 12, 22),
                'new_year': datetime(2023, 12, 29),
                'martin_luther_king': datetime(2024, 1, 12),
                'washington_birthday': datetime(2024, 2, 16),
                'memorial_day': datetime(2024, 5, 24),
                'independence_day': datetime(2024, 7, 3),
            }

            if holiday in holiday_dates:
                end_date_1d = holiday_dates[holiday]
            elif holiday in ['independence_day+1','labor_day+1', 'christmas_day+1'] and not is_afternoon:
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