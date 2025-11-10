import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import json
import time

class Past_Market_Movers:
    def __init__(self):
        self.con = sqlite3.connect('backup_db/stocks.db')
        self.cursor = self.con.cursor()
        self.cursor.execute("PRAGMA journal_mode = wal")
        self.symbols = self.get_stock_symbols()

    def get_stock_symbols(self):
        self.cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol != ?", ('%5EGSPC',))
        return [row[0] for row in self.cursor.fetchall()]

    @staticmethod
    def check_if_holiday():
        holidays = {
            datetime(2023, 5, 29): 'memorial_day',
            datetime(2023, 6, 19): 'independence_day',
            datetime(2023, 6, 20): 'independence_day+1',
            datetime(2023, 9, 4): 'labor_day',
        }

        current_datetime = datetime.today()
        for holiday_date, holiday_name in holidays.items():
            if current_datetime == holiday_date:
                return holiday_name
        return None

    def correct_weekday_interval(self, prev_day):
        holiday = self.check_if_holiday()
        if holiday:
            if holiday == 'memorial_day':
                start_date = datetime(2023, 5, 26)
            elif holiday in ('independence_day', 'independence_day+1'):
                start_date = datetime(2023, 6, 16)
            else:
                start_date = datetime(2023, 9, 1)
        else:
            current_date = datetime.today() - timedelta(prev_day)
            current_weekday = current_date.weekday()
            if current_weekday in (5, 6):  # Saturday or Sunday
                start_date = current_date - timedelta(days=current_weekday % 5 + 1)
            else:
                start_date = current_date
        return start_date.strftime("%Y-%m-%d")

    def run(self, time_periods=[7,20,252,756,1260]):
        performance_data = []
        query_template = """
            SELECT date, close, volume FROM "{ticker}" WHERE date >= ?
        """
        query_fundamental_template = """
            SELECT marketCap, name FROM stocks WHERE symbol = ?
        """
        gainer_json = {}
        loser_json = {}
        active_json = {}

        for time_period in time_periods:
            performance_data = []
            high_volume = []
            gainer_data = []
            loser_data = []
            active_data = []

            start_date = self.correct_weekday_interval(time_period)
            for ticker in self.symbols:
                try:
                    query = query_template.format(ticker=ticker)
                    df = pd.read_sql_query(query, self.con, params=(start_date,))
                    if not df.empty:
                        fundamental_data = pd.read_sql_query(query_fundamental_template, self.con, params=(ticker,))
                        avg_volume = df['volume'].mean()
                        market_cap = int(fundamental_data['marketCap'].iloc[0])
                        if avg_volume > 1E6 and df['close'].mean() > 1 and market_cap >=50E6:
                            changes_percentage = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
                            performance_data.append((ticker, fundamental_data['name'].iloc[0], df['close'].iloc[-1], changes_percentage, avg_volume, market_cap))
                except:
                    pass

            # Sort the stocks by percentage change in descending order
            performance_data.sort(key=lambda x: x[3], reverse=True)
            high_volume = sorted(performance_data, key=lambda x: x[4], reverse=True)

            for symbol, name, price, changes_percentage, volume, market_cap in [entry for entry in performance_data if entry[3] > 0]:
                gainer_data.append({
                    'symbol': symbol,
                    'name': name,
                    'price': price,
                    'changesPercentage': changes_percentage,
                    'volume': volume,
                    'marketCap': market_cap
                })
            for symbol, name, price, changes_percentage, volume, market_cap in [entry for entry in performance_data if entry[3] < 0]:
                loser_data.append({
                    'symbol': symbol,
                    'name': name,
                    'price': price,
                    'changesPercentage': changes_percentage,
                    'volume': volume,
                    'marketCap': market_cap
                })
            for symbol, name, price, changes_percentage, volume, market_cap in high_volume:
                active_data.append({'symbol': symbol, 'name': name, 'price': price, 'changesPercentage': changes_percentage, 'volume': volume, 'marketCap': market_cap})

            loser_data.sort(key=lambda x: x['changesPercentage'], reverse=False)

            if time_period == 7:
                gainer_json['1W'] = gainer_data
                loser_json['1W'] = loser_data
                active_json['1W'] = active_data
            elif time_period == 20:
                gainer_json['1M'] = gainer_data
                loser_json['1M'] = loser_data
                active_json['1M'] = active_data
            elif time_period == 252:
                gainer_json['1Y'] = gainer_data
                loser_json['1Y'] = loser_data
                active_json['1Y'] = active_data
            elif time_period == 756:
                gainer_json['3Y'] = gainer_data
                loser_json['3Y'] = loser_data
                active_json['3Y'] = active_data
            elif time_period == 1260:
                gainer_json['5Y'] = gainer_data
                loser_json['5Y'] = loser_data
                active_json['5Y'] = active_data

        return gainer_json, loser_json, active_json


    def create_table(self):
        """
        Create the 'market_movers' table if it doesn't exist and add 'gainer', 'loser', and 'most_active' columns.
        """
        query_drop = "DROP TABLE IF EXISTS market_movers"
        self.con.execute(query_drop)
        query_create = """
        CREATE TABLE IF NOT EXISTS market_movers (
            gainer TEXT,
            loser TEXT,
            most_active TEXT
        )
        """
        self.con.execute(query_create)
        self.con.commit()


    def update_database(self, gainer_json, loser_json, active_json):
        """
        Update the 'gainer', 'loser', and 'most_active' columns in the 'market_movers' table with the provided JSON data.
        """
        query = "INSERT INTO market_movers (gainer, loser, most_active) VALUES (?, ?, ?)"
        gainer_json_str = json.dumps(gainer_json)
        loser_json_str = json.dumps(loser_json)
        active_json_str = json.dumps(active_json)
        self.con.execute(query, (gainer_json_str, loser_json_str, active_json_str))
        self.con.commit()

    def close_database_connection(self):
        self.con.close()

if __name__ == "__main__":
    analyzer = Past_Market_Movers()
    analyzer.create_table()  # Create the 'market_movers' table with the 'gainer', 'loser', and 'most_active' columns
    gainer_json, loser_json, active_json = analyzer.run()  # Retrieve the gainer_json, loser_json, and active_json data
    analyzer.update_database(gainer_json, loser_json, active_json)  # Update the 'gainer', 'loser', and 'most_active' columns with the respective data
    analyzer.close_database_connection()