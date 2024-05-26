import quantstats as qs
from datetime import datetime
import pandas as pd
import sqlite3
from math import sqrt, ceil
from dateutil.relativedelta import relativedelta
import time
import json
from tqdm import tqdm
import concurrent.futures
import numpy as np
import argparse


pd.set_option('display.max_rows', 150)


def parse_args():
    parser = argparse.ArgumentParser(description='Process stock, etf or crypto data.')
    parser.add_argument('--db', choices=['stocks', 'etf', 'crypto'], required=True, help='Database name (stocks or etf)')
    parser.add_argument('--table', choices=['stocks', 'etfs', 'cryptos'], required=True, help='Table name (stocks or etfs)')
    return parser.parse_args()

# Define a function to get the ticker from the database
def get_ticker_data_from_database(database_path, sp500_ticker, start_date, end_date):
    con_etf = sqlite3.connect(database_path)
    
    # Fetch data for the selected ticker (SPY or another ticker)
    query_template = """
        SELECT
            date, close
        FROM
            "{ticker}"
        WHERE
            date BETWEEN ? AND ?
    """
    query = query_template.format(ticker=sp500_ticker)
    df = pd.read_sql_query(query, con_etf, params=(start_date, end_date))
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'Date'})
    df[sp500_ticker] = df['close'].pct_change()
    df.set_index("Date", inplace=True)
    df.drop(columns=['close'], inplace=True)
    con_etf.close()
    
    return sp500_ticker, df

class Quant_Stats:
    def __init__(self):
        pass

    def get_trading_periods(self):
        periods_per_year = 252
        half_year = ceil(periods_per_year / 2)
        return periods_per_year, half_year


    def get_data(self, df, ticker):
        benchmark = "SPY"
        compounded = True 
        rf = 0 
        today = df.index[-1]
        comp_func = qs.stats.comp
        win_year, win_half_year = self.get_trading_periods()

        metrics = pd.DataFrame()
        
        metrics['Expected Daily %'] = round(qs.stats.expected_return(df, compounded=True)*100,2)
        metrics['Expected Monthly %'] = round(qs.stats.expected_return(df, compounded=True,  aggregate="M")*100,2)
        metrics['Expected Yearly %'] = round(qs.stats.expected_return(df, compounded=True,  aggregate="A")*100,2)
        metrics["Cumulative Return %"] = round(qs.stats.comp(df) * 100, 2)
        metrics["CAGR %"] = round(qs.stats.cagr(df, rf, compounded) * 100, 2) 
        metrics["Sharpe"] = qs.stats.sharpe(df, rf, win_year, compounded)
        metrics["Sortino"] = qs.stats.sortino(df, rf, win_year, True)
        metrics["Volatility (ann.) %"] = round(qs.stats.volatility(df, win_year, True)* 100, 2)
        metrics["Calmar"] = round(qs.stats.calmar(df),2)
        metrics["Skew"] = qs.stats.skew(df, prepare_returns=False)
        metrics["Kurtosis"] = qs.stats.kurtosis(df, prepare_returns=False)
        metrics["Kelly Criterion %"] = round(qs.stats.kelly_criterion(df, prepare_returns=False) * 100, 2)
        metrics["Risk of Ruin %"] = round(qs.stats.risk_of_ruin(df, prepare_returns=False), 2)
        metrics["Daily Value-at-Risk %"] = -abs(qs.stats.var(df, prepare_returns=False) * 100)
        metrics["Expected Shortfall (cVaR) %"] = -abs(qs.stats.cvar(df, prepare_returns=False) * 100)
        metrics["Max Consecutive Wins"] = qs.stats.consecutive_wins(df)
        metrics["Max Consecutive Losses"] = qs.stats.consecutive_losses(df)
        metrics["Gain/Pain Ratio"] = qs.stats.gain_to_pain_ratio(df, rf)
        metrics["Gain/Pain (1M)"] = qs.stats.gain_to_pain_ratio(df, rf, "M")
        metrics["Payoff Ratio"] = qs.stats.payoff_ratio(df, prepare_returns=False)
        metrics["Profit Factor"] = qs.stats.profit_factor(df, prepare_returns=False)
        metrics["Common Sense Ratio"] = qs.stats.common_sense_ratio(df, prepare_returns=False)
        metrics["CPC Index"] = qs.stats.cpc_index(df, prepare_returns=False)
        metrics["Tail Ratio"] = qs.stats.tail_ratio(df, prepare_returns=False)
        metrics["Outlier Win Ratio"] = qs.stats.outlier_win_ratio(df, prepare_returns=False)
        metrics["Outlier Loss Ratio"] = qs.stats.outlier_loss_ratio(df, prepare_returns=False)

        #Yearly return is included since eoy = end of the year True
        ticker_monthly_returns = round(qs.stats.monthly_returns(df[ticker], eoy = True, compounded = True) * 100,2)
        benchmark_monthly_returns = round(qs.stats.monthly_returns(df[benchmark], eoy = True, compounded = True) * 100,2)
        metrics['Monthly Return'] = [ticker_monthly_returns.T.to_dict('list'), benchmark_monthly_returns.T.to_dict('list')]


        metrics["MTD %"] = round(comp_func(df[df.index >= datetime(today.year, today.month, 1)]) * 100,2)

        d = today - relativedelta(months=3)
        metrics["3M %"] = comp_func(df[df.index >= d]) * 100

        d = today - relativedelta(months=6)
        metrics["6M %"] = comp_func(df[df.index >= d]) * 100

        metrics["YTD %"] = comp_func(df[df.index >= datetime(today.year, 1, 1)]) * 100

        d = today - relativedelta(years=1)
        metrics["1Y %"] = comp_func(df[df.index >= d]) * 100

        d = today - relativedelta(months=35)
        metrics["3Y (ann.) %"] = qs.stats.cagr(df[df.index >= d], 0.0, compounded) * 100

        d = today - relativedelta(months=59)
        metrics["5Y (ann.) %"] = qs.stats.cagr(df[df.index >= d], 0.0, compounded) * 100

        d = today - relativedelta(years=10)
        metrics["10Y (ann.) %"] = qs.stats.cagr(df[df.index >= d], 0.0, compounded) * 100
        metrics["All-time (ann.) %"] = qs.stats.cagr(df, 0.0, compounded) * 100

        metrics["Best Day %"] = qs.stats.best(df, compounded=compounded, prepare_returns=False) * 100
        metrics["Worst Day %"] = qs.stats.worst(df, prepare_returns=False) * 100
        metrics["Best Month %"] = (qs.stats.best(df, compounded=compounded, aggregate="M", prepare_returns=False) * 100)
        metrics["Worst Month %"] = (qs.stats.worst(df, aggregate="M", prepare_returns=False) * 100)
        metrics["Best Year %"] = (qs.stats.best(df, compounded=compounded, aggregate="A", prepare_returns=False) * 100)
        metrics["Worst Year %"] = (qs.stats.worst(df, compounded=compounded, aggregate="A", prepare_returns=False) * 100)
        
        avg_dd_list = []
        avg_dd_days_list = []
        max_dd_list = []
        longest_dd_days_list = []

        for tt in [ticker, benchmark]:
            dd = qs.stats.to_drawdown_series(df[tt])
            dd_info = qs.stats.drawdown_details(dd).sort_values(by="max drawdown", ascending = True)
            dd_info = dd_info[["start", "end", "max drawdown", "days"]]
            dd_info.columns = ["Started", "Recovered", "Drawdown", "Days"]

            avg_dd_list.append(round(dd_info['Drawdown'].mean(),2))
            max_dd_list.append(round(dd_info['Drawdown'].min(),2))

            avg_dd_days_list.append(round(dd_info['Days'].mean()))
            longest_dd_days_list.append(round(dd_info['Days'].max()))

        metrics["Max Drawdown"] = max_dd_list
        metrics["Avg. Drawdown"] = avg_dd_list

        metrics["Longest DD Days"] = longest_dd_days_list
        metrics["Avg. Drawdown Days"] = avg_dd_days_list

        worst_dd_list = []
        dd = qs.stats.to_drawdown_series(df[ticker])
        dd_info = qs.stats.drawdown_details(dd).sort_values(by="max drawdown", ascending = True)[0:10]
        dd_info = dd_info[["start", "end", "max drawdown", "days"]]
        dd_info.columns = ["Started", "Recovered", "Drawdown", "Days"]

        for key, value in dd_info.T.to_dict().items():
            worst_dd_list.append(value)
        metrics['Worst 10 Drawdowns'] = [worst_dd_list, '-']
        

        metrics["Recovery Factor"] = qs.stats.recovery_factor(df)
        metrics["Ulcer Index"] = qs.stats.ulcer_index(df)
        metrics["Serenity Index"] = qs.stats.serenity_index(df, rf)

        metrics["Avg. Up Month %"] = (qs.stats.avg_win(df, compounded=compounded, aggregate="M", prepare_returns=False) * 100)
        metrics["Avg. Down Month %"] = (qs.stats.avg_loss(df, compounded=compounded, aggregate="M", prepare_returns=False) * 100)
        metrics["Win Days %"] = qs.stats.win_rate(df, prepare_returns=False) * 100
        metrics["Win Month %"] = (qs.stats.win_rate(df, compounded=compounded, aggregate="M", prepare_returns=False) * 100)
        metrics["Win Quarter %"] = (qs.stats.win_rate(df, compounded=compounded, aggregate="Q", prepare_returns=False) * 100)
        metrics["Win Year %"] = (qs.stats.win_rate(df, compounded=compounded, aggregate="A", prepare_returns=False) * 100)


        greeks = qs.stats.greeks(df[ticker], df[benchmark], win_year, prepare_returns=False)

        metrics["Beta"] = [round(greeks["beta"], 2), "-"]
        metrics["Alpha"] = [round(greeks["alpha"], 2), "-"]
        metrics["Correlation"] = [round(df[benchmark].corr(df[ticker]) * 100, 2), "-",]
        metrics["Treynor Ratio"] = [round(qs.stats.treynor_ratio(df[ticker], df[benchmark], win_year, rf) * 100, 2,), "-" ]
        metrics["R^2"] = ([qs.stats.r_squared(df[ticker], df[benchmark], prepare_returns=False ).round(2), "-"]) 
        
        metrics["Start Period"] = df.index[0].strftime("%Y-%m-%d")
        metrics['End Period'] = df.index[-1].strftime("%Y-%m-%d")
        

        metrics = metrics.T

        return metrics


def create_quantstats_column(con):
    """
    Create the 'quantStats' column if it doesn't exist in the db table.
    """
    query_check = f"PRAGMA table_info({table_name})"
    cursor = con.execute(query_check)
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'quantStats' not in columns:
        query = f"ALTER TABLE {table_name} ADD COLUMN quantStats TEXT"
        con.execute(query)
        con.commit()


def update_database_with_stats(stats_dict, symbol, con):
    """
    Update the SQLite3 table with calculated statistics for a given symbol.
    """

    query = f"UPDATE {table_name} SET quantStats = ? WHERE symbol = ?"
    stats_json = json.dumps(stats_dict)  # Convert the stats dictionary to JSON string
    con.execute(query, (stats_json, symbol))
    con.commit()





def process_symbol(ticker, sp500_ticker, sp500_df):
    df = pd.DataFrame()
    combined_df = pd.DataFrame()
    try:
        query_template = """
            SELECT
                date, close
            FROM
                "{ticker}"
            WHERE
                date BETWEEN ? AND ?
        """

        query = query_template.format(ticker=ticker)
        df = pd.read_sql_query(query, con, params=(start_date, end_date))
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'date': 'Date'})
        df[ticker] = df['close'].pct_change()
        df.set_index("Date", inplace=True)
        df.drop(columns=['close'], inplace=True)

        combined_df = pd.concat([sp500_df, df], axis=1)

        df = combined_df.dropna()
        df = df[[ticker, sp500_ticker]]
        stats = Quant_Stats().get_data(df, ticker)
        stats_dict = stats.to_dict()

        create_quantstats_column(con)
        update_database_with_stats(stats_dict, ticker, con)

    except Exception as e:
        print(e)
        print(f"Failed create quantStats for {ticker}")



#Production Code

args = parse_args()
db_name = args.db
table_name = args.table

start_date = datetime(1970, 1, 1)
end_date = datetime.today()

con = sqlite3.connect(f'backup_db/{db_name}.db')

# Load the S&P 500 ticker from the database
sp500_ticker, sp500_df = get_ticker_data_from_database('backup_db/etf.db', "SPY", start_date, end_date)

symbol_query = f"SELECT DISTINCT symbol FROM {table_name}"

symbol_cursor = con.execute(symbol_query)
symbols = [symbol[0] for symbol in symbol_cursor.fetchall()]

# Number of concurrent workers
num_processes = 4 # You can adjust this based on your system's capabilities
futures = []

with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    for symbol in symbols:
        futures.append(executor.submit(process_symbol, symbol, sp500_ticker, sp500_df))

    # Use tqdm to wrap around the futures for progress tracking
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(symbols), desc="Processing"):
        pass


con.close()



#Test Code
'''
con = sqlite3.connect('backup_db/etf.db')
start_date = datetime(1970, 1, 1)
end_date = datetime.today()



query_template = """
    SELECT
        date, close
    FROM
        "{ticker}"
    WHERE
        date BETWEEN ? AND ?
"""

ticker_list = ['IVV','SPY']

combined_df = pd.DataFrame()
for ticker in ticker_list:
    query = query_template.format(ticker=ticker)
    df = pd.read_sql_query(query, con, params=(start_date, end_date))
    print(df)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'Date'})
    df[ticker] = df['close'].pct_change()
    df.set_index("Date", inplace=True)
    df.drop(columns=['close'], inplace=True)
    combined_df = pd.concat([combined_df, df], axis=1)
df = combined_df.dropna()


#monthly_returns = round(qs.stats.monthly_returns(df[ticker], eoy = False, compounded = True) * 100,2)
#yearly_returns = round(qs.stats.monthly_returns(df[ticker], eoy = True, compounded = True) * 100,2)
#print(yearly_returns)
#stats = Quant_Stats().get_data(df, ticker)
#print(stats)


con.close()

'''