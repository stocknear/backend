import sqlite3
import os
import json


frontend_json_url = "../../frontend/src/lib/hedge-funds"

def format_company_name(company_name):
    remove_strings = [', LLC','LLC', ',', 'LP', 'LTD', 'LTD.', 'INC.', 'INC', '.', '/DE/','/MD/','PLC']
    preserve_words = ['FMR','MCF']

    remove_strings_set = set(remove_strings)
    preserve_words_set = set(preserve_words)

    words = company_name.split()

    formatted_words = []
    for word in words:
        if word in preserve_words_set:
            formatted_words.append(word)
        else:
            new_word = word
            for string in remove_strings_set:
                new_word = new_word.replace(string, '')
            formatted_words.append(new_word.title())
    
    return ' '.join(formatted_words)


def best_hedge_funds(con):
    
    # Connect to the SQLite database
    cursor = con.cursor()

    # Execute a SQL query to select the top 10 best performing cik entries by winRate
    cursor.execute("SELECT cik, name, numberOfStocks, marketValue, winRate, turnover, performancePercentage3year FROM institutes WHERE marketValue > 200000000 AND numberOfStocks > 15 ORDER BY winRate DESC LIMIT 50")
    best_performing_ciks = cursor.fetchall()

    res_list = [{
        'cik': row[0],
        'name': format_company_name(row[1]),
        'numberOfStocks': row[2],
        'marketValue': row[3],
        'winRate': row[4],
        'turnover': row[5],
        'performancePercentage3year': row[6]
    } for row in best_performing_ciks]

    with open(f"json/hedge-funds/best-hedge-funds.json", 'w') as file:
        json.dump(res_list, file)


def worst_hedge_funds(con):
    
    # Connect to the SQLite database
    cursor = con.cursor()

    cursor.execute("SELECT cik, name, numberOfStocks, marketValue, winRate, turnover, performancePercentage3year FROM institutes WHERE marketValue > 200000000 AND numberOfStocks > 15 AND winRate > 0 ORDER BY winRate ASC LIMIT 50")
    worst_performing_ciks = cursor.fetchall()

    res_list = [{
        'cik': row[0],
        'name': format_company_name(row[1]),
        'numberOfStocks': row[2],
        'marketValue': row[3],
        'winRate': row[4],
        'turnover': row[5],
        'performancePercentage3year': row[6]
    } for row in worst_performing_ciks]

    with open(f"json/hedge-funds/worst-hedge-funds.json", 'w') as file:
        json.dump(res_list, file)


def all_hedge_funds(con):
    
    # Connect to the SQLite database
    cursor = con.cursor()

    cursor.execute("SELECT cik, name, numberOfStocks, marketValue, winRate, turnover, performancePercentage3year FROM institutes")
    all_ciks = cursor.fetchall()

    res_list = [{
    'cik': row[0],
    'name': format_company_name(row[1]),
    'numberOfStocks': row[2],
    'marketValue': row[3],
    'winRate': row[4],
    'turnover': row[5],
    'performancePercentage3year': row[6]
    } for row in all_ciks if row[2] >= 3]

    sorted_res_list = sorted(res_list, key=lambda x: x['marketValue'], reverse=True)

    with open(f"json/hedge-funds/all-hedge-funds.json", 'w') as file:
        json.dump(sorted_res_list, file)


def spy_performance():
    import pandas as pd
    import yfinance as yf
    from datetime import datetime

    # Define the start date and end date
    start_date = '1993-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Generate the range of dates with quarterly frequency
    date_range = pd.date_range(start=start_date, end=end_date, freq='Q')

    # Convert the dates to the desired format (end of quarter dates)
    end_of_quarters = date_range.strftime('%Y-%m-%d').tolist()

    data = []

    df = yf.download('SPY', start='1993-01-01', end=datetime.today(), interval="1d").reset_index()
    df = df.rename(columns={'Adj Close': 'close', 'Date': 'date'})

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    for target_date in end_of_quarters:
        original_date = target_date
        # Find close price for '2015-03-31' or the closest available date prior to it    
        while target_date not in df['date'].values:
            # If the target date doesn't exist, move one day back
            target_date = (pd.to_datetime(target_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        # Get the close price for the found or closest date
        close_price = round(df[df['date'] == target_date]['close'].values[0],2)
        data.append({'date': original_date, 'price': close_price})



if __name__ == '__main__':
    con = sqlite3.connect('institute.db')
    #best_hedge_funds(con)
    #worst_hedge_funds(con)
    all_hedge_funds(con)
    spy_performance()
    con.close()