import sqlite3
import os
import ujson
import time
from collections import Counter
from tqdm import tqdm

keys_to_keep = [
    "type", "securityName", "symbol", "weight", 
    "changeInSharesNumberPercentage", "sharesNumber", 
    "marketValue", "avgPricePaid", "putCallShare"
]

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
        ujson.dump(sorted_res_list, file)


def spy_performance():
    import pandas as pd
    import yfinance as yf
    from datetime import datetime

    # Define the start date and end date
    start_date = '1993-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Generate the range of dates with quarterly frequency
    date_range = pd.date_range(start=start_date, end=end_date, freq='QE')

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


def get_data(cik, stock_sectors):
    cursor.execute("SELECT cik, name, numberOfStocks, performancePercentage3year, performancePercentage5year, performanceSinceInceptionPercentage, averageHoldingPeriod, turnover, marketValue, winRate, holdings, summary FROM institutes WHERE cik = ?", (cik,))
    cik_data = cursor.fetchall()
    res = [{
        'cik': row[0],
        'name': row[1],
        'numberOfStocks': row[2],
        'performancePercentage3year': row[3],
        'performancePercentage5year': row[4],
        'performanceSinceInceptionPercentage': row[5],
        'averageHoldingPeriod': row[6],
        'turnover': row[7],
        'marketValue': row[8],
        'winRate': row[9],
        'holdings': ujson.loads(row[10]),
        'summary': ujson.loads(row[11]),
    } for row in cik_data]

    if not res:
        return None  # Exit if no data is found

    res = res[0] #latest data

    filtered_holdings = [
        {key: holding[key] for key in keys_to_keep}
        for holding in res['holdings']
    ]

    res['holdings'] = filtered_holdings

    # Cross-reference symbols in holdings with stock_sectors to determine sectors
    sector_counts = Counter()
    for holding in res['holdings']:
        symbol = holding['symbol']
        sector = next((item['sector'] for item in stock_sectors if item['symbol'] == symbol), None)
        if sector:
            sector_counts[sector] += 1

    # Calculate the total number of holdings
    total_holdings = sum(sector_counts.values())

    # Calculate the percentage for each sector and get the top 5
    top_5_sectors_percentage = [
        {sector: round((count / total_holdings) * 100, 2)}
        for sector, count in sector_counts.most_common(5)
    ]

    # Add the top 5 sectors information to the result
    res['topSectors'] = top_5_sectors_percentage
    if res:
        with open(f"json/hedge-funds/companies/{cik}.json", 'w') as file:
            ujson.dump(res, file)

if __name__ == '__main__':
    con = sqlite3.connect('institute.db')
    stock_con = sqlite3.connect('stocks.db')
    
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT cik FROM institutes")
    cik_symbols = [row[0] for row in cursor.fetchall()]

    try:
        stock_cursor = stock_con.cursor()
        stock_cursor.execute("SELECT DISTINCT symbol, sector FROM stocks")
        stock_sectors = [{'symbol': row[0], 'sector': row[1]} for row in stock_cursor.fetchall()]
    finally:
        # Ensure that the cursor and connection are closed even if an error occurs
        stock_cursor.close()
        stock_con.close()

    all_hedge_funds(con)
    spy_performance()
    for cik in tqdm(cik_symbols):
        try:
            get_data(cik, stock_sectors)
        except Exception as e:
            print(e)

    con.close()