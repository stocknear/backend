import sqlite3
import os
import orjson
import time
from datetime import datetime
from collections import Counter
from tqdm import tqdm


# Load stock screener data
with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

keys_to_keep = [
    "type", "securityName", "symbol", "weight", 
    "changeInSharesNumberPercentage", "sharesNumber", 
    "marketValue", "avgPricePaid", "putCallShare","filingDate"
]

quote_cache = {}

cutoff_date = datetime.strptime("2015-01-01", "%Y-%m-%d")

def get_quote_data(symbol):
    """Get quote data for a symbol from JSON file"""
    if symbol in quote_cache:
        return quote_cache[symbol]
    else:
        try:
            with open(f"json/quote/{symbol}.json") as file:
                quote_data = orjson.loads(file.read())
                quote_cache[symbol] = quote_data  # Cache the loaded data
                return quote_data
        except:
            return None

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

def remove_stock_duplicates(stocks):
    """
    Remove duplicate stocks keeping the highest weight entry for each symbol.
    
    Args:
        stocks (list): List of dictionaries containing stock information
        
    Returns:
        list: List with duplicates removed
    """
    symbol_dict = {}
    
    for stock in stocks:
        symbol = stock['symbol']
        weight = stock['weight']
        
        if symbol not in symbol_dict or weight > symbol_dict[symbol]['weight']:
            symbol_dict[symbol] = stock
    
    return list(symbol_dict.values())

def all_hedge_funds(con):
    
    # Connect to the SQLite database
    cursor = con.cursor()
    cursor.execute("SELECT cik, name, numberOfStocks, marketValue, winRate, turnover, performancePercentage3year FROM institutes")
    all_ciks = cursor.fetchall()
    res_list = [{
        'cik': row[0],
        'name': format_company_name(row[1]).title(),
        'numberOfStocks': row[2],
        'marketValue': row[3],
        'winRate': row[4],
        'turnover': row[5],
        'performancePercentage3Year': row[6],
        #'performance3yearRelativeToSP500Percentage': row[7]
    } for row in all_ciks if (
        row[2] is not None and row[2] > 5 and row[4] > 0 and
        row[6] is not None and abs(row[6]) < 500
    )]
    sorted_res_list = sorted(res_list, key=lambda x: x['marketValue'], reverse=True)
    sorted_res_list = [x for x in sorted_res_list if x['marketValue'] <= 15e12]

    #print(sorted_res_list[:10])
    
    for i, item in enumerate(sorted_res_list, 1):
        item['rank'] = i

    with open(f"json/hedge-funds/all-hedge-funds.json", 'w') as file:
        file.write(orjson.dumps(sorted_res_list).decode("utf-8"))


def get_data(cik, stock_sectors):
    cursor.execute("SELECT cik, name, numberOfStocks, performancePercentage3year, averageHoldingPeriod, marketValue, winRate, holdings FROM institutes WHERE cik = ?", (cik,))
    cik_data = cursor.fetchall()
    res = [{
        'cik': row[0],
        'name': row[1],
        'numberOfStocks': row[2],
        'performancePercentage3Year': row[3],
        'averageHoldingPeriod': row[4],
        'marketValue': row[5],
        'winRate': row[6],
        'holdings': orjson.loads(row[7]),
    } for row in cik_data][0]

    if not res:
        return None  # Exit if no data is found

    filtered_holdings = [
        {key: holding[key] for key in keys_to_keep}
        for holding in res['holdings']
    ]

    filtered_holdings = [
        {
            **{k: v for k, v in item.items() if k not in ['securityName']}, 
            'name': item['securityName'].title()
        }
        for item in filtered_holdings 
        if (
            item['avgPricePaid'] > 0 and 
            item['marketValue'] > 0 and 
            item['sharesNumber'] > 0
        )
    ]

    filtered_holdings = remove_stock_duplicates(filtered_holdings)
    #add current price and changespercentage
    for item in filtered_holdings:
        try:
            symbol = item['symbol']
            item['putCallShare'] = item['putCallShare'].title()
            quote_data = get_quote_data(symbol)
            if quote_data:
                item['price'] = quote_data.get('price',None)
                item['changesPercentage'] = round(quote_data.get('changesPercentage'), 2) if quote_data.get('changesPercentage') is not None else None
                item['name'] = quote_data.get('name')
        except:
            pass

    res['filingDate'] = filtered_holdings[0]['filingDate']
    filtered_holdings = [
        {k: v for k, v in item.items() if k != 'filingDate'}
        for item in filtered_holdings
        if item.get('filingDate')
    ]
    res['holdings'] = filtered_holdings


    for rank, item in enumerate(res['holdings'], 1):
        item['rank'] = rank

    sector_list = []
    industry_list = []

    for item in res['holdings']:
        symbol = item['symbol']
        ticker_data = stock_screener_data_dict.get(symbol, {})
        
        # Extract specified columns data for each ticker
        sector = ticker_data.get('sector',None)
        industry = ticker_data.get('industry',None)

        # Append data to relevant lists if values are present
        if sector:
            sector_list.append(sector)
        if industry:
            industry_list.append(industry)       

    # Get the top 3 most common sectors and industries
    sector_counts = Counter(sector_list)
    industry_counts = Counter(industry_list)
    main_sectors = [item[0] for item in sector_counts.most_common(3)]
    main_industries = [item[0] for item in industry_counts.most_common(3)]

    # Add main sectors and industries to the item dictionary
    res['mainSectors'] = main_sectors
    res['mainIndustries'] = main_industries


    if res:
        with open(f"json/hedge-funds/companies/{cik}.json", 'w') as file:
            file.write(orjson.dumps(res).decode("utf-8"))

if __name__ == '__main__':
    con = sqlite3.connect('institute.db')
    stock_con = sqlite3.connect('stocks.db')
    
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT cik FROM institutes")
    cik_symbols = [row[0] for row in cursor.fetchall()]
    #Test mode
    #cik_symbols = ['0001649339']
    try:
        stock_cursor = stock_con.cursor()
        stock_cursor.execute("SELECT DISTINCT symbol, sector FROM stocks")
        stock_sectors = [{'symbol': row[0], 'sector': row[1]} for row in stock_cursor.fetchall()]
    finally:
        # Ensure that the cursor and connection are closed even if an error occurs
        stock_cursor.close()
        stock_con.close()

    all_hedge_funds(con)
    #spy_performance()
    for cik in tqdm(cik_symbols):
        try:
            get_data(cik, stock_sectors)
        except Exception as e:
            print(e)

    con.close()