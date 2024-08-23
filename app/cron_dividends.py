import aiohttp
import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm
import orjson

async def save_as_json(symbol, data):
    with open(f"json/dividends/companies/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


async def get_data(ticker, con, etf_con, stock_symbols, etf_symbols):
    try:
        if ticker in etf_symbols:
            table_name = 'etfs'
            column_name = 'etf_dividend'
        else:
            table_name = 'stocks'
            column_name = 'stock_dividend'

        query_template = f"""
        SELECT 
            {column_name}, quote
        FROM 
            {table_name}
        WHERE
            symbol = ?
        """

        df = pd.read_sql_query(query_template, etf_con if table_name == 'etfs' else con, params=(ticker,))
    
        dividend_data = orjson.loads(df[column_name].iloc[0])
        
        res = dividend_data.get('historical', [])

        filtered_res = [item for item in res if item['recordDate'] != '' and item['paymentDate'] != '']

        # Calculate payout frequency based on dividends recorded in 2023
        payout_frequency = sum(1 for item in filtered_res if '2023' in item['recordDate'])
        quote_data = orjson.loads(df['quote'].iloc[0])[0]
        eps = quote_data.get('eps')
        current_price = quote_data.get('price')

        amount = filtered_res[0]['adjDividend'] if filtered_res else 0
        annual_dividend = round(amount * payout_frequency, 2)
        dividend_yield = round((annual_dividend / current_price) * 100, 2) if current_price else None

        payout_ratio = round((1 - (eps - annual_dividend) / eps) * 100, 2) if eps else None

        previous_index = next((i for i, item in enumerate(filtered_res) if '2023' in item['recordDate']), None)

        # Calculate previousAnnualDividend and dividendGrowth
        previous_annual_dividend = (filtered_res[previous_index]['adjDividend'] * payout_frequency) if previous_index is not None else 0
        dividend_growth = round(((annual_dividend - previous_annual_dividend) / previous_annual_dividend) * 100, 2) if previous_annual_dividend else None


        return {
            'payoutFrequency': payout_frequency,
            'annualDividend': annual_dividend,
            'dividendYield': dividend_yield,
            'payoutRatio': payout_ratio,
            'dividendGrowth': dividend_growth,
            'history': filtered_res,
        }
    
    except:
        res = {}

    return res


async def run():

    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    total_symbols = stock_symbols + etf_symbols

    for ticker in tqdm(total_symbols):
        res = await get_data(ticker, con, etf_con, stock_symbols, etf_symbols)
        try:
            if len(res.get('history')) > 0 and res.get('dividendGrowth') != None:
                await save_as_json(ticker, res)
        except:
            pass


    con.close()
    etf_con.close()


try:
    asyncio.run(run())
except Exception as e:
    print(e)
