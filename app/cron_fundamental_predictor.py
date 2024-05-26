import ujson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
from ml_models.fundamental_predictor import FundamentalPredictor
import yfinance as yf
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import re


async def save_json(symbol, data):
    with open(f"json/fundamental-predictor-analysis/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

async def download_data(ticker, con, start_date, end_date):
    try:
        query_template = """
            SELECT 
                income, income_growth, balance, balance_growth, cashflow, cashflow_growth, ratios
            FROM 
                stocks 
            WHERE
                symbol = ?
        """

        query_df = pd.read_sql_query(query_template, con, params=(ticker,))

        income =  ujson.loads(query_df['income'].iloc[0])
        #Only consider company with at least 10 year worth of data
        if len(income) < 40:
            raise ValueError("Income data length is too small.")

        income = [{k: v for k, v in item.items() if k not in ["symbol","reportedCurrency","calendarYear","fillingDate","acceptedDate","period","cik","link", "finalLink"]} for item in income if int(item["date"][:4]) >= 2000]
        income_growth = ujson.loads(query_df['income_growth'].iloc[0])
        income_growth = [{k: v for k, v in item.items() if k not in ["symbol","reportedCurrency","calendarYear","fillingDate","acceptedDate","period","cik","link", "finalLink"]} for item in income_growth if int(item["date"][:4]) >= 2000]
        
        balance = ujson.loads(query_df['balance'].iloc[0])
        balance = [{k: v for k, v in item.items() if k not in ["symbol","reportedCurrency","calendarYear","fillingDate","acceptedDate","period","cik","link", "finalLink"]} for item in balance if int(item["date"][:4]) >= 2000]
        balance_growth = ujson.loads(query_df['balance_growth'].iloc[0])
        balance_growth = [{k: v for k, v in item.items() if k not in ["symbol","reportedCurrency","calendarYear","fillingDate","acceptedDate","period","cik","link", "finalLink"]} for item in balance_growth if int(item["date"][:4]) >= 2000]

        cashflow = ujson.loads(query_df['cashflow'].iloc[0])
        cashflow = [{k: v for k, v in item.items() if k not in ["symbol","reportedCurrency","calendarYear","fillingDate","acceptedDate","period","cik","link", "finalLink"]} for item in cashflow if int(item["date"][:4]) >= 2000]
        cashflow_growth = ujson.loads(query_df['cashflow_growth'].iloc[0])
        cashflow_growth = [{k: v for k, v in item.items() if k not in ["symbol","reportedCurrency","calendarYear","fillingDate","acceptedDate","period","cik","link", "finalLink"]} for item in cashflow_growth if int(item["date"][:4]) >= 2000]


        ratios = ujson.loads(query_df['ratios'].iloc[0])
        ratios = [{k: v for k, v in item.items() if k not in ["symbol","reportedCurrency","calendarYear","fillingDate","acceptedDate","period","cik","link", "finalLink"]} for item in ratios if int(item["date"][:4]) >= 2000]

        combined_data = defaultdict(dict)
        # Iterate over all lists simultaneously
        for entries in zip(income, income_growth, balance, balance_growth, cashflow, cashflow_growth, ratios):
            # Iterate over each entry in the current set of entries
            for entry in entries:
                date = entry['date']
                # Merge entry data into combined_data, skipping duplicate keys
                for key, value in entry.items():
                    if key not in combined_data[date]:
                        combined_data[date][key] = value
        
        combined_data = list(combined_data.values())

        df = yf.download(ticker, start=start_date, end=end_date, interval="1d").reset_index()
        df = df.rename(columns={'Adj Close': 'close', 'Date': 'date'})
        #print(df[['date','close']])
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')


        for item in combined_data:
            # Find close price for '2023-09-30' or the closest available date prior to it
            target_date = item['date']
            counter = 0
            max_attempts = 10
            
            while target_date not in df['date'].values and counter < max_attempts:
                # If the target date doesn't exist, move one day back
                target_date = (pd.to_datetime(target_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                counter += 1
            if counter == max_attempts:
                break

            # Get the close price for the found or closest date
            close_price = round(df[df['date'] == target_date]['close'].values[0],2)
            item['price'] = close_price

            #print(f"Close price for {target_date}: {close_price}")
            
        

        combined_data = sorted(combined_data, key=lambda x: x['date'])

        
        df_income = pd.DataFrame(combined_data).dropna()

        df_income['Target'] = ((df_income['price'].shift(-1) - df_income['price']) / df_income['price'] > 0).astype(int)

        df_copy = df_income.copy()
        #print(df_copy)
        
        return df_copy

    except Exception as e:
        print(e)


async def process_symbol(ticker, con, start_date, end_date):
    try:
        test_size = 0.4
        start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        predictor = FundamentalPredictor(path="ml_models/weights")
        df = await download_data(ticker, con, start_date, end_date)
        split_size = int(len(df) * (1-test_size))
        test_data = df.iloc[split_size:]
        selected_features = ['shortTermCoverageRatios','netProfitMargin','debtRepayment','totalDebt','interestIncome','researchAndDevelopmentExpenses','priceEarningsToGrowthRatio','priceCashFlowRatio','cashPerShare','debtRatio','growthRevenue','revenue','growthNetIncome','ebitda','priceEarningsRatio','priceToBookRatio','epsdiluted','priceToSalesRatio','growthOtherCurrentLiabilities', 'receivablesTurnover', 'totalLiabilitiesAndStockholdersEquity', 'totalLiabilitiesAndTotalEquity', 'totalAssets', 'growthOtherCurrentAssets', 'retainedEarnings', 'totalEquity']
        data, prediction_list = predictor.evaluate_model(test_data[selected_features], test_data['Target'])
        

        '''
        output_list = [{'date': date, 'price': price, 'prediction': prediction, 'target': target} 
                                for (date, price,target), prediction in zip(test_data[['date', 'price','Target']].iloc[-6:].values, prediction_list[-6:])]
        '''
        #print(output_list)

        if len(data) != 0:
            if data['precision'] >= 50 and data['accuracy'] >= 50:
                await save_json(ticker, data)
    
    except Exception as e:
        print(e)

async def run():
    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 1E9")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    
    total_symbols = stock_symbols
    
    print(f"Total tickers: {len(total_symbols)}")
    start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")

    chunk_size = len(total_symbols) #// 70  # Divide the list into N chunks
    chunks = [total_symbols[i:i + chunk_size] for i in range(0, len(total_symbols), chunk_size)]
    for chunk in chunks:
        tasks = []
        for ticker in tqdm(chunk):
            tasks.append(process_symbol(ticker, con, start_date, end_date))

        await asyncio.gather(*tasks)

    con.close()
try:
    asyncio.run(run())
except Exception as e:
    print(e)

