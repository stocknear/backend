import orjson
import asyncio
import aiohttp
import aiofiles
import sqlite3
from datetime import datetime
from ml_models.fundamental_predictor import FundamentalPredictor
import yfinance as yf
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import re
import subprocess


async def save_json(symbol, data):
    with open(f"json/fundamental-predictor-analysis/{symbol}.json", 'w') as file:
        file.write(orjson.dumps(data))


async def download_data(ticker, con, start_date, end_date):
    try:
        # Define paths to the statement files
        statements = [
            f"json/financial-statements/ratios/quarter/{ticker}.json",
            f"json/financial-statements/cash-flow-statement/quarter/{ticker}.json",
            f"json/financial-statements/income-statement/quarter/{ticker}.json",
            f"json/financial-statements/balance-sheet-statement/quarter/{ticker}.json",
            f"json/financial-statements/income-statement-growth/quarter/{ticker}.json",
            f"json/financial-statements/balance-sheet-statement-growth/quarter/{ticker}.json",
            f"json/financial-statements/cash-flow-statement-growth/quarter/{ticker}.json"
        ]

        # Helper function to load JSON data asynchronously
        async def load_json_from_file(path):
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
            return orjson.loads(content)

        # Helper function to filter data based on keys and year
        async def filter_data(data, ignore_keys, year_threshold=2000):
            return [{k: v for k, v in item.items() if k not in ignore_keys} for item in data if int(item["date"][:4]) >= year_threshold]

        # Define keys to ignore
        ignore_keys = ["symbol", "reportedCurrency", "calendarYear", "fillingDate", "acceptedDate", "period", "cik", "link", "finalLink"]

        # Load and filter data for each statement type
        income = await load_json_from_file(statements[2])
        income = await filter_data(income, ignore_keys)

        income_growth = await load_json_from_file(statements[4])
        income_growth = await filter_data(income_growth, ignore_keys)

        balance = await load_json_from_file(statements[3])
        balance = await filter_data(balance, ignore_keys)

        balance_growth = await load_json_from_file(statements[5])
        balance_growth = await filter_data(balance_growth, ignore_keys)

        cashflow = await load_json_from_file(statements[1])
        cashflow = await filter_data(cashflow, ignore_keys)

        cashflow_growth = await load_json_from_file(statements[6])
        cashflow_growth = await filter_data(cashflow_growth, ignore_keys)

        ratios = await load_json_from_file(statements[0])
        ratios = await filter_data(ratios, ignore_keys)

        # Combine all the data
        combined_data = defaultdict(dict)

        # Merge the data based on 'date'
        for entries in zip(income, income_growth, balance, balance_growth, cashflow, cashflow_growth, ratios):
            for entry in entries:
                date = entry['date']
                for key, value in entry.items():
                    if key not in combined_data[date]:
                        combined_data[date][key] = value

        combined_data = list(combined_data.values())

        # Download historical stock data using yfinance
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d").reset_index()
        df = df.rename(columns={'Adj Close': 'close', 'Date': 'date'})
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        # Match each combined data entry with the closest available stock price in df
        for item in combined_data:
            target_date = item['date']
            counter = 0
            max_attempts = 10

            # Look for the closest matching date in the stock data
            while target_date not in df['date'].values and counter < max_attempts:
                target_date = (pd.to_datetime(target_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                counter += 1

            # If max attempts are reached and no matching date is found, skip the entry
            if counter == max_attempts:
                continue

            # Find the close price for the matching date
            close_price = round(df[df['date'] == target_date]['close'].values[0], 2)
            item['price'] = close_price

        # Sort the combined data by date
        combined_data = sorted(combined_data, key=lambda x: x['date'])

        # Convert combined data into a DataFrame
        df_combined = pd.DataFrame(combined_data).dropna()

        # Create 'Target' column based on price change
        df_combined['Target'] = ((df_combined['price'].shift(-1) - df_combined['price']) / df_combined['price'] > 0).astype(int)

        # Return a copy of the combined DataFrame
        df_copy = df_combined.copy()
        return df_copy

    except:
        pass


async def process_symbol(ticker, con, start_date, end_date):
    try:
        test_size = 0.4
        start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        predictor = FundamentalPredictor()
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


#Train mode
async def train_process(tickers, con):
    tickers = list(set(tickers))
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    test_size = 0.4
    start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    predictor = FundamentalPredictor()
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    
    tasks = [download_data(ticker, con, start_date, end_date) for ticker in tickers]
    dfs = await asyncio.gather(*tasks)
    for df in dfs:
        try:
            split_size = int(len(df) * (1-test_size))
            train_data = df.iloc[:split_size]
            test_data = df.iloc[split_size:]
            df_train = pd.concat([df_train, train_data], ignore_index=True)
            df_test = pd.concat([df_test, test_data], ignore_index=True)
        except:
            pass

    
    best_features = [col for col in df_train.columns if col not in ['date','price','Target']]

    df_train = df_train.sample(frac=1).reset_index(drop=True)
    print('======Train Set Datapoints======')
    print(len(df_train))
    #selected_features = predictor.feature_selection(df_train[best_features], df_train['Target'],k=10)
    #print(selected_features)
    #selected_features = [col for col in df_train if col not in ['price','date','Target']]
    selected_features = ['shortTermCoverageRatios','netProfitMargin','debtRepayment','totalDebt','interestIncome','researchAndDevelopmentExpenses','priceEarningsToGrowthRatio','priceCashFlowRatio','cashPerShare','debtRatio','growthRevenue','revenue','growthNetIncome','ebitda','priceEarningsRatio','priceToBookRatio','epsdiluted','priceToSalesRatio','growthOtherCurrentLiabilities', 'receivablesTurnover', 'totalLiabilitiesAndStockholdersEquity', 'totalLiabilitiesAndTotalEquity', 'totalAssets', 'growthOtherCurrentAssets', 'retainedEarnings', 'totalEquity']

    predictor.train_model(df_train[selected_features], df_train['Target'])
    predictor.evaluate_model(df_test[selected_features], df_test['Target'])

async def test_process(con):
    test_size = 0.4
    start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    predictor = FundamentalPredictor()
    df = await download_data('GME', con, start_date, end_date)
    split_size = int(len(df) * (1-test_size))
    test_data = df.iloc[split_size:]
    #selected_features = [col for col in test_data if col not in ['price','date','Target']]
    selected_features = ['shortTermCoverageRatios','netProfitMargin','debtRepayment','totalDebt','interestIncome','researchAndDevelopmentExpenses','priceEarningsToGrowthRatio','priceCashFlowRatio','cashPerShare','debtRatio','growthRevenue','revenue','growthNetIncome','ebitda','priceEarningsRatio','priceToBookRatio','epsdiluted','priceToSalesRatio','growthOtherCurrentLiabilities', 'receivablesTurnover', 'totalLiabilitiesAndStockholdersEquity', 'totalLiabilitiesAndTotalEquity', 'totalAssets', 'growthOtherCurrentAssets', 'retainedEarnings', 'totalEquity']
    predictor.evaluate_model(test_data[selected_features], test_data['Target'])


async def run():

    #Train first model
    
    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 300E9")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    print('Number of Stocks')
    print(len(stock_symbols))
    await train_process(stock_symbols, con)


    #Prediction Steps for all stock symbols
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 1E9")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    total_symbols = stock_symbols
    
    print(f"Total tickers: {len(total_symbols)}")
    start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")

    chunk_size = len(total_symbols) // 100  # Divide the list into N chunks
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

