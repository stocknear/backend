import orjson
import asyncio
import aiohttp
import aiofiles
import sqlite3
from datetime import datetime
from ml_models.score_model import ScorePredictor
import yfinance as yf
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import re
from itertools import combinations
import os
import gc
from utils.feature_engineering import *
#Enable automatic garbage collection
gc.enable()

async def save_json(symbol, data):
    with open(f"json/ai-score/companies/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))

def top_uncorrelated_features(df, target_col='Target', top_n=10, threshold=0.75):
    # Drop the columns to exclude from the DataFrame
    df_filtered = df.drop(columns=['date','price'])
    
    # Compute the correlation matrix
    correlation_matrix = df_filtered.corr()
    
    # Get the correlations with the target column, sorted by absolute value
    correlations_with_target = correlation_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
    
    # Initialize the list of selected features
    selected_features = []
    
    # Iteratively select the most correlated features while minimizing correlation with each other
    for feature in correlations_with_target.index:
        # If we already have enough features, break
        if len(selected_features) >= top_n:
            break
        
        # Check correlation of this feature with already selected features
        is_uncorrelated = True
        for selected in selected_features:
            if abs(correlation_matrix.loc[feature, selected]) > threshold:
                is_uncorrelated = False
                break
        
        # If it's uncorrelated with the selected features, add it to the list
        if is_uncorrelated:
            selected_features.append(feature)
    return selected_features

async def download_data(ticker, con, start_date, end_date):

    file_path = f"ml_models/training_data/ai-score/{ticker}.json"

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pd.DataFrame(orjson.loads(file.read()))
    else:

        try:
            # Define paths to the statement files
            statements = [
                f"json/financial-statements/ratios/quarter/{ticker}.json",
                f"json/financial-statements/key-metrics/quarter/{ticker}.json",
                f"json/financial-statements/cash-flow-statement/quarter/{ticker}.json",
                f"json/financial-statements/income-statement/quarter/{ticker}.json",
                f"json/financial-statements/balance-sheet-statement/quarter/{ticker}.json",
                f"json/financial-statements/income-statement-growth/quarter/{ticker}.json",
                f"json/financial-statements/balance-sheet-statement-growth/quarter/{ticker}.json",
                f"json/financial-statements/cash-flow-statement-growth/quarter/{ticker}.json",
                f"json/financial-statements/owner-earnings/quarter/{ticker}.json",
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
            ignore_keys = ["symbol", "reportedCurrency", "calendarYear", "fillingDate", "acceptedDate", "period", "cik", "link", "finalLink","pbRatio","ptbRatio"]

            # Load and filter data for each statement type

            ratios = await load_json_from_file(statements[0])
            ratios = await filter_data(ratios, ignore_keys)

            #Threshold of enough datapoints needed!
            if len(ratios) < 50:
                return

            key_metrics = await load_json_from_file(statements[1])
            key_metrics = await filter_data(key_metrics, ignore_keys)
            
            
            cashflow = await load_json_from_file(statements[2])
            cashflow = await filter_data(cashflow, ignore_keys)

            income = await load_json_from_file(statements[3])
            income = await filter_data(income, ignore_keys)

            balance = await load_json_from_file(statements[4])
            balance = await filter_data(balance, ignore_keys)
            
            income_growth = await load_json_from_file(statements[5])
            income_growth = await filter_data(income_growth, ignore_keys)

            balance_growth = await load_json_from_file(statements[6])
            balance_growth = await filter_data(balance_growth, ignore_keys)


            cashflow_growth = await load_json_from_file(statements[7])
            cashflow_growth = await filter_data(cashflow_growth, ignore_keys)

            owner_earnings = await load_json_from_file(statements[8])
            owner_earnings = await filter_data(owner_earnings, ignore_keys)


            # Combine all the data
            combined_data = defaultdict(dict)

            # Merge the data based on 'date'
            for entries in zip(ratios,key_metrics,income, balance, cashflow, owner_earnings, income_growth, balance_growth, cashflow_growth):
                for entry in entries:
                    date = entry['date']
                    for key, value in entry.items():
                        if key not in combined_data[date]:
                            combined_data[date][key] = value

            combined_data = list(combined_data.values())

            # Download historical stock data using yfinance
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d").reset_index()
            df = df.rename(columns={'Adj Close': 'close', 'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')


            # Get the list of columns in df
            df_columns = df.columns
            df_stats = generate_statistical_features(df)
            df_ta = generate_ta_features(df)

            # Filter columns in df_stats and df_ta that are not in df
            df_stats_filtered = df_stats.drop(columns=df_columns.intersection(df_stats.columns), errors='ignore')
            df_ta_filtered = df_ta.drop(columns=df_columns.intersection(df_ta.columns), errors='ignore')
            ta_columns = df_ta_filtered.columns.tolist()
            stats_columns = df_stats_filtered.columns.tolist()

            # Concatenate df with the filtered df_stats and df_ta
            df = pd.concat([df, df_ta_filtered, df_stats_filtered], axis=1)


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

                # Dynamically add all indicator values to the combined_data entry
                
                for column in ta_columns:
                    column_value = df[df['date'] == target_date][column].values[0]
                    item[column] = column_value  # Add the column value to the combined_data entry
                for column in stats_columns:
                    column_value = df[df['date'] == target_date][column].values[0]
                    item[column] = column_value  # Add the column value to the combined_data entry
                

            # Sort the combined data by date
            combined_data = sorted(combined_data, key=lambda x: x['date'])
            # Convert combined data into a DataFrame
            df_combined = pd.DataFrame(combined_data).dropna()
            '''
            fundamental_columns = [
                'revenue',
                'costOfRevenue',
                'grossProfit',
                'netIncome',
                'operatingIncome',
                'operatingExpenses',
                'researchAndDevelopmentExpenses',
                'ebitda',
                'freeCashFlow',
                'incomeBeforeTax',
                'incomeTaxExpense',
                'debtRepayment',
                'dividendsPaid',
                'depreciationAndAmortization',
                'netCashUsedProvidedByFinancingActivities',
                'changeInWorkingCapital',
                'stockBasedCompensation',
                'deferredIncomeTax',
                'commonStockRepurchased',
                'operatingCashFlow',
                'capitalExpenditure',
                'accountsReceivables',
                'purchasesOfInvestments',
                'cashAndCashEquivalents',
                'shortTermInvestments',
                'cashAndShortTermInvestments',
                'longTermInvestments',
                'otherCurrentLiabilities',
                'totalCurrentLiabilities',
                'longTermDebt',
                'totalDebt',
                'netDebt',
                'commonStock',
                'totalEquity',
                'totalLiabilitiesAndStockholdersEquity',
                'totalStockholdersEquity',
                'totalInvestments',
                'taxAssets',
                'totalAssets',
                'inventory',
                'propertyPlantEquipmentNet',
                'ownersEarnings',
            ]

            # Compute ratios for all combinations of key elements
            new_columns = {}
            # Loop over combinations of column pairs
            for columns in [fundamental_columns]:
                for num, denom in combinations(columns, 2):
                    # Compute ratio and reverse ratio
                    ratio = df_combined[num] / df_combined[denom]
                    reverse_ratio = round(df_combined[denom] / df_combined[num],2)

                    # Define column names for both ratios
                    column_name = f'{num}_to_{denom}'
                    reverse_column_name = f'{denom}_to_{num}'

                    # Store the new columns in the dictionary, replacing invalid values with 0
                    new_columns[column_name] = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
                    new_columns[reverse_column_name] = np.nan_to_num(reverse_ratio, nan=0, posinf=0, neginf=0)

                # Add all new columns to the original DataFrame at once
            df_combined = pd.concat([df_combined, pd.DataFrame(new_columns)], axis=1)
            '''
            # To defragment the DataFrame, make a copy
            df_combined = df_combined.copy()
            df_combined = df_combined.dropna()
            df_combined = df_combined.where(~df_combined.isin([np.inf, -np.inf]), 0)
            

            df_combined['Target'] = ((df_combined['price'].shift(-1) - df_combined['price']) / df_combined['price'] > 0).astype(int)

            df_copy = df_combined.copy()
            df_copy = df_copy.map(lambda x: round(x, 2) if isinstance(x, float) else x)

            if df_copy.shape[0] > 0:
                with open(file_path, 'wb') as file:
                    file.write(orjson.dumps(df_copy.to_dict(orient='records')))

            return df_copy

        except Exception as e:
            print(e)
            pass


async def chunked_gather(tickers, con, start_date, end_date, chunk_size=10):
    # Helper function to divide the tickers into chunks
    def chunks(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i+size]
    
    results = []
    
    for chunk in chunks(tickers, chunk_size):
        # Create tasks for each chunk
        tasks = [download_data(ticker, con, start_date, end_date) for ticker in chunk]
        # Await the results for the current chunk
        chunk_results = await asyncio.gather(*tasks)
        # Accumulate the results
        results.extend(chunk_results)
    
    return results

async def warm_start_training(tickers, con):
    start_date = datetime(1995, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    test_size = 0.2

    dfs = await chunked_gather(tickers, con, start_date, end_date, chunk_size=1)

    train_list = []
    test_list = []

    for df in dfs:
        try:
            split_size = int(len(df) * (1 - test_size))
            train_data = df.iloc[:split_size]
            test_data = df.iloc[split_size:]
            
            # Append to the lists
            train_list.append(train_data)
            test_list.append(test_data)
        except:
            pass

    # Concatenate all at once outside the loop
    df_train = pd.concat(train_list, ignore_index=True)
    df_test = pd.concat(test_list, ignore_index=True)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    print('======Warm Start Train Set Datapoints======')
    print(len(df_train))

    predictor = ScorePredictor()
    selected_features = [col for col in df_train if col not in ['price', 'date', 'Target']] #top_uncorrelated_features(df_train, top_n=200)
    predictor.warm_start_training(df_train[selected_features], df_train['Target'])
    predictor.evaluate_model(df_test[selected_features], df_test['Target'])

    return predictor

async def fine_tune_and_evaluate(ticker, con, start_date, end_date):
    try:
        df = await download_data(ticker,con, start_date, end_date)
        if df is None or len(df) == 0:
            print(f"No data available for {ticker}")
            return
        
        test_size = 0.2
        split_size = int(len(df) * (1-test_size))
        train_data = df.iloc[:split_size]
        test_data = df.iloc[split_size:]
        
        selected_features = top_uncorrelated_features(train_data,top_n=50) #[col for col in train_data if col not in ['price', 'date', 'Target']] #top_uncorrelated_features(train_data,top_n=20)
        # Fine-tune the model
        predictor = ScorePredictor()
        predictor.fine_tune_model(train_data[selected_features], train_data['Target'])
        
        print(f"Evaluating fine-tuned model for {ticker}")
        data = predictor.evaluate_model(test_data[selected_features], test_data['Target'])
        
        if len(data) != 0:
            if data['precision'] >= 50 and data['accuracy'] >= 50 and data['accuracy'] < 100 and data['precision'] < 100:
                res = {'score': data['score']}
                await save_json(ticker, res)
                print(f"Saved results for {ticker}")
        gc.collect()
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
    finally:
        # Ensure any remaining cleanup if necessary
        if 'predictor' in locals():
            del predictor  # Explicitly delete the predictor to aid garbage collection

async def run():
    train_mode = True  # Set this to False for fine-tuning and evaluation
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    
    if train_mode:
        # Warm start training
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 1E9 AND symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        warm_start_symbols = [row[0] for row in cursor.fetchall()]
        print('Warm Start Training for:', warm_start_symbols)
        predictor = await warm_start_training(warm_start_symbols, con)
    else:
        # Fine-tuning and evaluation for all stocks
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 1E9 AND symbol NOT LIKE '%.%'")
        stock_symbols = ['AWR'] #[row[0] for row in cursor.fetchall()]
        
        print(f"Total tickers for fine-tuning: {len(stock_symbols)}")
        start_date = datetime(1995, 1, 1).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")


        tasks = []
        for ticker in tqdm(stock_symbols):
            await fine_tune_and_evaluate(ticker, con, start_date, end_date)
            
    con.close()

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(f"Main execution error: {e}")
