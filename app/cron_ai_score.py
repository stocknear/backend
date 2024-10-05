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

from dotenv import load_dotenv
import os
import gc
from utils.feature_engineering import *
#Enable automatic garbage collection
gc.enable()



load_dotenv()
api_key = os.getenv('FMP_API_KEY')


async def save_json(symbol, data):
    with open(f"json/ai-score/companies/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))

async def fetch_historical_price(ticker):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from=1995-10-10&apikey={api_key}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            # Check if the request was successful
            if response.status == 200:
                data = await response.json()
                # Extract historical price data
                historical_data = data.get('historical', [])
                # Convert to DataFrame
                df = pd.DataFrame(historical_data).reset_index(drop=True)
                return df
            else:
                raise Exception(f"Error fetching data: {response.status} {response.reason}")


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

async def download_data(ticker, con, start_date, end_date, skip_downloading):

    file_path = f"ml_models/training_data/ai-score/{ticker}.json"

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pd.DataFrame(orjson.loads(file.read()))
    elif skip_downloading == False:

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

            # Async loading and filtering
            ignore_keys = ["symbol", "reportedCurrency", "calendarYear", "fillingDate", "acceptedDate", "period", "cik", "link", "finalLink","pbRatio","ptbRatio"]
            async def load_and_filter_json(path):
                async with aiofiles.open(path, 'r') as f:
                    data = orjson.loads(await f.read())
                return [{k: v for k, v in item.items() if k not in ignore_keys and int(item["date"][:4]) >= 2000} for item in data]

            # Load all files concurrently
            data = await asyncio.gather(*(load_and_filter_json(s) for s in statements))
            ratios, key_metrics, cashflow, income, balance, income_growth, balance_growth, cashflow_growth, owner_earnings = data

            #Threshold of enough datapoints needed!
            if len(ratios) < 50:
                return


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
            df = await fetch_historical_price(ticker)

            # Get the list of columns in df
            df_columns = df.columns
            df_stats = generate_statistical_features(df)
            df_ta = generate_ta_features(df)

            # Filter columns in df_stats and df_ta that are not in df
            # Drop unnecessary columns from df_stats and df_ta
            df_stats_filtered = df_stats.drop(columns=df_columns.intersection(df_stats.columns), errors='ignore')
            df_ta_filtered = df_ta.drop(columns=df_columns.intersection(df_ta.columns), errors='ignore')

            # Extract the column names for indicators
            ta_columns = df_ta_filtered.columns.tolist()
            stats_columns = df_stats_filtered.columns.tolist()

            # Concatenate df with the filtered df_stats and df_ta
            df = pd.concat([df, df_ta_filtered, df_stats_filtered], axis=1)

            # Set up a dictionary for faster lookup of close prices and columns by date
            df_dict = df.set_index('date').to_dict(orient='index')

            # Helper function to find closest date within max_attempts
            def find_closest_date(target_date, max_attempts=10):
                counter = 0
                while target_date not in df_dict and counter < max_attempts:
                    target_date = (pd.to_datetime(target_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    counter += 1
                return target_date if target_date in df_dict else None

            # Match combined data entries with stock data
            for item in combined_data:
                target_date = item['date']
                closest_date = find_closest_date(target_date)

                # Skip if no matching date is found
                if not closest_date:
                    continue

                # Fetch data from the dictionary for the closest matching date
                data = df_dict[closest_date]

                # Add close price to the item
                item['price'] = round(data['close'], 2)

                # Dynamically add indicator values from ta_columns and stats_columns
                for column in ta_columns + stats_columns:
                    item[column] = data.get(column, None)

            # Sort the combined data by date
            combined_data = sorted(combined_data, key=lambda x: x['date'])

            # Convert combined data to a DataFrame and drop rows with NaN values
            df_combined = pd.DataFrame(combined_data).dropna()

            
            fundamental_columns = [
                'revenue', 'costOfRevenue', 'grossProfit', 'netIncome', 'operatingIncome', 'operatingExpenses',
                'researchAndDevelopmentExpenses', 'ebitda', 'freeCashFlow', 'incomeBeforeTax', 'incomeTaxExpense',
                'debtRepayment', 'dividendsPaid', 'depreciationAndAmortization', 'netCashUsedProvidedByFinancingActivities',
                'changeInWorkingCapital', 'stockBasedCompensation', 'deferredIncomeTax', 'commonStockRepurchased',
                'operatingCashFlow', 'capitalExpenditure', 'accountsReceivables', 'purchasesOfInvestments',
                'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'longTermInvestments',
                'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'totalDebt', 'netDebt', 'commonStock',
                'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'totalStockholdersEquity', 'totalInvestments',
                'taxAssets', 'totalAssets', 'inventory', 'propertyPlantEquipmentNet', 'ownersEarnings',
            ]

            # Function to compute combinations within a group
            def compute_column_ratios(columns, df, new_columns):
                column_combinations = list(combinations(columns, 2))
                
                for num, denom in column_combinations:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        # Compute ratio and reverse ratio safely
                        ratio = df[num] / df[denom]
                        reverse_ratio = df[denom] / df[num]

                    # Define column names for both ratios
                    column_name = f'{num}_to_{denom}'
                    reverse_column_name = f'{denom}_to_{num}'

                    # Assign values to new columns, handling invalid values
                    new_columns[column_name] = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
                    new_columns[reverse_column_name] = np.nan_to_num(reverse_ratio, nan=0, posinf=0, neginf=0)

            # Create an empty dictionary for the new columns
            new_columns = {}

            # Compute combinations for each group of columns
            compute_column_ratios(fundamental_columns, df_combined, new_columns)
            compute_column_ratios(stats_columns, df_combined, new_columns)
            compute_column_ratios(ta_columns, df_combined, new_columns)

            # Concatenate the new ratio columns with the original DataFrame
            df_combined = pd.concat([df_combined, pd.DataFrame(new_columns, index=df_combined.index)], axis=1)

            # Clean up and replace invalid values
            df_combined = df_combined.replace([np.inf, -np.inf], 0).dropna()

            # Create 'Target' column to indicate if the next price is higher than the current one
            df_combined['Target'] = ((df_combined['price'].shift(-1) - df_combined['price']) / df_combined['price'] > 0).astype(int)

            # Copy DataFrame and round float values
            df_copy = df_combined.copy().map(lambda x: round(x, 2) if isinstance(x, float) else x)

            # Save to a file if there are rows in the DataFrame
            if not df_copy.empty:
                with open(file_path, 'wb') as file:
                    file.write(orjson.dumps(df_copy.to_dict(orient='records')))

            return df_copy

        except Exception as e:
            print(e)
            pass


async def chunked_gather(tickers, con, start_date, end_date, skip_downloading, chunk_size=10):
    # Helper function to divide the tickers into chunks
    def chunks(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i+size]
    
    results = []
    
    for chunk in tqdm(chunks(tickers, chunk_size)):
        # Create tasks for each chunk
        tasks = [download_data(ticker, con, start_date, end_date, skip_downloading) for ticker in chunk]
        # Await the results for the current chunk
        chunk_results = await asyncio.gather(*tasks)
        # Accumulate the results
        results.extend(chunk_results)
    
    return results

async def warm_start_training(tickers, con, skip_downloading):
    start_date = datetime(1995, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    test_size = 0.2

    dfs = await chunked_gather(tickers, con, start_date, end_date, skip_downloading, chunk_size=10)

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
    #predictor.warm_start_training(df_train[selected_features], df_train['Target'])
    predictor.batch_train_model(df_train[selected_features], df_train['Target'], batch_size=1000)
    predictor.evaluate_model(df_test[selected_features], df_test['Target'])

    return predictor

async def fine_tune_and_evaluate(ticker, con, start_date, end_date, skip_downloading):
    try:
        df = await download_data(ticker,con, start_date, end_date, skip_downloading)
        if df is None or len(df) == 0:
            print(f"No data available for {ticker}")
            return
        
        test_size = 0.2
        split_size = int(len(df) * (1-test_size))
        train_data = df.iloc[:split_size]
        test_data = df.iloc[split_size:]
        
        selected_features = [col for col in train_data if col not in ['price', 'date', 'Target']] #top_uncorrelated_features(train_data,top_n=20)
        # Fine-tune the model
        predictor = ScorePredictor()
        predictor.fine_tune_model(train_data[selected_features], train_data['Target'])
        
        print(f"Evaluating fine-tuned model for {ticker}")
        data = predictor.evaluate_model(test_data[selected_features], test_data['Target'])
        
        if len(data) != 0:
            if data['precision'] >= 50 and data['accuracy'] >= 50 and data['accuracy'] < 100 and data['precision'] < 100 and data['f1_score'] > 50 and data['recall_score'] > 50 and data['roc_auc_score'] > 50:
                await save_json(ticker, data)
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
    skip_downloading = False
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    
    if train_mode:
        # Warm start training
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 1E9 AND symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        warm_start_symbols = [row[0] for row in cursor.fetchall()]
        print(f'Warm Start Training: {len(warm_start_symbols)}')
        predictor = await warm_start_training(warm_start_symbols, con, skip_downloading)
    else:
        # Fine-tuning and evaluation for all stocks
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 500E6 AND symbol NOT LIKE '%.%'")
        stock_symbols = [row[0] for row in cursor.fetchall()]
        
        print(f"Total tickers for fine-tuning: {len(stock_symbols)}")
        start_date = datetime(1995, 1, 1).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")

        tasks = []
        for ticker in tqdm(stock_symbols):
            await fine_tune_and_evaluate(ticker, con, start_date, end_date, skip_downloading)
            
    con.close()

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(f"Main execution error: {e}")
