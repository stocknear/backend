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

from ta.momentum import *
from ta.trend import *
from ta.volatility import *
from ta.volume import *
import gc
#Enable automatic garbage collection
gc.enable()

async def save_json(symbol, data):
    with open(f"json/ai-score/companies/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))


def trend_intensity(close, window=20):
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    return ((close - ma) / std).abs().rolling(window=window).mean()


def calculate_fdi(high, low, close, window=30):
    n1 = (np.log(high.rolling(window=window).max() - low.rolling(window=window).min()) -
          np.log(close.rolling(window=window).max() - close.rolling(window=window).min())) / np.log(2)
    return (2 - n1) * 100




async def download_data(ticker, con, start_date, end_date):
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

        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['sma_crossover'] = ((df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)

        df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
        df['ema_crossover'] = ((df['ema_50'] > df['ema_200']) & (df['ema_50'].shift(1) <= df['ema_200'].shift(1))).astype(int)

        ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        bb = BollingerBands(close=df['close'])
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']

        df['volatility'] = df['close'].rolling(window=30).std()
        df['daily_return'] = df['close'].pct_change()
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        df['volume_change'] = df['volume'].pct_change()
        df['roc'] = df['close'].pct_change(periods=60)
        df['avg_volume'] = df['volume'].rolling(window=60).mean()
        df['drawdown'] = df['close'] / df['close'].rolling(window=252).max() - 1


        df['macd'] = macd(df['close'])
        df['macd_signal'] = macd_signal(df['close'])
        df['macd_hist'] = 2*macd_diff(df['close'])
        df['adx'] = adx(df['high'],df['low'],df['close'])
        df["adx_pos"] = adx_pos(df['high'],df['low'],df['close'])
        df["adx_neg"] = adx_neg(df['high'],df['low'],df['close'])
        df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
        df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
        
        df['nvi'] = NegativeVolumeIndexIndicator(close=df['close'], volume=df['volume']).negative_volume_index()
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        df['vpt'] = VolumePriceTrendIndicator(close=df['close'], volume=df['volume']).volume_price_trend()
        
        df['rsi'] = rsi(df["close"], window=60)
        df['rolling_rsi'] = df['rsi'].rolling(window=10).mean()
        df['stoch_rsi'] = stochrsi_k(df['close'], window=60, smooth1=3, smooth2=3)
        df['rolling_stoch_rsi'] = df['stoch_rsi'].rolling(window=10).mean()

        df['adi'] = acc_dist_index(high=df['high'],low=df['low'],close=df['close'],volume=df['volume'])
        df['cmf'] = chaikin_money_flow(high=df['high'],low=df['low'],close=df['close'],volume=df['volume'], window=20)
        df['emv'] = ease_of_movement(high=df['high'],low=df['low'],volume=df['volume'], window=20)
        df['fi'] = force_index(close=df['close'], volume=df['volume'], window= 13)

        df['williams'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
        df['kama'] = KAMAIndicator(close=df['close']).kama()

        df['stoch'] = stoch(df['high'], df['low'], df['close'], window=30)
        df['rocr'] = df['close'] / df['close'].shift(30) - 1 # Rate of Change Ratio (ROCR)
        df['ppo'] = (df['ema_50'] - df['ema_200']) / df['ema_50'] * 100
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['volatility_ratio'] = df['close'].rolling(window=30).std() / df['close'].rolling(window=60).std()

        df['fdi'] = calculate_fdi(df['high'], df['low'], df['close'])
        df['tii'] = trend_intensity(df['close'])


        ta_indicators = [
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'adx', 'adx_pos', 'adx_neg',
            'cci', 'mfi', 'nvi', 'obv', 'vpt', 'stoch_rsi','bb_width',
            'adi', 'cmf', 'emv', 'fi', 'williams', 'stoch','sma_crossover',
            'volatility','daily_return','cumulative_return', 'roc','avg_volume',
            'rolling_rsi','rolling_stoch_rsi', 'ema_crossover','ichimoku_a','ichimoku_b',
            'atr','kama','rocr','ppo','volatility_ratio','vwap','tii','fdi','drawdown',
            'volume_change'
        ]

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
            
            for indicator in ta_indicators:
                indicator_value = df[df['date'] == target_date][indicator].values[0]
                item[indicator] = indicator_value  # Add the indicator value to the combined_data entry
            

        # Sort the combined data by date
        combined_data = sorted(combined_data, key=lambda x: x['date'])
        # Convert combined data into a DataFrame
        df_combined = pd.DataFrame(combined_data).dropna()

        
        key_elements = [
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
        for num, denom in combinations(key_elements, 2):
            # Compute ratio and reverse ratio
            ratio = df_combined[num] / df_combined[denom]
            reverse_ratio = df_combined[denom] / df_combined[num]

            # Define column names for both ratios
            column_name = f'{num}_to_{denom}'
            reverse_column_name = f'{denom}_to_{num}'

            # Store the new columns in the dictionary, replacing invalid values with 0
            new_columns[column_name] = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
            new_columns[reverse_column_name] = np.nan_to_num(reverse_ratio, nan=0, posinf=0, neginf=0)

        # Add all new columns to the original DataFrame at once
        df_combined = pd.concat([df_combined, pd.DataFrame(new_columns)], axis=1)
        

        # To defragment the DataFrame, make a copy
        df_combined = df_combined.copy()

        
        # Create 'Target' column based on price change
        df_combined['Target'] = ((df_combined['price'].shift(-1) - df_combined['price']) / df_combined['price'] > 0).astype(int)

        # Return a copy of the combined DataFrame
        df_combined = df_combined.dropna()
        df_combined = df_combined.where(~df_combined.isin([np.inf, -np.inf]), 0)
        df_copy = df_combined.copy()

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

    dfs = await chunked_gather(tickers, con, start_date, end_date, chunk_size=10)

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
    
    print('======Warm Start Train Set Datapoints======')
    df_train = df_train.sample(frac=1).reset_index(drop=True) #df_train.reset_index(drop=True)
    print(len(df_train))
    
    predictor = ScorePredictor()
    selected_features = [col for col in df_train if col not in ['price', 'date', 'Target']]
    predictor.warm_start_training(df_train[selected_features], df_train['Target'])
    predictor.evaluate_model(df_test[selected_features], df_test['Target'])

    return predictor

async def fine_tune_and_evaluate(ticker, con, start_date, end_date):
    try:
        df = await download_data(ticker, con, start_date, end_date)
        if df is None or len(df) == 0:
            print(f"No data available for {ticker}")
            return
        
        test_size = 0.2
        split_size = int(len(df) * (1-test_size))
        train_data = df.iloc[:split_size]
        test_data = df.iloc[split_size:]
        
        selected_features = [col for col in df.columns if col not in ['date','price','Target']]
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
    train_mode = False  # Set this to False for fine-tuning and evaluation
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    
    if train_mode:
        # Warm start training
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 10E9 AND symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        warm_start_symbols = [row[0] for row in cursor.fetchall()]
        print('Warm Start Training for:', warm_start_symbols)
        predictor = await warm_start_training(warm_start_symbols, con)
    else:
        # Fine-tuning and evaluation for all stocks
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 1E9 AND symbol NOT LIKE '%.%'")
        stock_symbols = [row[0] for row in cursor.fetchall()]
        
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
