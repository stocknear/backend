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
from itertools import combinations

from ta.momentum import *
from ta.trend import *
from ta.volatility import *
from ta.volume import *
import gc
#Enable automatic garbage collection
gc.enable()

async def save_json(symbol, data):
    with open(f"json/fundamental-predictor-analysis/{symbol}.json", 'wb') as file:
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
            f"json/financial-statements/cash-flow-statement-growth/quarter/{ticker}.json",
            f"json/financial-statements/key-metrics/quarter/{ticker}.json",
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
        
        cashflow = await load_json_from_file(statements[1])
        cashflow = await filter_data(cashflow, ignore_keys)

        income = await load_json_from_file(statements[2])
        income = await filter_data(income, ignore_keys)

        balance = await load_json_from_file(statements[3])
        balance = await filter_data(balance, ignore_keys)

        income_growth = await load_json_from_file(statements[4])
        income_growth = await filter_data(income_growth, ignore_keys)

        balance_growth = await load_json_from_file(statements[5])
        balance_growth = await filter_data(balance_growth, ignore_keys)


        cashflow_growth = await load_json_from_file(statements[6])
        cashflow_growth = await filter_data(cashflow_growth, ignore_keys)

        key_metrics = await load_json_from_file(statements[7])
        key_metrics = await filter_data(key_metrics, ignore_keys)

        owner_earnings = await load_json_from_file(statements[8])
        owner_earnings = await filter_data(owner_earnings, ignore_keys)


        # Combine all the data
        combined_data = defaultdict(dict)

        # Merge the data based on 'date'
        for entries in zip(income, income_growth, balance, balance_growth, cashflow, cashflow_growth, ratios, key_metrics, owner_earnings):
            for entry in entries:
                date = entry['date']
                for key, value in entry.items():
                    if key not in combined_data[date]:
                        combined_data[date][key] = value

        combined_data = list(combined_data.values())
        #Generate more features
        #combined_data = calculate_combinations(combined_data)

        # Download historical stock data using yfinance
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d").reset_index()
        df = df.rename(columns={'Adj Close': 'close', 'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['golden_cross'] = ((df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)

        df['volatility'] = df['close'].rolling(window=30).std()
        df['daily_return'] = df['close'].pct_change()
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        df['volume_change'] = df['volume'].pct_change()
        df['roc'] = df['close'].pct_change(periods=30) * 100  # 12-day ROC
        df['avg_volume_30d'] = df['volume'].rolling(window=30).mean()
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
        
        df['rsi'] = rsi(df["close"], window=30)
        df['rolling_rsi'] = df['rsi'].rolling(window=10).mean()
        df['stoch_rsi'] = stochrsi_k(df['close'], window=30, smooth1=3, smooth2=3)
        df['rolling_stoch_rsi'] = df['stoch_rsi'].rolling(window=10).mean()
        df['bb_hband'] = bollinger_hband(df['close'], window=30)/df['close']
        df['bb_lband'] = bollinger_lband(df['close'], window=30)/df['close']

        df['adi'] = acc_dist_index(high=df['high'],low=df['low'],close=df['close'],volume=df['volume'])
        df['cmf'] = chaikin_money_flow(high=df['high'],low=df['low'],close=df['close'],volume=df['volume'], window=20)
        df['emv'] = ease_of_movement(high=df['high'],low=df['low'],volume=df['volume'], window=20)
        df['fi'] = force_index(close=df['close'], volume=df['volume'], window= 13)

        df['williams'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()


        df['stoch'] = stoch(df['high'], df['low'], df['close'], window=30)

        ta_indicators = [
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'adx', 'adx_pos', 'adx_neg',
            'cci', 'mfi', 'nvi', 'obv', 'vpt', 'stoch_rsi', 'bb_hband', 'bb_lband',
            'adi', 'cmf', 'emv', 'fi', 'williams', 'stoch','sma_50','sma_200','golden_cross',
            'volatility','daily_return','cumulative_return', 'roc','avg_volume_30d',
            'rolling_rsi','rolling_stoch_rsi'
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
            'epsdiluted',
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
            'averagePPE'

        ]

        # Compute ratios for all combinations of key elements
        for num, denom in combinations(key_elements, 2):
            # Compute ratio num/denom
            column_name = f'{num}_to_{denom}'
            try:
                df_combined[column_name] = df_combined[num] / df_combined[denom]
            except:
                df_combined[column_name] = 0
            # Compute reverse ratio denom/num
            reverse_column_name = f'{denom}_to_{num}'
            try:
                df_combined[reverse_column_name] = df_combined[denom] / df_combined[num]
            except:
                df_combined[reverse_column_name] = 0
    
        # Create 'Target' column based on price change
        df_combined['Target'] = ((df_combined['price'].shift(-1) - df_combined['price']) / df_combined['price'] > 0).astype(int)

        # Return a copy of the combined DataFrame
        df_copy = df_combined.copy()
        #print(df_copy[['date','revenue','ownersEarnings','revenuePerShare']])
        return df_copy

    except Exception as e:
        print(e)
        pass


async def process_symbol(ticker, con, start_date, end_date):
    try:
        test_size = 0.4
        start_date = datetime(1995, 1, 1).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        predictor = FundamentalPredictor()
        df = await download_data(ticker, con, start_date, end_date)
        split_size = int(len(df) * (1-test_size))
        test_data = df.iloc[split_size:]
        best_features = [col for col in df.columns if col not in ['date','price','Target']]
        data, prediction_list = predictor.evaluate_model(test_data[best_features], test_data['Target'])
        

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
    test_size = 0.2
    start_date = datetime(1995, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
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
    print(df_train)
    print(df_test)
    #selected_features = predictor.feature_selection(df_train[best_features], df_train['Target'],k=10)
    #print(selected_features)
    selected_features = [col for col in df_train if col not in ['price','date','Target']]

    predictor = FundamentalPredictor()
    predictor.train_model(df_train[selected_features], df_train['Target'])
    predictor.evaluate_model(df_test[selected_features], df_test['Target'])

async def test_process(con):
    test_size = 0.2
    start_date = datetime(1995, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    predictor = FundamentalPredictor()
    df = await download_data('GME', con, start_date, end_date)
    split_size = int(len(df) * (1-test_size))
    test_data = df.iloc[split_size:]
    selected_features = [col for col in test_data if col not in ['price','date','Target']]
    predictor.evaluate_model(test_data[selected_features], test_data['Target'])


async def run():

    #Train first model
    
    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 300E9")
    stock_symbols = ['AAPL'] #[row[0] for row in cursor.fetchall()] 
    print('Number of Stocks')
    print(len(stock_symbols))
    await train_process(stock_symbols, con)


    #Prediction Steps for all stock symbols
    
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 1E9")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    total_symbols = ['GME'] #stock_symbols
    
    print(f"Total tickers: {len(total_symbols)}")
    start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")

    chunk_size = len(total_symbols)# // 100  # Divide the list into N chunks
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

