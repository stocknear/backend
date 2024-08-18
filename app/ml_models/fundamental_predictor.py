import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, Bidirectional, Attention,Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.backend import clear_session
from keras import regularizers

from tqdm import tqdm
from collections import defaultdict
import asyncio
import aiohttp
import pickle
import time
import sqlite3
import ujson


#Based on the paper: https://arxiv.org/pdf/1603.00751


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
        '''
        if len(income) < 40:
            raise ValueError("Income data length is too small.")
        '''

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

        
        df_combined = pd.DataFrame(combined_data).dropna()

        df_combined['Target'] = ((df_combined['price'].shift(-1) - df_combined['price']) / df_combined['price'] > 0).astype(int)

        df_copy = df_combined.copy()
        
        return df_copy

    except Exception as e:
        print(e)


class FundamentalPredictor:
    def __init__(self, path='weights'):
        self.model = self.build_model() #RandomForestClassifier(n_estimators=1000, max_depth = 20, min_samples_split=10, random_state=42, n_jobs=10)
        self.scaler = MinMaxScaler()
        self.path = path

    def build_model(self):
        clear_session()
        model = Sequential()
        
        model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(None, 1)))

        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(None, 1)))

        # First LSTM layer with dropout and batch normalization
        model.add(LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        
        # Second LSTM layer with dropout and batch normalization
        model.add(LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        
        # Third LSTM layer with dropout and batch normalization
        model.add(LSTM(128, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        # Dense layer with sigmoid activation for binary classification
        model.add(Dense(1, activation='sigmoid'))


        # Adam optimizer with a learning rate of 0.001
        optimizer = Adam(learning_rate=0.01)
        
        # Compile model with binary crossentropy loss and accuracy metric
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def preprocess_data(self, X):
        #X = X.applymap(lambda x: 9999 if x == 0 else x) # Replace 0 with 9999 as suggested in the paper
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X)
        X = self.scaler.fit_transform(X)
        return X

    def reshape_for_lstm(self, X):
        return X.reshape((X.shape[0], X.shape[1], 1))

    def train_model(self, X_train, y_train):
        X_train = self.preprocess_data(X_train)
        X_train = self.reshape_for_lstm(X_train)
        
        checkpoint = ModelCheckpoint(f'{self.path}/fundamental_weights/weights.keras', save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.model.fit(X_train, y_train, epochs=250, batch_size=32, validation_split=0.2, callbacks=[checkpoint, early_stopping])
        self.model.save(f'{self.path}/fundamental_weights/weights.keras')

    def evaluate_model(self, X_test, y_test):
        X_test = self.preprocess_data(X_test)
        X_test = self.reshape_for_lstm(X_test)
        
        self.model = self.build_model()
        self.model = load_model(f'{self.path}/fundamental_weights/weights.keras')
        
        test_predictions = self.model.predict(X_test).flatten()
        
        test_predictions[test_predictions >= 0.5] = 1
        test_predictions[test_predictions < 0.5] = 0
        
        test_precision = precision_score(y_test, test_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print("Test Set Metrics:")
        print(f"Precision: {round(test_precision * 100)}%")
        print(f"Accuracy: {round(test_accuracy * 100)}%")
        
        next_value_prediction = 1 if test_predictions[-1] >= 0.5 else 0
        return {'accuracy': round(test_accuracy*100), 'precision': round(test_precision*100), 'sentiment': 'Bullish' if next_value_prediction == 1 else 'Bearish'}, test_predictions

    def feature_selection(self, X_train, y_train,k=8):
        
        selector = SelectKBest(score_func=f_classif, k=8)
        selector.fit(X_train, y_train)

        selector.transform(X_train)
        selected_features = [col for i, col in enumerate(X_train.columns) if selector.get_support()[i]]

        return selected_features
        
        # Calculate the variance of each feature with respect to the target
        '''
        variances = {}
        for col in X_train.columns:
            grouped_variance = X_train.groupby(y_train)[col].var().mean()
            variances[col] = grouped_variance

        # Sort features by variance and select top k features
        sorted_features = sorted(variances, key=variances.get, reverse=True)[:k]
        return sorted_features
        '''

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

async def main():
    con = sqlite3.connect('../stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE marketCap >= 500E9")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    print('Number of Stocks')
    print(len(stock_symbols))
    await train_process(stock_symbols, con)
    await test_process(con)

    con.close()

# Run the main function
#asyncio.run(main())