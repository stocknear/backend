from ta.momentum import *
from ta.trend import *
from ta.volatility import *
from ta.volume import *

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
import asyncio
import time

class TrendPredictor:
    def __init__(self, nth_day, path="ml_models/weights/ai_score"):
        self.model = RandomForestClassifier(n_estimators=1000, max_depth=500, min_samples_split=500, random_state=42, n_jobs=-1)
        self.scaler = MinMaxScaler()
        self.nth_day = nth_day
        self.path = path
        self.model_loaded = False

    def generate_features(self, df):
        new_predictors = []

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

        df['rsi'] = rsi(df["close"], window=14)
        df['stoch_rsi'] = stochrsi_k(df['close'], window=14, smooth1=3, smooth2=3)
        df['bb_hband'] = bollinger_hband(df['close'], window=14)/df['close']
        df['bb_lband'] = bollinger_lband(df['close'], window=14)/df['close']

        df['adi'] = acc_dist_index(high=df['high'],low=df['low'],close=df['close'],volume=df['volume'])
        df['cmf'] = chaikin_money_flow(high=df['high'],low=df['low'],close=df['close'],volume=df['volume'], window=20)
        df['emv'] = ease_of_movement(high=df['high'],low=df['low'],volume=df['volume'], window=20)
        df['fi'] = force_index(close=df['close'], volume=df['volume'], window= 13)

        df['williams'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()


        df['stoch'] = stoch(df['high'], df['low'], df['close'], window=14)

        new_predictors+=['williams','fi','emv','cmf','adi','bb_hband','bb_lband','vpt','stoch','stoch_rsi','rsi','nvi','obv','macd','macd_signal','macd_hist','adx','adx_pos','adx_neg','cci','mfi']
        return new_predictors

    def train_model(self, X_train, y_train):
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_train = np.nan_to_num(X_train)
        X_train = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train, y_train)
        with open(f'{self.path}/weights.pkl', 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        if not self.model_loaded:
            with open(f'{self.path}/weights.pkl', 'rb') as f:
                self.model = pickle.load(f)
            self.model_loaded = True

    def alpha_to_score(self, alpha):
        # Convert alpha (Target) to AI Score
        if alpha <= -20:
            return 1  # Very Low Alpha
        elif -20 < alpha <= -10:
            return 2  # Low Alpha
        elif -10 < alpha <= -5:
            return 3  # Low Alpha
        elif -5 < alpha <= 0:
            return 4  # Medium Alpha
        elif 0 < alpha <= 2:
            return 5  # Medium Alpha
        elif 2 < alpha <= 4:
            return 6  # High Alpha
        elif 4 < alpha <= 6:
            return 7  # High Alpha
        elif 6 < alpha <= 8:
            return 8  # High Alpha
        elif 8 < alpha <= 10:
            return 9  # High Alpha
        elif 10 < alpha:
            return 10  # Very High Alpha
        else:
        	return None

    def predict_and_score(self, df):
        self.load_model()  # Ensure model is loaded once

        latest_data = df.iloc[-1].values.reshape(1, -1)
        latest_data = self.scaler.fit_transform(latest_data)

        # Predict the class (AI score)
        prediction = self.model.predict(latest_data)[0]

        # Return structured result with ticker information and score
        print(f"Predicted AI Score: {prediction}")
        return prediction

    def evaluate_model(self, X_test, y_test):
        self.load_model()
        X_test = np.where(np.isinf(X_test), np.nan, X_test)
        X_test = np.nan_to_num(X_test)
        X_test = self.scaler.transform(X_test)

        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_report(y_test, predictions))

        return accuracy

async def download_data(ticker, start_date, end_date, spy_df, nth_day):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    df = df.rename(columns={'Adj Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})

    df = df.reindex(spy_df.index)
    df['spy_close'] = spy_df['spy_close']
    df['stock_return'] = df['close'].pct_change()
    df['spy_return'] = df['spy_close'].pct_change()
    df['excess_return'] = df['stock_return'] - df['spy_return']

    df["Target"] = df['excess_return'].rolling(window=nth_day).sum().shift(-nth_day)*100
    # Convert the continuous Target (alpha) to a score (class)
    df["Target"] = df["Target"].apply(lambda x: TrendPredictor.alpha_to_score(self=None, alpha=x))
    return df



async def train_process(nth_day):
    tickers = ['KO','WMT','BA','PLD','AZN','LLY','INFN','GRMN','VVX','EPD','PII','WY','BLMN','AAP','ON','TGT','SMG','EL','EOG','ULTA','DV','PLNT','GLOB','LKQ','CWH','PSX','SO','TGT','GD','MU','NKE','AMGN','BX','CAT','PEP','LIN','ABBV','COST','MRK','HD','JNJ','PG','SPCB','CVX','SHEL','MS','GS','MA','V','JPM','XLF','DPZ','CMG','MCD','ALTM','PDD','MNST','SBUX','AMAT','ZS','IBM','SMCI','ORCL','XLK','VUG','VTI','VOO','IWM','IEFA','PEP','WMT','XOM','V','AVGO','BIDU','GOOGL','SNAP','DASH','SPOT','NVO','META','MSFT','ADBE','DIA','PFE','BAC','RIVN','NIO','CISS','INTC','AAPL','BYND','MSFT','HOOD','MARA','SHOP','CRM','PYPL','UBER','SAVE','QQQ','IVV','SPY','EVOK','GME','F','NVDA','AMD','AMZN','TSM','TSLA']
    tickers = list(set(tickers))
    #print(len(tickers))

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    best_features = ['close','williams','fi','emv','adi','cmf','bb_hband','bb_lband','vpt','stoch','stoch_rsi','rsi','nvi','macd','mfi','cci','obv','adx','adx_pos','adx_neg']
    test_size = 0.1
    start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    predictor = TrendPredictor(nth_day=nth_day)
    
    spy_df = yf.download("SPY", start=start_date, end=end_date, interval="1d")
    spy_df = spy_df.rename(columns={'Adj Close': 'spy_close'})

    tasks = [download_data(ticker, start_date, end_date, spy_df, nth_day) for ticker in tickers]
    dfs = await asyncio.gather(*tasks)
    for df in dfs:
        try:
            predictors = predictor.generate_features(df)
            predictors = [pred for pred in predictors if pred in df.columns]
            df = df.dropna(subset=df.columns[df.columns != "nth_day"])
            split_size = int(len(df) * (1-test_size))
            train_data = df.iloc[:split_size]
            test_data = df.iloc[split_size:]
            df_train = pd.concat([df_train, train_data], ignore_index=True)
            df_test = pd.concat([df_test, test_data], ignore_index=True)
        except:
            pass


    df_train = df_train.sample(frac=1).reset_index(drop=True)

    predictor.train_model(df_train[best_features], df_train['Target'])
    predictor.evaluate_model(df_test[best_features], df_test['Target'])

async def test_process(nth_day):
    best_features = ['close','williams','fi','emv','adi','cmf','bb_hband','bb_lband','vpt','stoch','stoch_rsi','rsi','nvi','macd','mfi','cci','obv','adx','adx_pos','adx_neg']
    test_size = 0.1
    start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    predictor = TrendPredictor(nth_day=nth_day)

    spy_df = yf.download("SPY", start=start_date, end=end_date, interval="1d")
    spy_df = spy_df.rename(columns={'Adj Close': 'spy_close'})

    df = await download_data('AAPL', start_date, end_date, spy_df, nth_day)
    predictors = predictor.generate_features(df)
    
    #save it to get the latest date with the latest row otherwise it drops it since of NaN for Target
    df_copy = df.copy()
    
    df = df.dropna(subset=df.columns[df.columns != "nth_day"])
    split_size = int(len(df) * (1-test_size))
    test_data = df.iloc[split_size:]
    predictor.evaluate_model(test_data[best_features], test_data['Target'])

    #Evaluate based on non-nan results of target but predict the latest date
    predictor.predict_and_score(df_copy[best_features])
    print(df_copy)

async def main():
    nth_day = 60  # 60 days forward prediction

    await train_process(nth_day = 60)
    #await test_process(nth_day = 60)

if __name__ == "__main__":
    asyncio.run(main())
