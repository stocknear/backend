import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta.utils import *
from ta.volatility import *
from ta.momentum import *
from ta.trend import *
from ta.volume import *
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_classif
import asyncio
import aiohttp
import pickle
import time


async def download_data(ticker, start_date, end_date, nth_day):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        df = df.rename(columns={'Adj Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume', 'Date': 'date'})
        df["Target"] = ((df["close"].shift(-nth_day) > df["close"])).astype(int)
        df_copy = df.copy()
        if len(df_copy) > 252*2: #At least 2 years of history is necessary
            return df_copy
    except Exception as e:
        print(e)


class TrendPredictor:
    def __init__(self, nth_day, path="weights"):
        self.model = RandomForestClassifier(n_estimators=500, max_depth = 10, min_samples_split=10, random_state=42, n_jobs=10)
        self.scaler = MinMaxScaler()
        self.nth_day = nth_day
        self.path = path

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

        #df['atr'] = average_true_range(df['high'], df['low'], df['close'], window=20)
        #df['roc'] = roc(df['close'], window=20)
        df['williams'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
        #df['vwap'] = VolumeWeightedAveragePrice(high=df['high'],low=df['low'],close=df['close'], volume=df['volume'],window=14).volume_weighted_average_price()
        #df['sma_cross'] = (sma_indicator(df['close'], window=10) -sma_indicator(df['close'], window=50)).fillna(0).astype(int)
        #df['ema_cross'] = (ema_indicator(df['close'], window=10) -ema_indicator(df['close'], window=50)).fillna(0).astype(int)
        #df['wma_cross'] = (wma_indicator(df['close'], window=10) -wma_indicator(df['close'], window=50)).fillna(0).astype(int)
        #each data is reducing accuracy

        df['stoch'] = stoch(df['high'], df['low'], df['close'], window=14)

        new_predictors+=['williams','fi','emv','cmf','adi','bb_hband','bb_lband','vpt','stoch','stoch_rsi','rsi','nvi','obv','macd','macd_signal','macd_hist','adx','adx_pos','adx_neg','cci','mfi']
        return new_predictors

    def feature_selection(self, df, predictors):
        X = df[predictors]
        y = df['Target']

        selector = SelectKBest(score_func=f_classif, k=15)
        selector.fit(X, y)

        selector.transform(X)
        selected_features = [col for i, col in enumerate(X.columns) if selector.get_support()[i]]

        return selected_features

    def train_model(self, X_train, y_train):
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_train = np.nan_to_num(X_train)

        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        pickle.dump(self.model, open(f'{self.path}/model_weights_{self.nth_day}.pkl', 'wb'))

    def evaluate_model(self, X_test, y_test):
        X_test = np.where(np.isinf(X_test), np.nan, X_test)
        X_test = np.nan_to_num(X_test)

        X_test = self.scaler.fit_transform(X_test)

        with open(f'{self.path}/model_weights_{self.nth_day}.pkl', 'rb') as f:
            self.model = pickle.load(f)

        test_predictions = self.model.predict(X_test)
        #test_predictions[test_predictions >=.55] = 1
        #test_predictions[test_predictions <.55] = 0
        
    
        test_precision = precision_score(y_test, test_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        #test_recall = recall_score(y_test, test_predictions)
        #test_f1 = f1_score(y_test, test_predictions)
        #test_roc_auc = roc_auc_score(y_test, test_predictions)
        
    
        #print("Test Set Metrics:")
        print(f"Precision: {round(test_precision * 100)}%")
        print(f"Accuracy: {round(test_accuracy * 100)}%")
        #print(f"Recall: {round(test_recall * 100)}%")
        #print(f"F1-Score: {round(test_f1 * 100)}%")
        #print(f"ROC-AUC: {round(test_roc_auc * 100)}%")
        #print("Number of value counts in the test set")
        #print(pd.DataFrame(test_predictions).value_counts())
        
        next_value_prediction = 1 if test_predictions[-1] >= 0.5 else 0
        return {'accuracy': round(test_accuracy*100), 'precision': round(test_precision*100), 'sentiment': 'Bullish' if next_value_prediction == 1 else 'Bearish'}


#Train mode

async def train_process(nth_day):
    tickers =['KO','WMT','BA','PLD','AZN','LLY','INFN','GRMN','VVX','EPD','PII','WY','BLMN','AAP','ON','TGT','SMG','EL','EOG','ULTA','DV','PLNT','GLOB','LKQ','CWH','PSX','SO','TGT','GD','MU','NKE','AMGN','BX','CAT','PEP','LIN','ABBV','COST','MRK','HD','JNJ','PG','SPCB','CVX','SHEL','MS','GS','MA','V','JPM','XLF','DPZ','CMG','MCD','ALTM','PDD','MNST','SBUX','AMAT','ZS','IBM','SMCI','ORCL','XLK','VUG','VTI','VOO','IWM','IEFA','PEP','WMT','XOM','V','AVGO','BIDU','GOOGL','SNAP','DASH','SPOT','NVO','META','MSFT','ADBE','DIA','PFE','BAC','RIVN','NIO','CISS','INTC','AAPL','BYND','MSFT','HOOD','MARA','SHOP','CRM','PYPL','UBER','SAVE','QQQ','IVV','SPY','EVOK','GME','F','NVDA','AMD','AMZN','TSM','TSLA']
    tickers = list(set(tickers))
    print(len(tickers))

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    best_features = ['close','williams','fi','emv','adi','cmf','bb_hband','bb_lband','vpt','stoch','stoch_rsi','rsi','nvi','macd','mfi','cci','obv','adx','adx_pos','adx_neg']
    test_size = 0.2
    start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    predictor = TrendPredictor(nth_day=nth_day)
    
    tasks = [download_data(ticker, start_date, end_date, nth_day) for ticker in tickers]
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
    #df_train.to_csv('train_set.csv')
    #df_test.to_csv('test_set.csv')
    predictor.train_model(df_train[best_features], df_train['Target'])
    predictor.evaluate_model(df_test[best_features], df_test['Target'])

async def test_process(nth_day):
    best_features = ['close','williams','fi','emv','adi','cmf','bb_hband','bb_lband','vpt','stoch','stoch_rsi','rsi','nvi','macd','mfi','cci','obv','adx','adx_pos','adx_neg']
    test_size = 0.2
    start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    predictor = TrendPredictor(nth_day=nth_day)

    df = await download_data('BTC-USD', start_date, end_date, nth_day)
    predictors = predictor.generate_features(df)
    df = df.dropna(subset=df.columns[df.columns != "nth_day"])
    split_size = int(len(df) * (1-test_size))
    test_data = df.iloc[split_size:]

    predictor.evaluate_model(test_data[best_features], test_data['Target'])


async def main():
    
    for nth_day in [5,20,60]:
        await train_process(nth_day)
    
    await test_process(nth_day=5)

# Run the main function
#asyncio.run(main())