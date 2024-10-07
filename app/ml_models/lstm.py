import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from ta.utils import *
from ta.volatility import *
from ta.momentum import *
from ta.trend import *
from ta.volume import *
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_classif
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2

class StockPredictor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.nth_day = 10
        self.model = None #RandomForestClassifier(n_estimators=3500, min_samples_split=100, random_state=42, n_jobs=-1) #XGBClassifier(n_estimators=200, max_depth=2, learning_rate=1, objective='binary:logistic')
        self.horizons = [3,5,10, 15, 20]
        self.test_size = 0.2

    def download_data(self):
        df_original = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval="1d")
        df_original.index = pd.to_datetime(df_original.index)
        return df_original

    def preprocess_data(self, df):
        df['Target'] = (df['Close'].shift(-self.nth_day) > df['Close']).astype(int)
        df.dropna(inplace=True)
        return df

    
    def generate_features(self, df):
        new_predictors = []
        for horizon in self.horizons:
            rolling_averages = df.rolling(horizon).mean()

            ratio_column = f"Close_Ratio_{horizon}"
            df[ratio_column] = df["Close"] / rolling_averages["Close"]
            new_predictors.append(ratio_column)

            trend_column = f"Trend_{horizon}"
            df[trend_column] = df["Close"].pct_change(periods=horizon)
            new_predictors.append(trend_column)

            volatility_column = f"Volatility_{horizon}"
            df[volatility_column] = df["Close"].pct_change().rolling(horizon).std()
            new_predictors.append(volatility_column)

            volatility_mean_column = f"Volatility_Mean_{horizon}"
            df[volatility_mean_column] = df["Close"].pct_change().rolling(horizon).mean()
            new_predictors.append(volatility_mean_column)

            sma_column = f"SMA_{horizon}"
            df[sma_column] = sma_indicator(df['Close'], window=horizon)

            ema_column = f"EMA_{horizon}"
            df[ema_column] = ema_indicator(df['Close'], window=horizon)

            rsi_column = f"RSI_{horizon}"
            df[rsi_column] = rsi(df["Close"], window=horizon)
            new_predictors.append(rsi_column)

            stoch_rsi_column = f"STOCH_RSI_{horizon}"
            df[stoch_rsi_column] = stochrsi_k(df['Close'], window=horizon, smooth1=3, smooth2=3)
            new_predictors.append(stoch_rsi_column)

            stoch_column = f"STOCH_{horizon}"
            df[stoch_column] = stoch(df['High'], df['Low'], df['Close'], window=horizon)
            new_predictors.append(stoch_column)

            roc_column = f"ROC_{horizon}"
            df[roc_column] = roc(df['Close'], window=horizon)
            new_predictors.append(roc_column)

            wma_column = f"WMA_{horizon}"
            df[wma_column] = wma_indicator(df['Close'], window=horizon)
            new_predictors.append(wma_column)

            # Additional features
            atr_column = f"ATR_{horizon}"
            df[atr_column] = average_true_range(df['High'], df['Low'], df['Close'], window=horizon)
            new_predictors.append(atr_column)


            adx_column = f"ADX_{horizon}"
            df[adx_column] = adx(df['High'], df['Low'], df['Close'], window=horizon)
            new_predictors.append(adx_column)

            bb_bands_column = f"BB_{horizon}"
            df[bb_bands_column] = bollinger_hband(df['Close'], window=horizon) / df['Close']
            new_predictors.append(bb_bands_column)


        df['macd'] = macd(df['Close'])
        df['macd_signal'] = macd_signal(df['Close'])
        df['macd_hist'] = 2*macd_diff(df['Close'])
        new_predictors.append('macd')
        new_predictors.append('macd_signal')
        new_predictors.append('macd_hist')
        return new_predictors

    def feature_selection(self, df, predictors):
        X = df[predictors]
        y = df['Target']

        selector = SelectKBest(score_func=f_classif, k=5)
        selector.fit(X, y)

        selector.transform(X)
        selected_features = [col for i, col in enumerate(X.columns) if selector.get_support()[i]]

        return selected_features

    def build_lstm_model(self,input_shape):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=1024, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(units=128, return_sequences=True, kernel_regularizer=l2(0.01))))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Bidirectional(LSTM(units=64, kernel_regularizer=l2(0.01))))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))

        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
    

    def train_model(self, X_train, y_train):
        # Learning rate scheduler
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        history = self.model.fit(X_train, y_train, epochs=500, batch_size=1024, validation_split=0.1, callbacks=[early_stop])

    def evaluate_model(self, X_test, y_test):
        # Reshape X_test to remove the extra dimension
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[2])
        
        X_test_df = pd.DataFrame(X_test_reshaped, columns=predictors)
        y_test_df = pd.DataFrame(y_test, columns=['Target'])
        
        test_df = X_test_df.join(y_test_df)
        test_df = test_df.iloc[int(len(test_df) * (1 - self.test_size)):]
        
        # Implement the rest of your evaluation logic here
        test_predictions = self.model.predict(X_test)
        test_predictions = (test_predictions > 0.5).astype(int)
        print(test_predictions)
        # Assuming you have the model already defined and trained
        # Perform evaluation metrics on test_predictions and y_test
        
        test_precision = precision_score(y_test, test_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_recall = recall_score(y_test, test_predictions)
        test_f1 = f1_score(y_test, test_predictions)
        test_roc_auc = roc_auc_score(y_test, test_predictions)
    
        print("Test Set Metrics:")
        print(f"Precision: {round(test_precision * 100)}%")
        print(f"Accuracy: {round(test_accuracy * 100)}%")
        print(f"Recall: {round(test_recall * 100)}%")
        print(f"F1-Score: {round(test_f1 * 100)}%")
        print(f"ROC-AUC: {round(test_roc_auc * 100)}%")

    def predict_next_value(self, df, predictors):
        latest_data_point = df.iloc[-1][predictors]
        next_value_prediction = self.model.predict([latest_data_point])[0]
        next_value_probability = self.model.predict_proba([latest_data_point])[0][1]
        print("Predicted next value:", next_value_prediction)
        print("Probability of predicted next value:", round(next_value_probability * 100, 2), "%")
        latest_date_index = df.index[-1]
        next_prediction_date = latest_date_index + pd.DateOffset(days=self.nth_day)
        print("Corresponding date for the next prediction:", next_prediction_date)


if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = datetime(2000, 1, 1)
    end_date = datetime.today()

    predictor = StockPredictor(ticker, start_date, end_date)
    df = predictor.download_data()

    predictors = predictor.generate_features(df)
    df = predictor.preprocess_data(df)

    X = df[predictors].values
    y = df['Target'].values
    print(df)
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    # Reshape data for LSTM
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    train_size = int(len(X) * (1 - predictor.test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    predictor.train_model(X_train, y_train)
    predictor.evaluate_model(X_test, y_test)
    predictor.predict_next_value(X[-1])