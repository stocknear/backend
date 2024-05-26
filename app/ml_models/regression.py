import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.regularizers import l2
import time
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from backtesting import Backtesting
import yfinance as yf


class regression_model:
    def __init__(self, model_name, data, test_size, time_step, nth_day):
        self.model_name = model_name
        self.data = data
        self.test_size = test_size
        self.time_step = time_step
        self.nth_day = nth_day

    def correct_weekday(self, select_date):
        # Monday is 0 and Sunday is 6
        if select_date.weekday() > 4:
            select_date = select_date - timedelta(select_date.weekday() - 4)
        else:
            pass
        return select_date

    def run(self):
        dates = self.data['Date']
        df = self.data['Close']
        scaler = MinMaxScaler(feature_range=(0, 1))
        df = scaler.fit_transform(np.array(df).reshape(-1, 1))

        test_split_idx = int(df.shape[0] * (1 - self.test_size))

        train_data = df[:test_split_idx].copy()
        test_data = df[test_split_idx:].copy()



        # convert an array of values into a dataset matrix
        def create_dataset(dataset):
            dataX, dataY = [], []
            for i in range(len(dataset) - self.time_step - 1 - self.nth_day):
                a = dataset[i:(i + self.time_step), 0]  
                dataX.append(a)
                dataY.append(dataset[i + self.time_step + self.nth_day, 0])
            return np.array(dataX), np.array(dataY)

        def create_date_dataset(dataset):
            dataX = []
            for i in range(len(dataset) - self.time_step - 1 - self.nth_day):
                a = dataset[i:(i + self.time_step)].iloc[-1] 
                dataX.append(a)
            return pd.DataFrame(dataX)

        X_train, y_train = create_dataset(train_data)
        X_test, y_test = create_dataset(test_data)

        def fit_model(model, X_train, y_train):
            if self.model_name == 'LSTM':
                model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
            else:
                model.fit(X_train, y_train)

        if self.model_name == 'LinearRegression':
            model = LinearRegression(n_jobs=-1)
        elif self.model_name == "XGBoost":
            model = XGBRegressor(max_depth=10)
        elif self.model_name == "SVR":
            model = SVR()
        elif self.model_name == 'RandomForestRegressor':
            model = RandomForestRegressor()
        elif self.model_name == 'KNeighborsRegressor':
            model = KNeighborsRegressor()
        elif self.model_name == 'LSTM':
            model = Sequential()
            model.add(Bidirectional(LSTM(units=100, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=(self.time_step, 1)))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Bidirectional(LSTM(units=50, return_sequences=True, kernel_regularizer=l2(0.01))))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))
            model.add(Bidirectional(LSTM(units=10, kernel_regularizer=l2(0.01))))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='sgd', loss='mean_squared_error')
        else:
            model = LinearRegression()

        fit_model(model, X_train, y_train)

        train_predict = model.predict(X_train)
        train_predict = train_predict.reshape(-1, 1)
        test_predict = model.predict(X_test)
        test_predict = test_predict.reshape(-1, 1)

        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
        original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

        performance = Backtesting(original_ytrain, train_predict, original_ytest, test_predict).run()

        train_dates = dates[:test_split_idx].copy()
        test_dates = dates[test_split_idx:].copy()

        train_dates = create_date_dataset(train_dates)
        test_dates = create_date_dataset(test_dates)

        train_res = pd.DataFrame()
        train_res['Date'] = train_dates
        train_res['train'] = pd.DataFrame(train_predict)

        test_res = pd.DataFrame()
        test_res['Date'] = test_dates
        test_res['test'] = pd.DataFrame(test_predict)

        # Predict nth_day
        x_input = test_data[len(test_data) - self.time_step:].reshape(1, -1)
        yhat = model.predict(x_input)
        new_pred_df = pd.DataFrame(scaler.inverse_transform(yhat.reshape(-1, 1)).reshape(1, -1).tolist()[0])

        pred_res = pd.DataFrame()
        pred_res['yhat'] = new_pred_df
        print(performance)
        print(pred_res)

        return performance, train_res, test_res, pred_res


ticker = 'AMD'
start_date = datetime(2000, 1, 1)
end_date = datetime(2024,2,1) #datetime.today()
df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
df = df.reset_index()
model_name = 'LinearRegression'
test_size = 0.2
time_step = 1
nth_day = 20  # Change this value to the desired nth_day
metric, train_df, test_df, pred_df = regression_model(model_name, df, test_size=test_size, \
                                                      time_step=time_step, nth_day=nth_day).run()
