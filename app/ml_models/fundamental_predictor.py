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
from keras.layers import LSTM, Dense, Conv1D, Dropout, BatchNormalization
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
import aiofiles
import pickle
import time

# Based on the paper: https://arxiv.org/pdf/1603.00751

class FundamentalPredictor:
    def __init__(self):
        self.model = self.build_model()
        self.scaler = MinMaxScaler()

    def build_model(self):
        clear_session()
        model = Sequential()
        
        model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(None, 1)))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        
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

        # Adam optimizer with a learning rate of 0.01
        optimizer = Adam(learning_rate=0.01)
        
        # Compile model with binary crossentropy loss and accuracy metric
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def preprocess_data(self, X):
        # X = X.applymap(lambda x: 9999 if x == 0 else x)  # Replace 0 with 9999 as suggested in the paper
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X)
        X = self.scaler.fit_transform(X)
        return X

    def reshape_for_lstm(self, X):
        return X.reshape((X.shape[0], X.shape[1], 1))

    def train_model(self, X_train, y_train):
        X_train = self.preprocess_data(X_train)
        X_train = self.reshape_for_lstm(X_train)
        
        checkpoint = ModelCheckpoint('ml_models/weights/fundamental_weights/weights.keras', 
                                      save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.model.fit(X_train, y_train, epochs=250, batch_size=32, 
                       validation_split=0.2, callbacks=[checkpoint, early_stopping])
        self.model.save('ml_models/weights/fundamental_weights/weights.keras')

    def evaluate_model(self, X_test, y_test):
        X_test = self.preprocess_data(X_test)
        X_test = self.reshape_for_lstm(X_test)
        
        self.model = load_model('ml_models/weights/fundamental_weights/weights.keras')
        
        test_predictions = self.model.predict(X_test).flatten()
        
        test_predictions[test_predictions >= 0.5] = 1
        test_predictions[test_predictions < 0.5] = 0
        
        test_precision = precision_score(y_test, test_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print("Test Set Metrics:")
        print(f"Precision: {round(test_precision * 100)}%")
        print(f"Accuracy: {round(test_accuracy * 100)}%")
        
        next_value_prediction = 1 if test_predictions[-1] >= 0.5 else 0
        return {'accuracy': round(test_accuracy * 100), 
                'precision': round(test_precision * 100), 
                'sentiment': 'Bullish' if next_value_prediction == 1 else 'Bearish'}, test_predictions

    def feature_selection(self, X_train, y_train, k=8):
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)

        selector.transform(X_train)
        selected_features = [col for i, col in enumerate(X_train.columns) if selector.get_support()[i]]

        return selected_features
