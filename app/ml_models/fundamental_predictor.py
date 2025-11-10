import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential, Model
from keras.layers import Input, Multiply, Reshape, LSTM, Dense, Conv1D, Dropout, BatchNormalization, GlobalAveragePooling1D, MaxPooling1D, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.backend import clear_session
from keras import regularizers
from keras.layers import Layer


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
        self.scaler = MinMaxScaler()
        self.model = self.build_model()

    def build_model(self):
        clear_session()
        
        # Input layer
        inputs = Input(shape=(2139,))
        
        # First dense layer
        x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        # Additional dense layers
        for units in [512,256, 256]:
            x = Dense(units, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
        
        # Reshape for attention mechanism
        x = Reshape((256, 1))(x)
        
        # Attention mechanism
        attention = Dense(256, activation='relu')(x)
        attention = Dense(1, activation='softmax')(attention)
        
        # Apply attention
        x = Multiply()([x, attention])
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Create the model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Optimizer with a lower learning rate
        optimizer = Adam(learning_rate=0.1, clipnorm = 1.0)
        
        # Compile the model
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
        #X_train = self.reshape_for_lstm(X_train)
        
        checkpoint = ModelCheckpoint('ml_models/weights/fundamental_weights/weights.keras', 
                                      save_best_only=True, save_freq = 1,
                                      monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=70, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=60, min_lr=0.00001)

        self.model.fit(X_train, y_train, epochs=100_000, batch_size=32, 
                       validation_split=0.1, callbacks=[checkpoint, early_stopping, reduce_lr])
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

    def feature_selection(self, X_train, y_train, k=100):
        print('feature selection:')
        print(X_train.shape, y_train.shape)
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)

        selector.transform(X_train)
        selected_features = [col for i, col in enumerate(X_train.columns) if selector.get_support()[i]]

        return selected_features
