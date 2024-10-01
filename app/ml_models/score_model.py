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
from tensorflow.keras import backend as K

from tqdm import tqdm
from collections import defaultdict
import asyncio
import aiohttp
import aiofiles
import pickle
import time

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        super(SelfAttention, self).build(input_shape)
    
    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W))
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensor of same shape as x for multiplication
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context, alpha

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])


class ScorePredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.warm_start_model_path = 'ml_models/weights/ai-score/warm_start_weights.keras'

    def build_model(self):
        clear_session()
        
        inputs = Input(shape=(335,))
        x = Dense(512, activation='elu')(inputs)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        
        for units in [64, 32]:
            x = Dense(units, activation='elu')(x)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
        
        x = Reshape((32, 1))(x)
        x, _ = SelfAttention()(x)
        outputs = Dense(2, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.01, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def preprocess_data(self, X):
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X)
        X = self.scaler.fit_transform(X)
        return X

    def warm_start_training(self, X_train, y_train):
        X_train = self.preprocess_data(X_train)
        self.model = self.build_model()
        
        checkpoint = ModelCheckpoint(self.warm_start_model_path, save_best_only=True, save_freq=1, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.001)

        self.model.fit(X_train, y_train, epochs=100_000, batch_size=32, validation_split=0.1, callbacks=[checkpoint, early_stopping, reduce_lr])
        self.model.save(self.warm_start_model_path)
        print("Warm start model saved.")

    def fine_tune_model(self, X_train, y_train):
        X_train = self.preprocess_data(X_train)
        
        if self.model is None:
            self.model = load_model(self.warm_start_model_path, custom_objects={'SelfAttention': SelfAttention})
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

        self.model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping, reduce_lr])
        print("Model fine-tuned (not saved).")

    def evaluate_model(self, X_test, y_test):
        X_test = self.preprocess_data(X_test)
        
        if self.model is None:
            raise ValueError("Model has not been trained or fine-tuned. Call warm_start_training or fine_tune_model first.")
        
        test_predictions = self.model.predict(X_test)
        class_1_probabilities = test_predictions[:, 1]
        binary_predictions = (class_1_probabilities >= 0.5).astype(int)
        print(test_predictions)
        test_precision = precision_score(y_test, binary_predictions)
        test_accuracy = accuracy_score(y_test, binary_predictions)
        
        print("Test Set Metrics:")
        print(f"Precision: {round(test_precision * 100)}%")
        print(f"Accuracy: {round(test_accuracy * 100)}%")
        
        thresholds = [0.8, 0.75, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2]
        scores = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        last_prediction_prob = class_1_probabilities[-1]
        score = 0
        print(f"Last prediction probability: {last_prediction_prob}")
        
        for threshold, value in zip(thresholds, scores):
            if last_prediction_prob >= threshold:
                score = value
                break

        return {'accuracy': round(test_accuracy * 100), 
                'precision': round(test_precision * 100), 
                'score': score}

    def feature_selection(self, X_train, y_train, k=100):
        print('Feature selection:')
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)

        selector.transform(X_train)
        selected_features = [col for i, col in enumerate(X_train.columns) if selector.get_support()[i]]

        return selected_features