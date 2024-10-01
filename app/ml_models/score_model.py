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
        self.model = self.build_model()

    def build_model(self):
        clear_session()
        
        # Input layer
        inputs = Input(shape=(139,))
        
        # First dense layer
        x = Dense(128, activation='elu')(inputs)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        
        # Additional dense layers
        for units in [64,32]:
            x = Dense(units, activation='elu')(x)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
        
        # Reshape for attention mechanism
        x = Reshape((32, 1))(x)
        
        # Attention mechanism
        #attention = Dense(32, activation='elu')(x)
        #attention = Dense(1, activation='softmax')(attention)
        
        # Apply attention
        #x = Multiply()([x, attention])
        
        x, _ = SelfAttention()(x)

        # Global average pooling
        #x = GlobalAveragePooling1D()(x)
        
        # Output layer (for class probabilities)
        outputs = Dense(2, activation='softmax')(x)  # Two neurons for class probabilities with softmax
        
        # Create the model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Optimizer with a lower learning rate
        optimizer = Adam(learning_rate=0.01, clipnorm=1.0)
        
        # Compile the model
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
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
        
        checkpoint = ModelCheckpoint('ml_models/weights/ai-score/weights.keras', 
                                      save_best_only=True, save_freq = 1,
                                      monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.001)

        self.model.fit(X_train, y_train, epochs=100_000, batch_size=32, 
                       validation_split=0.1, callbacks=[checkpoint, early_stopping, reduce_lr])
        self.model.save('ml_models/weights/ai-score/weights.keras')

    def evaluate_model(self, X_test, y_test):
        # Preprocess the test data
        X_test = self.preprocess_data(X_test)
        #X_test = self.reshape_for_lstm(X_test)
        
        # Load the trained model
        self.model = load_model('ml_models/weights/ai-score/weights.keras')
        
        # Get the model's predictions
        test_predictions = self.model.predict(X_test)
        print(test_predictions)

        # Extract the probabilities for class 1 (index 1 in the softmax output)
        class_1_probabilities = test_predictions[:, 1]
        # Convert probabilities to binary predictions using a threshold of 0.5
        binary_predictions = (class_1_probabilities >= 0.5).astype(int)
        
        # Calculate precision and accuracy using binary predictions
        test_precision = precision_score(y_test, binary_predictions)
        test_accuracy = accuracy_score(y_test, binary_predictions)
        
        print("Test Set Metrics:")
        print(f"Precision: {round(test_precision * 100)}%")
        print(f"Accuracy: {round(test_accuracy * 100)}%")
        
        # Define thresholds and corresponding scores
        thresholds = [0.8, 0.75, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2]
        scores = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        # Get the last prediction value (class 1 probability) for scoring
        last_prediction_prob = class_1_probabilities[-1]

        # Initialize score to 0 (or any default value)
        score = 0
        print(last_prediction_prob)
        # Determine the score based on the last prediction probability
        for threshold, value in zip(thresholds, scores):
            if last_prediction_prob >= threshold:
                score = value
                break  # Exit the loop once the score is determined

        # Return the evaluation results
        return {'accuracy': round(test_accuracy * 100), 
                'precision': round(test_precision * 100), 
                'score': score}



    def feature_selection(self, X_train, y_train, k=100):
        print('feature selection:')
        print(X_train.shape, y_train.shape)
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)

        selector.transform(X_train)
        selected_features = [col for i, col in enumerate(X_train.columns) if selector.get_support()[i]]

        return selected_features
