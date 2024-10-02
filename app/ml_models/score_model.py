import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
from collections import defaultdict
import asyncio
import aiohttp
import aiofiles
import pickle
import time

class ScorePredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.warm_start_model_path = 'ml_models/weights/ai-score/warm_start_weights.pkl'
        self.model = XGBClassifier(
            n_estimators=200,          # Increased from 100 due to problem complexity
            max_depth=6,               # Reduced to prevent overfitting with many features
            learning_rate=0.1,         # Added to control the learning process
            colsample_bytree=0.8,      # Added to randomly sample columns for each tree
            subsample=0.8,             # Added to randomly sample training data
            reg_alpha=1,               # L1 regularization to handle many features
            reg_lambda=1,              # L2 regularization to handle many features
            random_state=42,
            n_jobs=10
        )

    def preprocess_data(self, X):
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X)
        X = self.scaler.fit_transform(X)
        return X

    def warm_start_training(self, X_train, y_train):
        X_train = self.preprocess_data(X_train)
        self.model.fit(X_train, y_train)
        pickle.dump(self.model, open(f'{self.warm_start_model_path}', 'wb'))
        print("Warm start model saved.")

    def fine_tune_model(self, X_train, y_train):
        X_train = self.preprocess_data(X_train)

        with open(f'{self.warm_start_model_path}', 'rb') as f:
            self.model = pickle.load(f)

        self.model.fit(X_train, y_train)
        print("Model fine-tuned")


    def evaluate_model(self, X_test, y_test):
        X_test = self.preprocess_data(X_test)
        
        test_predictions = self.model.predict_proba(X_test)
        class_1_probabilities = test_predictions[:, 1]
        binary_predictions = (class_1_probabilities >= 0.5).astype(int)
        #print(test_predictions)
        test_precision = precision_score(y_test, binary_predictions)
        test_accuracy = accuracy_score(y_test, binary_predictions)
        
        print("Test Set Metrics:")
        print(f"Precision: {round(test_precision * 100)}%")
        print(f"Accuracy: {round(test_accuracy * 100)}%")
        print(pd.DataFrame({'y_test': y_test, 'y_pred': binary_predictions}))
        thresholds = [0.8, 0.75, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0]
        scores = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        last_prediction_prob = class_1_probabilities[-1]
        score = None
        print(f"Last prediction probability: {last_prediction_prob}")
        
        for threshold, value in zip(thresholds, scores):
            if last_prediction_prob >= threshold:
                score = value
                break

        return {'accuracy': round(test_accuracy * 100), 
                'precision': round(test_precision * 100), 
                'score': score}