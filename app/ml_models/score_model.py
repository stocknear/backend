import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import lightgbm as lgb

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
        self.pca = PCA(n_components=0.95)  # Retain components explaining 95% variance
        self.warm_start_model_path = 'ml_models/weights/ai-score/warm_start_weights.pkl'
        self.model = lgb.LGBMClassifier(
            n_estimators=1000,           # Number of boosting iterations - good balance between performance and training time
            learning_rate=0.005,         # Smaller learning rate for better generalization
            max_depth=8,                 # Controlled depth to prevent overfitting
            num_leaves=31,              # 2^5-1, prevents overfitting while maintaining model complexity
            colsample_bytree=0.8,       # Use 80% of features per tree to reduce overfitting
            subsample=0.8,              # Use 80% of data per tree to reduce overfitting
            min_child_samples=20,       # Minimum samples per leaf to ensure reliable splits
            random_state=42,            # For reproducibility
            class_weight='balanced',    # Important for potentially imbalanced stock data
            reg_alpha=0.1,             # L1 regularization
            reg_lambda=0.1,            # L2 regularization
            n_jobs=-1,                 # Use all CPU cores
            verbose=-1,                # Reduce output noise
            warm_start= True,
        )
        '''
        XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=10
        )
        '''

    def preprocess_train_data(self, X):
        """Preprocess training data by scaling and applying PCA."""
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X)
        X = self.scaler.fit_transform(X)  # Transform using the fitted scaler
        return self.pca.fit_transform(X)  # Fit PCA and transform

    def preprocess_test_data(self, X):
        """Preprocess test data by scaling and applying PCA."""
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X)
        X = self.scaler.transform(X)  # Transform using the fitted scaler
        return self.pca.transform(X)  # Transform using the fitted PCA

    def warm_start_training(self, X_train, y_train):
        X_train = self.preprocess_train_data(X_train)
        self.model.fit(X_train, y_train)
        pickle.dump(self.model, open(f'{self.warm_start_model_path}', 'wb'))
        print("Warm start model saved.")

    def batch_train_model(self, X_train, y_train, batch_size=1000):
        """Train the model in batches to handle large datasets."""
        num_samples = len(X_train)
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            # Preprocess each batch
            X_batch = self.preprocess_train_data(X_batch)

            # Fit model on each batch (incremental training with warm_start=True)
            self.model.fit(X_batch, y_batch, eval_set=[(X_batch, y_batch)])

            print(f"Trained on batch {start_idx} to {end_idx}")
        
        # After batch training, save the model
        pickle.dump(self.model, open(f'{self.warm_start_model_path}', 'wb'))
        print("Batch learning completed and model saved.")

    def fine_tune_model(self, X_train, y_train):
        X_train = self.preprocess_train_data(X_train)
        with open(f'{self.warm_start_model_path}', 'rb') as f:
            self.model = pickle.load(f)

        self.model.fit(X_train, y_train)
        print("Model fine-tuned")


    def evaluate_model(self, X_test, y_test):
        X_test = self.preprocess_test_data(X_test)
        
        test_predictions = self.model.predict_proba(X_test)
        class_1_probabilities = test_predictions[:, 1]
        binary_predictions = (class_1_probabilities >= 0.5).astype(int)
        #print(test_predictions)
        test_precision = precision_score(y_test, binary_predictions)
        test_accuracy = accuracy_score(y_test, binary_predictions)
        test_f1_score = f1_score(y_test, binary_predictions)
        test_recall_score = recall_score(y_test, binary_predictions)
        test_roc_auc_score = roc_auc_score(y_test, binary_predictions)
        
        print("Test Set Metrics:")
        print(f"Precision: {round(test_precision * 100)}%")
        print(f"Accuracy: {round(test_accuracy * 100)}%")
        print(f"F1 Score: {round(test_f1_score * 100)}%")
        print(f"Recall Score: {round(test_recall_score * 100)}%")
        print(f"ROC AUC Score: {round(test_roc_auc_score * 100)}%")
        
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
                'f1_score': round(test_f1_score * 100),
                'recall_score': round(test_recall_score * 100),
                'roc_auc_score': round(test_roc_auc_score * 100),
                'score': score}