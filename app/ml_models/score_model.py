import pandas as pd
from datetime import datetime, timedelta
import numpy as np
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
import os


class ScorePredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = lgb.LGBMClassifier(
            n_estimators=1_000,
            learning_rate=0.001,
            max_depth=12,
            num_leaves=2**12-1,
            n_jobs=10,
            random_state=42
        )
        self.warm_start_model_path = 'ml_models/weights/ai-score/stacking_weights.pkl'
        #self.pca = PCA(n_components=3)
    
    def preprocess_train_data(self, X):
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X)
        X = self.scaler.fit_transform(X)
        return X #self.pca.fit_transform(X)

    def preprocess_test_data(self, df):
        selected_features = [col for col in df.columns if col not in ['date','price','Target','fiscalYear']]

        X = df[selected_features]

        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X)
        X = self.scaler.fit_transform(X)
        return X #self.pca.fit_transform(X)

    def warm_start_training(self, X_train, y_train):
        X_train = self.preprocess_train_data(X_train)
        
        self.model.fit(X_train, y_train)
        pickle.dump(self.model, open(self.warm_start_model_path, 'wb'))
        print("Warm start model saved.")

    def fine_tune_model(self, X_train, y_train):
        X_train = self.preprocess_train_data(X_train)
        if self.model is None:
            self.model = load_model(self.warm_start_model_path, custom_objects={'SelfAttention': SelfAttention})
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.0001)

        self.model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.1, callbacks=[early_stopping, reduce_lr])
        print("Model fine-tuned (not saved).")

    def evaluate_model(self, df):
        X_test = self.preprocess_test_data(df)
        y_test = df['Target']
        
        with open(self.warm_start_model_path, 'rb') as f:
            self.model = pickle.load(f)

        test_predictions = self.model.predict_proba(X_test)
        class_1_probabilities = test_predictions[:, 1]
        binary_predictions = (class_1_probabilities >= 0.5).astype(int)

        # Calculate and print metrics
        test_precision = precision_score(y_test, binary_predictions)
        test_accuracy = accuracy_score(y_test, binary_predictions)
        test_f1_score = f1_score(y_test, binary_predictions)
        test_recall_score = recall_score(y_test, binary_predictions)
        test_roc_auc_score = roc_auc_score(y_test, binary_predictions)

        print(f"Test Precision: {round(test_precision * 100)}%")
        print(f"Test Accuracy: {round(test_accuracy * 100)}%")
        print(f"F1 Score: {round(test_f1_score * 100)}%")
        print(f"Recall: {round(test_recall_score * 100)}%")
        print(f"ROC AUC: {round(test_roc_auc_score * 100)}%")

        last_prediction_prob = class_1_probabilities[-1]
        backtest_results = pd.DataFrame({'date': df['date'], 'y_test': y_test, 'y_pred': binary_predictions, 'score': class_1_probabilities})

        print(f"Last prediction probability: {round(last_prediction_prob,2)}")

        thresholds = [0.8, 0.75, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0]
        scores = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        score = None        
        for threshold, value in zip(thresholds, scores):
            if last_prediction_prob >= threshold:
                score = value
                break

        conditions = [backtest_results['score'] >= t for t in thresholds]
        backtest_results['score'] = np.select(conditions, scores, default=1)  # Default score if no condition matches


        return {
            'accuracy': round(test_accuracy * 100),
            'precision': round(test_precision * 100),
            'f1_score': round(test_f1_score * 100),
            'recall_score': round(test_recall_score * 100),
            'roc_auc_score': round(test_roc_auc_score * 100),
            'score': score,
            'backtest': backtest_results.to_dict(orient="records")
        }
    def feature_selection(self, X_train, y_train, k=100):
        print('Feature selection:')
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)

        selector.transform(X_train)
        selected_features = [col for i, col in enumerate(X_train.columns) if selector.get_support()[i]]

        return selected_features