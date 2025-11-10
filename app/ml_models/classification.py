import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import ta
import orjson

warnings.filterwarnings('ignore')

class StockDirectionPredictor:
    def __init__(self, symbol='AAPL'):
        self.symbol = symbol
        self.data = None
        self.features = None
        self.target = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.prediction_horizon = 1
        
    def fetch_data(self):
        try:
            with open(f"stocknear/backend/app/json/historical-price/adj/{self.symbol}.json", "rb") as file:
                data = orjson.loads(file.read())
                data = sorted(data, key=lambda x: x['date'])
                self.data = pd.DataFrame(data)
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
                self.data.set_index('date', inplace=True)
            print(f"Successfully fetched {len(self.data)} days of data for {self.symbol}")
            print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_technical_indicators(self):
        df = self.data.copy()
        df['returns'] = df['adjClose'].pct_change()
        df['high_low_Ratio'] = df['adjHigh'] / df['adjLow']
        df['close_open_Ratio'] = df['adjClose'] / df['adjOpen']
        
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['adjClose'].rolling(window=window).mean()
            df[f'MA_Ratio_{window}'] = df['adjClose'] / df[f'MA_{window}']
        
        df['RSI'] = ta.momentum.RSIIndicator(df['adjClose']).rsi()
        macd = ta.trend.MACD(df['adjClose'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        bollinger = ta.volatility.BollingerBands(df['adjClose'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_Width'] = df['BB_Upper'] - df['BB_lower']
        df['BB_Position'] = (df['adjClose'] - df['BB_lower']) / df['BB_Width']
        
        stoch = ta.momentum.StochasticOscillator(df['adjHigh'], df['adjLow'], df['adjClose'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        df['volume_MA'] = df['volume'].rolling(window=20).mean()
        df['volume_Ratio'] = df['volume'] / df['volume_MA']
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        self.data = df
        return df
    
    def create_target_variable(self, time_period=1):
        """Create target variable: Will price be higher N days from now?"""
        self.prediction_horizon = time_period
        
        # For each date, look N days into the future
        future_close = self.data['adjClose'].shift(-time_period)
        current_close = self.data['adjClose']
        self.data['Target'] = (future_close > current_close).astype(int)
        
        # Count valid targets (excludes last N days where future data doesn't exist)
        valid_targets = self.data['Target'].dropna()
        
        print(f"\n=== Target Variable Creation ===")
        print(f"Prediction horizon: {time_period} days")
        print(f"Question being asked: 'Will price be higher {time_period} days from now?'")
        print(f"Total rows in data: {len(self.data)}")
        print(f"Valid targets (excluding last {time_period} days): {len(valid_targets)}")
        print(f"Target distribution: {valid_targets.value_counts().to_dict()}")
        
        # Show example of what we're predicting
        print(f"\nExample predictions:")
        sample_data = self.data[['adjClose', 'Target']].iloc[-time_period-5:-time_period].head(3)
        for idx, row in sample_data.iterrows():
            future_date = idx + pd.Timedelta(days=time_period)
            if future_date in self.data.index:
                future_price = self.data.loc[future_date, 'adjClose']
                direction = "UP" if row['Target'] == 1 else "DOWN"
                print(f"  {idx.date()}: Price ${row['adjClose']:.2f} → {future_date.date()}: ${future_price:.2f} = {direction}")
        
        return self.data['Target']
    
    def prepare_features(self):
        """Prepare features with proper handling of prediction horizon"""
        feature_columns = [
            'returns', 'high_low_Ratio', 'close_open_Ratio',
            'MA_Ratio_5', 'MA_Ratio_10', 'MA_Ratio_20', 'MA_Ratio_50',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Width', 'BB_Position', 'Stoch_K', 'Stoch_D',
            'volume_Ratio', 'volatility',
        ]

        # Create working dataframe
        work_df = self.data[feature_columns + ['Target']].copy()
        
        # Remove rows with NaN features
        initial_len = len(work_df)
        work_df = work_df.dropna()
        
        # Remove last N rows (they don't have valid targets due to future looking)
        if len(work_df) > self.prediction_horizon:
            work_df = work_df.iloc[:-self.prediction_horizon]
        
        self.features = work_df[feature_columns].copy()
        self.target = work_df['Target'].copy()

        print(f"\n=== Feature Preparation ===")
        print(f"Initial rows: {initial_len}")
        print(f"After removing NaN features: {len(work_df) + self.prediction_horizon}")
        print(f"After removing last {self.prediction_horizon} rows (no future data): {len(work_df)}")
        print(f"Final feature matrix shape: {self.features.shape}")
        print(f"Training data date range: {self.features.index[0].date()} to {self.features.index[-1].date()}")
        
        # The dates we CANNOT predict (no future data available)
        unpredictable_start = self.features.index[-1] + pd.Timedelta(days=1)
        unpredictable_end = self.data.index[-1]
        print(f"Cannot predict for dates: {unpredictable_start.date()} to {unpredictable_end.date()}")
        
        return self.features, self.target

    def get_prediction_ready_features(self):
        """Get the most recent feature vector for making new predictions"""
        feature_columns = [
            'returns', 'high_low_Ratio', 'close_open_Ratio',
            'MA_Ratio_5', 'MA_Ratio_10', 'MA_Ratio_20', 'MA_Ratio_50',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Width', 'BB_Position', 'Stoch_K', 'Stoch_D',
            'volume_Ratio', 'volatility',
        ]
        
        # Get the most recent complete feature vector
        recent_features = self.data[feature_columns].dropna().iloc[-1:]
        
        print(f"\n=== Ready for Prediction ===")
        print(f"Using features from: {recent_features.index[0].date()}")
        print(f"Will predict for: {(recent_features.index[0] + pd.Timedelta(days=self.prediction_horizon)).date()}")
        
        return recent_features

    def make_prediction(self):
        """Make a prediction for N days from the most recent date"""
        if self.model is None:
            print("Model not trained yet!")
            return None
            
        recent_features = self.get_prediction_ready_features()
        recent_features_scaled = self.scaler.transform(recent_features)
        
        prediction = self.model.predict(recent_features_scaled)[0]
        probability = self.model.predict_proba(recent_features_scaled)[0]
        
        prediction_date = recent_features.index[0] + pd.Timedelta(days=self.prediction_horizon)
        current_price = self.data.loc[recent_features.index[0], 'adjClose']
        
        print(f"\n=== PREDICTION ===")
        print(f"Based on data from: {recent_features.index[0].date()}")
        print(f"Current price: ${current_price:.2f}")
        print(f"Predicting for: {prediction_date.date()} ({self.prediction_horizon} days ahead)")
        print(f"Prediction: {'UP' if prediction == 1 else 'DOWN'}")
        print(f"Confidence: {probability[1]:.1%} (UP) / {probability[0]:.1%} (DOWN)")
        
        return {
            'prediction_date': prediction_date.date(),
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'probability_up': probability[1],
            'probability_down': probability[0],
            'current_price': current_price
        }

    def time_series_split(self, random_shuffle=False, test_size=0.2):
        """Time series split that maintains temporal order"""
        if random_shuffle:
            return train_test_split(
                self.features, self.target, test_size=test_size, shuffle=True, random_state=42
            )
        
        n_samples = len(self.features)
        split_idx = int(n_samples * (1 - test_size))
        
        X_train = self.features.iloc[:split_idx].copy()
        X_test = self.features.iloc[split_idx:].copy()
        y_train = self.target.iloc[:split_idx].copy()
        y_test = self.target.iloc[split_idx:].copy()
        
        print(f"\n=== Train/Test Split ===")
        print(f"Training: {len(X_train)} samples ({X_train.index[0].date()} to {X_train.index[-1].date()})")
        print(f"Testing: {len(X_test)} samples ({X_test.index[0].date()} to {X_test.index[-1].date()})")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, **rf_params):
        """Train the model with scaled features"""
        default_params = {
            'n_estimators': 500,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1,
        }
        default_params.update(rf_params)
        
        self.model = RandomForestClassifier(**default_params)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        
        print(f"\n=== Model Training Complete ===")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("Top 10 most important features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        precision = report['1']['precision'] if '1' in report else 0.0
        recall = report['1']['recall'] if '1' in report else 0.0
        
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n=== Model Evaluation ({self.prediction_horizon}-day horizon) ===")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision (UP predictions): {precision:.4f}")
        print(f"Recall (UP predictions): {recall:.4f}")
        print(f"Confusion Matrix:")
        print(f"  Predicted DOWN | Predicted UP")
        print(f"Actual DOWN  {cm[0,0]:3d}      |     {cm[0,1]:3d}")
        print(f"Actual UP    {cm[1,0]:3d}      |     {cm[1,1]:3d}")
        
        return accuracy, y_pred, y_pred_proba
    
    def plot_predictions(self, X_test, y_test, y_pred, y_pred_proba, threshold=0.6):
        """Plot stock price with prediction results on test set"""
        
        # Get the test period dates and prices
        test_dates = X_test.index
        test_prices = self.data.loc[test_dates, 'adjClose']
        
        # Create prediction results dataframe
        results_df = pd.DataFrame({
            'date': test_dates,
            'price': test_prices,
            'actual': y_test,
            'predicted': y_pred,
            'prob_up': y_pred_proba,
            'high_confidence': y_pred_proba > threshold
        })
        
        # Create the plot
        plt.figure(figsize=(16, 10))
        
        # Plot the price line
        plt.plot(range(len(test_prices)), test_prices, 'b-', linewidth=1.5, alpha=0.8, label='Stock Price')
        
        # Define colors and markers for different prediction outcomes
        colors = {
            'correct_up': 'green',
            'correct_down': 'darkgreen', 
            'wrong_up': 'red',
            'wrong_down': 'darkred'
        }
        
        markers = {
            'up': '^',
            'down': 'v'
        }
        
        # Plot predictions with different colors for correct/incorrect
        for i, row in results_df.iterrows():
            x_pos = list(results_df.index).index(i)
            y_pos = row['price']
            
            # Determine if prediction was correct and direction
            is_correct = (row['actual'] == row['predicted'])
            predicted_direction = 'up' if row['predicted'] == 1 else 'down'
            
            # Choose color based on correctness
            if is_correct:
                color = colors['correct_up'] if predicted_direction == 'up' else colors['correct_down']
            else:
                color = colors['wrong_up'] if predicted_direction == 'up' else colors['wrong_down']
            
            # Choose marker based on prediction direction
            marker = markers[predicted_direction]
            
            # Size based on confidence (only show high confidence predictions for clarity)
            if row['high_confidence']:
                marker_size = 120 if row['prob_up'] > 0.8 or row['prob_up'] < 0.2 else 80
                plt.scatter(x_pos, y_pos, c=color, marker=marker, s=marker_size, 
                           edgecolors='black', linewidth=0.5, alpha=0.8, zorder=5)
        
        # Add some sample annotations for clarity
        sample_indices = np.linspace(0, len(results_df)-1, 8, dtype=int)
        for idx in sample_indices:
            row = results_df.iloc[idx]
            if row['high_confidence']:
                x_pos = idx
                y_pos = row['price']
                direction = 'UP' if row['predicted'] == 1 else 'DOWN'
                confidence = row['prob_up'] if row['predicted'] == 1 else (1 - row['prob_up'])
                
                # Offset annotation to avoid overlap
                offset = 0.02 * (test_prices.max() - test_prices.min())
                y_offset = y_pos + offset if row['predicted'] == 1 else y_pos - offset
                
                plt.annotate(f'{direction}\n{confidence:.1%}', 
                           xy=(x_pos, y_pos), xytext=(x_pos, y_offset),
                           ha='center', va='center' if row['predicted'] == 1 else 'top',
                           fontsize=8, alpha=0.7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Customize the plot
        plt.title(f'{self.symbol} - Stock Price with ML Predictions\n'
                 f'Test Period: {test_dates[0].strftime("%Y-%m-%d")} to {test_dates[-1].strftime("%Y-%m-%d")}\n'
                 f'Prediction Horizon: {self.prediction_horizon} days | Confidence Threshold: {threshold}',
                 fontsize=14, pad=20)
        
        plt.xlabel('Time (Trading Days in Test Period)', fontsize=12)
        plt.ylabel('Stock Price ($)', fontsize=12)
        
        # Create custom legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', linewidth=1.5, label='Stock Price'),
            plt.scatter([], [], c='green', marker='^', s=100, edgecolors='black', 
                       label='Correct UP Prediction'),
            plt.scatter([], [], c='darkgreen', marker='v', s=100, edgecolors='black',
                       label='Correct DOWN Prediction'),
            plt.scatter([], [], c='red', marker='^', s=100, edgecolors='black',
                       label='Wrong UP Prediction'),
            plt.scatter([], [], c='darkred', marker='v', s=100, edgecolors='black',
                       label='Wrong DOWN Prediction')
        ]
        
        plt.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
        
        # Format x-axis with dates
        n_ticks = min(10, len(test_dates))
        tick_indices = np.linspace(0, len(test_dates)-1, n_ticks, dtype=int)
        tick_labels = [test_dates[i].strftime('%m/%d') for i in tick_indices]
        plt.xticks(tick_indices, tick_labels, rotation=45)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Calculate and display summary statistics
        correct_predictions = (results_df['actual'] == results_df['predicted']).sum()
        total_predictions = len(results_df)
        high_conf_predictions = results_df['high_confidence'].sum()
        high_conf_correct = ((results_df['actual'] == results_df['predicted']) & 
                            results_df['high_confidence']).sum()
        
        stats_text = (f'Total Predictions: {total_predictions}\n'
                     f'Correct: {correct_predictions} ({correct_predictions/total_predictions:.1%})\n'
                     f'High Confidence: {high_conf_predictions}\n'
                     f'High Conf. Accuracy: {high_conf_correct/max(high_conf_predictions,1):.1%}')
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        plt.show()
        
        return results_df


def main():
    predictor = StockDirectionPredictor(symbol='SPY')
    
    if not predictor.fetch_data():
        return
    
    predictor.calculate_technical_indicators()
    
    # Test with a single time period first
    time_period = 60
    print(f"\n{'='*80}")
    print(f"STOCK DIRECTION PREDICTION - {time_period} DAY HORIZON")
    print(f"{'='*80}")
    
    predictor.create_target_variable(time_period=time_period)
    predictor.prepare_features()
    
    if len(predictor.features) < 200:
        print(f"Insufficient data for {time_period}-day horizon. Need at least 200 samples.")
        return
    
    X_train, X_test, y_train, y_test = predictor.time_series_split(random_shuffle=True, test_size=0.2)
    
    predictor.train_model(X_train, y_train)
    accuracy, y_pred, y_pred_proba = predictor.evaluate_model(X_test, y_test)
    
    # Make a prediction for the future
    prediction_result = predictor.make_prediction()

    #predictor.plot_predictions( X_test, y_test, y_pred, y_pred_proba)
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()