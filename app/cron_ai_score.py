import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import ta
import orjson
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import sqlite3
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_symbol_list():
    symbols = []
    db_configs = [
        ("stocks.db", "SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'") ,
        ("etf.db",    "SELECT DISTINCT symbol FROM etfs"),
        ("index.db",  "SELECT DISTINCT symbol FROM indices")
    ]

    for db_file, query in db_configs:
        try:
            con = sqlite3.connect(db_file)
            cur = con.cursor()
            cur.execute(query)
            symbols.extend([r[0] for r in cur.fetchall()])
            con.close()
        except Exception:
            continue

    return symbols

@dataclass
class PredictionResult:
    """Data class for prediction results"""
    date: str
    score: int
    probability_up: float
    probability_down: float
    confidence: str
    price: float

@dataclass
class ModelMetrics:
    """Data class for model evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    confusion_matrix: np.ndarray

class DataLoader(ABC):
    """Abstract base class for data loading"""
    
    @abstractmethod
    def load_data(self, symbol: str) -> pd.DataFrame:
        pass

class JSONDataLoader(DataLoader):
    """Concrete implementation for JSON data loading"""
    
    def __init__(self, base_path: str = "json/historical-price/adj"):
        self.base_path = Path(base_path)
    
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load stock data from JSON file"""
        file_path = self.base_path / f"{symbol}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            with open(file_path, "rb") as file:
                data = orjson.loads(file.read())
                data = sorted(data, key=lambda x: x['date'])
                df = pd.DataFrame(data)
                
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.set_index('date', inplace=True)
                
            logger.info(f"Successfully loaded {len(df)} days of data for {symbol}")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            raise

class TechnicalIndicatorCalculator:
    """Handles calculation of technical indicators"""
    
    @staticmethod
    def calculate_basic_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price ratios"""
        df = df.copy()
        df['returns'] = df['adjClose'].pct_change()
        df['high_low_ratio'] = df['adjHigh'] / df['adjLow']
        df['close_open_ratio'] = df['adjClose'] / df['adjOpen']
        return df
    
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """Calculate moving averages and their ratios"""
        for window in windows:
            df[f'ma_{window}'] = df['adjClose'].rolling(window=window).mean()
            df[f'ma_ratio_{window}'] = df['adjClose'] / df[f'ma_{window}']
            df[f'ma_slope_{window}'] = df[f'ma_{window}'].pct_change(5)  # 5-day slope
        return df
    
    @staticmethod
    def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based indicators"""
        # RSI
        for window in [7,14,20,30,50]:
            df[f'rsi_{window}'] = ta.momentum.RSIIndicator(df['adjClose'], window=window).rsi()
            df[f'rsi_oversold_{window}'] = (df[f'rsi_{window}'] < 30).astype(int)
            df[f'rsi_overbought_{window}'] = (df[f'rsi_{window}'] > 70).astype(int)
        
        # MACD
        macd = ta.trend.MACD(df['adjClose'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['adjHigh'], df['adjLow'], df['adjClose'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['adjHigh'], df['adjLow'], df['adjClose']).williams_r()
        
        return df
    
    @staticmethod
    def calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based indicators"""
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['adjClose'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['adjClose'] - df['bb_lower']) / df['bb_width']
        
        # Average True Range
        df['atr'] = ta.volatility.AverageTrueRange(df['adjHigh'], df['adjLow'], df['adjClose']).average_true_range()
        
        # Historical volatility
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_50'] = df['returns'].rolling(window=50).std()
        
        return df
    
    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # On Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['adjClose'], df['volume']).on_balance_volume()
        df['obv_ma'] = df['obv'].rolling(window=20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_ma']
        
        return df
    
    @classmethod
    def calculate_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = cls.calculate_basic_ratios(df)
        df = cls.calculate_moving_averages(df)
        df = cls.calculate_momentum_indicators(df)
        df = cls.calculate_volatility_indicators(df)
        df = cls.calculate_volume_indicators(df)
        return df

class FeatureEngineer:
    """Handles feature engineering and selection"""
    
    def __init__(self):
        self.feature_columns = [
            'returns', 'high_low_ratio', 'close_open_ratio',
            'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50',
            'ma_slope_5', 'ma_slope_10', 'ma_slope_20',
            'rsi_7','rsi_14','rsi_20','rsi_30','rsi_50', 'rsi_oversold_7', 'rsi_overbought_7',
            'rsi_oversold_14', 'rsi_overbought_14','rsi_oversold_20','rsi_overbought_20',
            'rsi_oversold_30','rsi_overbought_30', 'rsi_oversold_50', 'rsi_overbought_50',
            'macd', 'macd_signal', 'macd_histogram', 'macd_bullish',
            'bb_width', 'bb_position', 'stoch_k', 'stoch_d',
            'williams_r', 'atr', 'volatility_20',
            'volume_ratio', 'obv_ratio'
        ]
    
    def create_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """Create lagged features"""
        base_features = ['returns', 'volume_ratio', 'rsi_7','rsi_14','rsi_20','rsi_30','rsi_50', 'bb_position']
        
        for feature in base_features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
                    self.feature_columns.append(f'{feature}_lag_{lag}')
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistical features"""
        windows = [5, 10, 20]
        
        for window in windows:
            df[f'returns_rolling_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_rolling_std_{window}'] = df['returns'].rolling(window).std()
            self.feature_columns.extend([
                f'returns_rolling_mean_{window}',
                f'returns_rolling_std_{window}'
            ])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering techniques"""
        df = self.create_lagged_features(df)
        df = self.create_rolling_features(df)
        return df

class ScoreCalculator:
    """Handles score calculation with configurable thresholds"""
    
    def __init__(self, thresholds: List[float] = None, scores: List[int] = None):
        self.thresholds = thresholds or [0.8, 0.75, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0]
        self.scores = scores or [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    def get_score(self, prob_up: float) -> int:
        """Calculate score based on probability"""
        for threshold, score in zip(self.thresholds, self.scores):
            if prob_up >= threshold:
                return score
        return 1
    
    def get_confidence_level(self, prob_up: float) -> str:
        """Get confidence level description"""
        if prob_up >= 0.8:
            return "Very High"
        elif prob_up >= 0.7:
            return "High"
        elif prob_up >= 0.6:
            return "Medium"
        elif prob_up >= 0.5:
            return "Low"
        else:
            return "Very Low"

class ModelTrainer:
    """Handles model training and hyperparameter optimization"""
    
    def __init__(self, use_robust_scaler: bool = True):
        self.scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.model = None
        self.best_params = None
    
    def get_default_params(self) -> Dict:
        """Get default RandomForest parameters"""
        return {
            'n_estimators': [300, 500, 700],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                cv_folds: int = 3) -> RandomForestClassifier:
        """Optimize hyperparameters using GridSearchCV"""
        logger.info("Starting hyperparameter optimization...")
        
        rf = RandomForestClassifier(random_state=42, n_jobs=4)
        param_grid = self.get_default_params()
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=tscv, scoring='roc_auc',
            n_jobs=4, verbose=1
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        grid_search.fit(X_train_scaled, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   optimize: bool = False, **rf_params) -> RandomForestClassifier:
        """Train the model with optional hyperparameter optimization"""
        
        if optimize:
            return self.optimize_hyperparameters(X_train, y_train)
        
        # Use default parameters
        default_params = {
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_split': 20,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': 4,
        }
        default_params.update(rf_params)
        
        self.model = RandomForestClassifier(**default_params)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        
        logger.info("Model training complete")
        self._log_feature_importance(X_train.columns)
        
        return self.model
    
    def _log_feature_importance(self, feature_names: List[str], top_n: int = 10):
        """Log feature importance"""
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top {top_n} most important features:")
        for _, row in feature_importance.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

class ModelEvaluator:
    """Handles model evaluation and metrics calculation"""
    
    def evaluate_model(self, model: RandomForestClassifier, scaler: Any,
                      X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Comprehensive model evaluation"""
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        precision = report.get('1', {}).get('precision', 0.0)
        recall = report.get('1', {}).get('recall', 0.0)
        f1 = report.get('1', {}).get('f1-score', 0.0)
        
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc_score = 0.0
        
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc_score,
            confusion_matrix=cm
        )
        
        self._log_evaluation_results(metrics)
        return metrics
    
    def _log_evaluation_results(self, metrics: ModelMetrics):
        """Log evaluation results"""
        logger.info(f"Model Evaluation Results:")
        logger.info(f"  Accuracy: {metrics.accuracy*100:.2f}%")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall: {metrics.recall:.4f}")
        logger.info(f"  F1-Score: {metrics.f1_score:.4f}")
        logger.info(f"  AUC Score: {metrics.auc_score:.4f}")
        
        cm = metrics.confusion_matrix
        if cm.shape == (2, 2):
            logger.info(f"  Confusion Matrix:")
            logger.info(f"    Predicted DOWN | Predicted UP")
            logger.info(f"  Actual DOWN  {cm[0,0]:3d}      |     {cm[0,1]:3d}")
            logger.info(f"  Actual UP    {cm[1,0]:3d}      |     {cm[1,1]:3d}")

class StockDirectionPredictor:
    """Main class for stock direction prediction"""
    
    def __init__(self, symbol: str = 'AAPL', data_loader: DataLoader = None):
        self.symbol = symbol
        self.data_loader = data_loader or JSONDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.score_calculator = ScoreCalculator()
        
        self.data = None
        self.features = None
        self.target = None
        self.prediction_horizon = 1
        
    def load_and_prepare_data(self) -> bool:
        """Load and prepare data with all indicators"""
        try:
            self.data = self.data_loader.load_data(self.symbol)
            self.data = TechnicalIndicatorCalculator.calculate_all_indicators(self.data)
            self.data = self.feature_engineer.engineer_features(self.data)
            return True
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return False
    
    def create_target_variable(self, time_period: int = 1) -> pd.Series:
        """Create target variable with validation"""
        self.prediction_horizon = time_period
        
        if len(self.data) < time_period + 50:  # Need minimum data
            raise ValueError(f"Insufficient data for {time_period}-day horizon")
        
        future_close = self.data['adjClose'].shift(-time_period)
        current_close = self.data['adjClose']
        self.data['target'] = (future_close > current_close).astype(int)
        
        valid_targets = self.data['target'].dropna()
        
        logger.info(f"Target variable created for {time_period}-day horizon")
        logger.info(f"Valid targets: {len(valid_targets)}")
        logger.info(f"Target distribution: {valid_targets.value_counts().to_dict()}")
        
        return self.data['target']
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features with proper validation"""
        # Get available feature columns
        available_features = [col for col in self.feature_engineer.feature_columns 
                            if col in self.data.columns]
        
        if not available_features:
            raise ValueError("No valid features found")
        
        # Create working dataframe
        work_df = self.data[available_features + ['target']].copy()
        work_df = work_df.dropna()
        
        # Remove last N rows (no future data available)
        if len(work_df) > self.prediction_horizon:
            work_df = work_df.iloc[:-self.prediction_horizon]
        
        self.features = work_df[available_features].copy()
        self.target = work_df['target'].copy()
        
        logger.info(f"Feature matrix prepared: {self.features.shape}")
        logger.info(f"Using features: {available_features[:10]}...")  # Show first 10
        
        return self.features, self.target
    
    def time_series_split(self, test_size: float = 0.2, shuffle_data: bool = False, 
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        n_samples = len(self.features)
        
        if shuffle_data:
            # Random split with shuffling - breaks temporal order
            logger.info("Using shuffled data split (temporal order not preserved)")
            
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, 
                self.target,
                test_size=test_size,
                random_state=random_state,
                stratify=None  # Can add stratification if needed
            )
            
            # Sort by index to maintain some semblance of order in logs
            X_train = X_train.sort_index()
            X_test = X_test.sort_index()
            y_train = y_train.sort_index()
            y_test = y_test.sort_index()
            
            logger.info(f"Shuffled split - Train: {len(X_train)}, Test: {len(X_test)}")
            logger.info(f"Train date range: {X_train.index[0]} to {X_train.index[-1]}")
            logger.info(f"Test date range: {X_test.index[0]} to {X_test.index[-1]}")
            
        else:
            # Traditional time series split - maintains temporal order
            logger.info("Using temporal data split (chronological order preserved)")
            
            split_idx = int(n_samples * (1 - test_size))
            
            X_train = self.features.iloc[:split_idx].copy()
            X_test = self.features.iloc[split_idx:].copy()
            y_train = self.target.iloc[:split_idx].copy()
            y_test = self.target.iloc[split_idx:].copy()
            
            logger.info(f"Temporal split - Train: {len(X_train)}, Test: {len(X_test)}")
            logger.info(f"Train period: {X_train.index[0]} to {X_train.index[-1]}")
            logger.info(f"Test period: {X_test.index[0]} to {X_test.index[-1]}")
        
        # Log class distribution for both splits
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        logger.info(f"Train target distribution: {train_dist.to_dict()}")
        logger.info(f"Test target distribution: {test_dist.to_dict()}")
        
        return X_train, X_test, y_train, y_test
        
    def train_and_evaluate(self, optimize_hyperparameters: bool = False) -> ModelMetrics:
        """Train model and evaluate performance"""
        X_train, X_test, y_train, y_test = self.time_series_split(shuffle_data=False)
        
        # Train model
        self.trainer.train_model(X_train, y_train, optimize=optimize_hyperparameters)
        
        # Evaluate model
        metrics = self.evaluator.evaluate_model(
            self.trainer.model, self.trainer.scaler, X_test, y_test
        )
        
        return metrics
    
    def make_predictions(self, n_predictions) -> Dict[str, Any]:
        """Make predictions for the last N available data points"""
        if self.trainer.model is None:
            raise ValueError("Model not trained yet!")
        
        available_features = [col for col in self.feature_engineer.feature_columns 
                            if col in self.data.columns]
        
        recent_data = self.data[available_features].dropna().iloc[-n_predictions:]
        recent_features_scaled = self.trainer.scaler.transform(recent_data)
        
        predictions = self.trainer.model.predict(recent_features_scaled)
        probabilities = self.trainer.model.predict_proba(recent_features_scaled)
        
        results = []
        
        # Sample every 30 days for backtest results
        for i in range(0, min(n_predictions, len(recent_data)), 20):
            date_used = recent_data.index[i]
            current_price = self.data.loc[date_used, 'adjClose']
            prob_up = probabilities[i][1]
            prob_down = probabilities[i][0]
            
            result = PredictionResult(
                date=date_used.date().strftime("%Y-%m-%d"),
                score=self.score_calculator.get_score(prob_up),
                probability_up=prob_up,
                probability_down=prob_down,
                confidence=self.score_calculator.get_confidence_level(prob_up),
                price=current_price
            )
            
            results.append({
                'date': result.date,
                'score': result.score
            })
            
            logger.info(f"Prediction for {result.date}: Score {result.score}, "
                       f"Confidence: {result.confidence} ({prob_up:.1%})")
        
        return {'backtest': results, 'score': results[-1]['score']} #latest forecast

def save_results(data: Dict, symbol: str, base_path: str = "json/ai-score/companies"):
    """Save results to JSON file"""
    path = Path(base_path) / f"{symbol}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as file:
        file.write(orjson.dumps(data))
    
    logger.info(f"Results saved to {path}")

def main():
    """Main execution function"""

    #total_symbols = load_symbol_list()
    #testing mode
    #total_symbols = ['AMD']

    for symbol in tqdm(total_symbols):
        time_period = 60
        
        logger.info(f"Starting stock direction prediction for {symbol}")
        logger.info(f"Prediction horizon: {time_period} days")
        
        try:
            # Initialize predictor
            predictor = StockDirectionPredictor(symbol=symbol)
            
            # Load and prepare data
            if not predictor.load_and_prepare_data():
                logger.error("Failed to load data")
                continue
            
            # Create target and prepare features
            predictor.create_target_variable(time_period=time_period)
            predictor.prepare_features()
            
            # Check minimum data requirements
            if len(predictor.features) < 200:
                logger.error(f"Insufficient data: {len(predictor.features)} samples")
                continue
            
            # Train and evaluate model
            metrics = predictor.train_and_evaluate(optimize_hyperparameters=False)
            
            # Make predictions
            prediction_results = predictor.make_predictions(n_predictions=252*2)
            
            # Prepare final results
            results = {
                'accuracy': round(metrics.accuracy * 100, 2),
                'auc_score': round(metrics.auc_score, 4),
                'precision': round(metrics.precision, 4),
                'recall': round(metrics.recall, 4),
                **prediction_results
            }
            
            #needed to make it compatible with orjson
            results = {
                'accuracy': float(round(metrics.accuracy * 100, 2)),
                'auc_score': float(round(metrics.auc_score, 4)),
                'precision': float(round(metrics.precision, 4)),
                'recall': float(round(metrics.recall, 4)),
                'backtest': [  # each date/score is already native types
                    {'date': r['date'], 'score': int(r['score'])}
                    for r in prediction_results['backtest']
                ],
                'score': int(prediction_results['score'])
            }


            # Save results
            save_results(results, symbol)
            
            logger.info("Prediction completed successfully")
            logger.info(f"Final results: {results}")
            
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            pass

if __name__ == "__main__":
    main()