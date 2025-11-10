import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
from ta.momentum import *
from ta.trend import *
from ta.volatility import *
from ta.volume import *



def trend_intensity(close, window=20):
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    return ((close - ma) / std).abs().rolling(window=window).mean()


def calculate_fdi(high, low, close, window=30):
    n1 = (np.log(high.rolling(window=window).max() - low.rolling(window=window).min()) -
          np.log(close.rolling(window=window).max() - close.rolling(window=window).min())) / np.log(2)
    return (2 - n1) * 100


def generate_ta_features(df):

    df_features = df.copy()

    df_features['sma_50'] = df['close'].rolling(window=50).mean()
    df_features['sma_200'] = df['close'].rolling(window=200).mean()
    df_features['sma_crossover'] = ((df_features['sma_50'] > df_features['sma_200']) & (df_features['sma_50'].shift(1) <= df_features['sma_200'].shift(1))).astype(int)

    df_features['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df_features['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
    df_features['ema_crossover'] = ((df_features['ema_50'] > df_features['ema_200']) & (df_features['ema_50'].shift(1) <= df_features['ema_200'].shift(1))).astype(int)

    df_features['wma'] = WMAIndicator(df['close'], window = 30).wma()

    ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
    df_features['ichimoku_a'] = ichimoku.ichimoku_a()
    df_features['ichimoku_b'] = ichimoku.ichimoku_b()
    df_features['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    bb = BollingerBands(close=df['close'])
    df_features['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']


    df_features['macd'] = macd(df['close'])
    df_features['macd_signal'] = macd_signal(df['close'])
    df_features['macd_hist'] = 2*macd_diff(df['close'])
    df_features['adx'] = adx(df['high'],df['low'],df['close'])
    df_features["adx_pos"] = adx_pos(df['high'],df['low'],df['close'])
    df_features["adx_neg"] = adx_neg(df['high'],df['low'],df['close'])
    df_features['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
    df_features['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
    
    df_features['nvi'] = NegativeVolumeIndexIndicator(close=df['close'], volume=df['volume']).negative_volume_index()
    df_features['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df_features['vpt'] = VolumePriceTrendIndicator(close=df['close'], volume=df['volume']).volume_price_trend()
    
    df_features['rsi'] = rsi(df["close"], window=60)
    df_features['rolling_rsi'] = df_features['rsi'].rolling(window=10).mean()
    df_features['stoch_rsi'] = stochrsi_k(df['close'], window=60, smooth1=3, smooth2=3)
    df_features['rolling_stoch_rsi'] = df_features['stoch_rsi'].rolling(window=10).mean()

    df_features['adi'] = acc_dist_index(high=df['high'],low=df['low'],close=df['close'],volume=df['volume'])
    df_features['cmf'] = chaikin_money_flow(high=df['high'],low=df['low'],close=df['close'],volume=df['volume'], window=20)
    df_features['emv'] = ease_of_movement(high=df['high'],low=df['low'],volume=df['volume'], window=20)
    df_features['fi'] = force_index(close=df['close'], volume=df['volume'], window= 13)

    df_features['williams'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
    df_features['kama'] = KAMAIndicator(close=df['close']).kama()

    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=60, smooth_window=3)
    df_features['stoch_k'] = stoch.stoch()
    df_features['stoch_d'] = stoch.stoch_signal()

    df_features['rocr'] = df['close'] / df['close'].shift(30) - 1 # Rate of Change Ratio (ROCR)
    df_features['ppo'] = (df_features['ema_50'] - df_features['ema_200']) / df_features['ema_50'] * 100
    df_features['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df_features['volatility_ratio'] = df['close'].rolling(window=30).std() / df['close'].rolling(window=60).std()

    df_features['fdi'] = calculate_fdi(df['high'], df['low'], df['close'])
    df_features['tii'] = trend_intensity(df['close'])

    df_features['fft'] = np.abs(np.fft.fft(df['close']))
    don_channel = DonchianChannel(high=df['high'], low=df['low'],close=df['close'], window=60)
    df_features['don_hband'] = don_channel.donchian_channel_hband()
    df_features['don_lband'] = don_channel.donchian_channel_lband()
    df_features['don_mband'] = don_channel.donchian_channel_mband()
    df_features['don_pband'] = don_channel.donchian_channel_pband()
    df_features['don_wband'] = don_channel.donchian_channel_wband()

    aroon = AroonIndicator(high=df['high'], low=df['low'], window=60)
    df_features['aroon_down'] = aroon.aroon_down()
    df_features['aroon_indicator'] = aroon.aroon_indicator()
    df_features['aroon_up'] = aroon.aroon_up()

    #df_features['ultimate_oscillator'] = UltimateOscillator(high=df['high'], low=df['low'], close=df['close']).ultimate_oscillator()
    #df_features['choppiness'] = 100 * np.log10((df['high'].rolling(window=60).max() - df['low'].rolling(window=30).min()) / df_features['atr']) / np.log10(14)
    df_features['ulcer'] = UlcerIndex(df['close'],window=60).ulcer_index()
    #df_features['keltner_hband'] = keltner_channel_hband_indicator(high=df['high'],low=df['low'],close=df['close'],window=60)
    #df_features['keltner_lband'] = keltner_channel_lband_indicator(high=df['high'],low=df['low'],close=df['close'],window=60)

    df_features = df_features.dropna()
    return df_features

def generate_statistical_features(df, windows=[20,50,200], price_col='close', 
                                high_col='high', low_col='low', volume_col='volume'):
    """
    Generate comprehensive statistical features for financial time series data.
    Focuses purely on statistical measures without technical indicators.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the price and volume data
    windows : list
        List of rolling window sizes to use for feature generation
    price_col : str
        Name of the closing price column
    high_col : str
        Name of the high price column
    low_col : str
        Name of the low price column
    volume_col : str
        Name of the volume column
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional statistical features
    """
    
    # Create a copy of the dataframe to avoid modifying the original
    df_features = df.copy()
    
    # Calculate features for each window size
    for window in windows:
        # Returns
        df_features[f'returns_{window}'] = df[price_col].pct_change(periods=window)
        
        # Log returns and statistics
        log_returns = np.log(df[price_col]/df[price_col].shift(1))
        df_features[f'log_returns_{window}'] = log_returns.rolling(window=window).mean()
        df_features[f'log_returns_std_{window}'] = log_returns.rolling(window=window).std()
        
        # Statistical moments
        df_features[f'mean_{window}'] = df[price_col].rolling(window=window).mean()
        df_features[f'std_{window}'] = df[price_col].rolling(window=window).std()
        df_features[f'var_{window}'] = df[price_col].rolling(window=window).var()
        df_features[f'skew_{window}'] = df[price_col].rolling(window=window).skew()
        df_features[f'kurt_{window}'] = df[price_col].rolling(window=window).kurt()
        
        # Quantile measures
        df_features[f'quantile_25_{window}'] = df[price_col].rolling(window=window).quantile(0.25)
        df_features[f'quantile_75_{window}'] = df[price_col].rolling(window=window).quantile(0.75)
        df_features[f'iqr_{window}'] = (
            df_features[f'quantile_75_{window}'] - df_features[f'quantile_25_{window}'])
        
        # Volatility measures
        df_features[f'realized_vol_{window}'] = (
            df_features[f'returns_{window}'].rolling(window=window).std() * np.sqrt(252))
        df_features[f'range_vol_{window}'] = (
            (df[high_col].rolling(window=window).max() - 
             df[low_col].rolling(window=window).min()) / df[price_col])
        
        # Z-scores and normalized values
        df_features[f'zscore_{window}'] = (
            (df[price_col] - df[price_col].rolling(window=window).mean()) / 
            df[price_col].rolling(window=window).std())
        
        # Volume statistics
        df_features[f'volume_mean_{window}'] = df[volume_col].rolling(window=window).mean()
        df_features[f'volume_std_{window}'] = df[volume_col].rolling(window=window).std()
        df_features[f'volume_zscore_{window}'] = (
            (df[volume_col] - df[volume_col].rolling(window=window).mean()) / 
            df[volume_col].rolling(window=window).std())
        df_features[f'volume_skew_{window}'] = df[volume_col].rolling(window=window).skew()
        df_features[f'volume_kurt_{window}'] = df[volume_col].rolling(window=window).kurt()
        
    # Clean up any NaN values
    df_features = df_features.dropna()
    
    return df_features