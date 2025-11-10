import pandas as pd
import numpy as np
from typing import Dict


class TechnicalIndicators:
    """Centralized technical indicator calculations"""

    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if len(prices) < window + 1:
            return pd.Series([50.0] * len(prices), index=prices.index)

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()

        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.replace([np.inf, -np.inf], np.nan)

        return rsi

    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=window).mean()

    @staticmethod
    def ema(prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=window).mean()

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD with signal and histogram"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower
        }

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)

        tr = TechnicalIndicators.atr(high, low, close, window)
        plus_di = 100 * pd.Series(plus_dm).rolling(window=window).mean() / tr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=window).mean() / tr
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.rolling(window=window).mean()

    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                               k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator (%K and %D)"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_window).mean()
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=window).mean()
        mad = tp.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean())
        return (tp - sma_tp) / (0.015 * mad)

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = [0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i - 1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        return cumulative_tp_vol / cumulative_vol

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)

    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """Money Flow Index (Volume-weighted RSI)"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Calculate positive and negative money flow
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.append(0)
                negative_flow.append(money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        # Add first value (no comparison possible)
        positive_flow.insert(0, 0)
        negative_flow.insert(0, 0)
        
        positive_flow = pd.Series(positive_flow, index=close.index)
        negative_flow = pd.Series(negative_flow, index=close.index)
        
        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            money_ratio = positive_mf / negative_mf
            mfi = 100 - (100 / (1 + money_ratio))
            mfi = mfi.replace([np.inf, -np.inf], np.nan)
        
        return mfi

    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.20) -> pd.Series:
        """Parabolic Stop and Reverse (SAR)"""
        sar = np.zeros(len(close))
        trend = np.zeros(len(close))  # 1 for uptrend, -1 for downtrend
        af = np.zeros(len(close))
        ep = np.zeros(len(close))  # Extreme point
        
        # Initialize first values
        sar[0] = low.iloc[0]
        trend[0] = 1  # Start with uptrend
        af[0] = af_start
        ep[0] = high.iloc[0]
        
        for i in range(1, len(close)):
            # Previous values
            prev_sar = sar[i-1]
            prev_trend = trend[i-1]
            prev_af = af[i-1]
            prev_ep = ep[i-1]
            
            # Calculate SAR
            sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
            
            # Check for trend reversal
            if prev_trend == 1:  # Was uptrend
                if low.iloc[i] <= sar[i]:  # Trend reversal to downtrend
                    trend[i] = -1
                    sar[i] = prev_ep  # SAR becomes previous EP
                    af[i] = af_start  # Reset AF
                    ep[i] = low.iloc[i]  # New EP is current low
                else:  # Continue uptrend
                    trend[i] = 1
                    # Adjust SAR if it's above previous lows
                    sar[i] = min(sar[i], min(low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1]))
                    
                    if high.iloc[i] > prev_ep:  # New high
                        ep[i] = high.iloc[i]
                        af[i] = min(prev_af + af_increment, af_max)
                    else:  # No new high
                        ep[i] = prev_ep
                        af[i] = prev_af
            
            else:  # Was downtrend
                if high.iloc[i] >= sar[i]:  # Trend reversal to uptrend
                    trend[i] = 1
                    sar[i] = prev_ep  # SAR becomes previous EP
                    af[i] = af_start  # Reset AF
                    ep[i] = high.iloc[i]  # New EP is current high
                else:  # Continue downtrend
                    trend[i] = -1
                    # Adjust SAR if it's below previous highs
                    sar[i] = max(sar[i], max(high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1]))
                    
                    if low.iloc[i] < prev_ep:  # New low
                        ep[i] = low.iloc[i]
                        af[i] = min(prev_af + af_increment, af_max)
                    else:  # No new low
                        ep[i] = prev_ep
                        af[i] = prev_af
        
        return pd.Series(sar, index=close.index)

    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Donchian Channels (Breakout system)"""
        # Shift by 1 period to exclude current day from calculation
        # This allows for proper breakout detection
        upper = high.shift(1).rolling(window=window).max()
        lower = low.shift(1).rolling(window=window).min()
        middle = (upper + lower) / 2
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

    @staticmethod
    def standard_deviation(prices: pd.Series, window: int = 20) -> pd.Series:
        """Standard Deviation - Price volatility measure"""
        return prices.rolling(window=window).std()

    @staticmethod
    def historical_volatility(prices: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
        """Historical Volatility - Annualized price volatility"""
        # Calculate daily returns
        returns = prices.pct_change()
        # Calculate rolling standard deviation of returns
        volatility = returns.rolling(window=window).std()
        
        if annualize:
            # Annualize using ~252 trading days
            volatility = volatility * np.sqrt(252)
        
        return volatility

    @staticmethod
    def chaikin_volatility(high: pd.Series, low: pd.Series, window: int = 10, period: int = 10) -> pd.Series:
        """Chaikin Volatility - Volatility based on high-low spread"""
        # Calculate exponential moving average of high-low spread
        hl_spread = high - low
        ema_spread = hl_spread.ewm(span=window).mean()
        
        # Calculate rate of change of the EMA spread over the period
        chaikin_vol = ((ema_spread - ema_spread.shift(period)) / ema_spread.shift(period)) * 100
        
        return chaikin_vol

    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Pivot Points - Daily support and resistance levels"""
        # Use previous day's data for pivot calculation
        prev_high = high.shift(1)
        prev_low = low.shift(1)  
        prev_close = close.shift(1)
        
        # Standard pivot point calculation
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Support and resistance levels
        r1 = 2 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        
        s1 = 2 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        }

    @staticmethod
    def fibonacci_retracements(high: pd.Series, low: pd.Series, window: int = 50, uptrend: bool = True) -> Dict[str, pd.Series]:
        """Fibonacci Retracements - Based on recent high/low range"""
        # Calculate rolling high and low over the window
        rolling_high = high.rolling(window=window).max()
        rolling_low = low.rolling(window=window).min()
        
        # Calculate range
        range_size = rolling_high - rolling_low
        
        if uptrend:
            # For uptrend: retracements from high
            base = rolling_high
            fib_236 = base - (range_size * 0.236)
            fib_382 = base - (range_size * 0.382) 
            fib_500 = base - (range_size * 0.500)
            fib_618 = base - (range_size * 0.618)
            fib_786 = base - (range_size * 0.786)
        else:
            # For downtrend: retracements from low
            base = rolling_low
            fib_236 = base + (range_size * 0.236)
            fib_382 = base + (range_size * 0.382)
            fib_500 = base + (range_size * 0.500)
            fib_618 = base + (range_size * 0.618)
            fib_786 = base + (range_size * 0.786)
        
        return {
            'fib_236': fib_236,
            'fib_382': fib_382,
            'fib_500': fib_500,
            'fib_618': fib_618,
            'fib_786': fib_786,
            'fib_high': rolling_high,
            'fib_low': rolling_low
        }

    @staticmethod
    def psychological_levels(prices: pd.Series, levels: list = None) -> Dict[str, pd.Series]:
        """Psychological Levels - Round number support/resistance"""
        if levels is None:
            # Default psychological levels (adjust based on typical stock price ranges)
            levels = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        
        # Find the nearest psychological level for each price
        result = {}
        
        for level in levels:
            # Create a series with the constant level value
            level_series = pd.Series([level] * len(prices), index=prices.index)
            result[f'psych_{level}'] = level_series
        
        # Also calculate nearest round levels dynamically
        nearest_round_10 = (prices / 10).round() * 10
        nearest_round_25 = (prices / 25).round() * 25
        nearest_round_50 = (prices / 50).round() * 50
        
        result.update({
            'psych_round_10': nearest_round_10,
            'psych_round_25': nearest_round_25, 
            'psych_round_50': nearest_round_50
        })
        
        return result

    @staticmethod
    def roc(prices: pd.Series, window: int = 12) -> pd.Series:
        """Rate of Change - Momentum oscillator measuring percent change"""
        roc = ((prices - prices.shift(window)) / prices.shift(window)) * 100
        return roc

    @staticmethod
    def tsi(prices: pd.Series, fast: int = 13, slow: int = 25, signal: int = 13) -> Dict[str, pd.Series]:
        """True Strength Index - Double-smoothed momentum oscillator"""
        # Calculate price change
        price_change = prices.diff()
        
        # Calculate absolute price change
        abs_price_change = price_change.abs()
        
        # First smoothing (slow EMA)
        ema_slow_pc = price_change.ewm(span=slow, adjust=False).mean()
        ema_slow_apc = abs_price_change.ewm(span=slow, adjust=False).mean()
        
        # Second smoothing (fast EMA)
        double_smoothed_pc = ema_slow_pc.ewm(span=fast, adjust=False).mean()
        double_smoothed_apc = ema_slow_apc.ewm(span=fast, adjust=False).mean()
        
        # Calculate TSI
        with np.errstate(divide='ignore', invalid='ignore'):
            tsi_line = 100 * (double_smoothed_pc / double_smoothed_apc)
            tsi_line = tsi_line.replace([np.inf, -np.inf], np.nan)
        
        # Calculate signal line
        signal_line = tsi_line.ewm(span=signal, adjust=False).mean()
        
        return {
            'tsi': tsi_line,
            'signal': signal_line
        }

    @staticmethod
    def aroon(high: pd.Series, low: pd.Series, window: int = 25) -> Dict[str, pd.Series]:
        """Aroon Indicator - Identifies trend changes and strength"""
        # Calculate Aroon Up: ((window - periods since window high) / window) * 100
        aroon_up = high.rolling(window=window + 1).apply(
            lambda x: ((window - (window - np.argmax(x))) / window * 100) if len(x) == window + 1 else np.nan,
            raw=True
        )
        
        # Calculate Aroon Down: ((window - periods since window low) / window) * 100
        aroon_down = low.rolling(window=window + 1).apply(
            lambda x: ((window - (window - np.argmin(x))) / window * 100) if len(x) == window + 1 else np.nan,
            raw=True
        )
        
        # Calculate Aroon Oscillator
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
