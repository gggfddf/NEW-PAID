"""
Advanced Technical Indicator Analysis System
Implements 40+ indicators with pattern detection within each indicator
"""

import pandas as pd
import numpy as np
# import talib - Using custom implementations instead
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Custom TA-Lib replacement functions
class TALibReplacement:
    @staticmethod
    def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2):
        """Bollinger Bands"""
        middle = close.rolling(timeperiod).mean()
        std = close.rolling(timeperiod).std()
        upper = middle + (std * nbdevup)
        lower = middle - (std * nbdevdn)
        return upper, middle, lower
    
    @staticmethod
    def RSI(close, timeperiod=14):
        """Relative Strength Index"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
        """MACD"""
        ema_fast = close.ewm(span=fastperiod).mean()
        ema_slow = close.ewm(span=slowperiod).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signalperiod).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    @staticmethod
    def SMA(close, timeperiod):
        """Simple Moving Average"""
        return close.rolling(timeperiod).mean()
    
    @staticmethod
    def EMA(close, timeperiod):
        """Exponential Moving Average"""
        return close.ewm(span=timeperiod).mean()
    
    @staticmethod
    def STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=fastk_period).min()
        highest_high = high.rolling(window=fastk_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent_slow = k_percent.rolling(window=slowk_period).mean()
        d_percent = k_percent_slow.rolling(window=slowd_period).mean()
        return k_percent_slow, d_percent
    
    @staticmethod
    def WILLR(high, low, close, timeperiod=14):
        """Williams %R"""
        highest_high = high.rolling(timeperiod).max()
        lowest_low = low.rolling(timeperiod).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def CCI(high, low, close, timeperiod=14):
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(timeperiod).mean()
        mad = tp.rolling(timeperiod).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def ROC(close, timeperiod=10):
        """Rate of Change"""
        return ((close - close.shift(timeperiod)) / close.shift(timeperiod)) * 100
    
    @staticmethod
    def MOM(close, timeperiod=10):
        """Momentum"""
        return close - close.shift(timeperiod)
    
    @staticmethod
    def ATR(high, low, close, timeperiod=14):
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(tr).rolling(timeperiod).mean()
        return atr
    
    @staticmethod
    def TRANGE(high, low, close):
        """True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr
    
    @staticmethod
    def ADX(high, low, close, timeperiod=14):
        """Average Directional Index"""
        tr = TALibReplacement.TRANGE(high, low, close)
        dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        dm_minus = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        tr_smooth = pd.Series(tr).rolling(timeperiod).mean()
        dm_plus_smooth = pd.Series(dm_plus).rolling(timeperiod).mean()
        dm_minus_smooth = pd.Series(dm_minus).rolling(timeperiod).mean()
        
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(timeperiod).mean()
        
        return adx
    
    @staticmethod
    def PLUS_DI(high, low, close, timeperiod=14):
        """Plus Directional Indicator"""
        tr = TALibReplacement.TRANGE(high, low, close)
        dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        tr_smooth = pd.Series(tr).rolling(timeperiod).mean()
        dm_plus_smooth = pd.Series(dm_plus).rolling(timeperiod).mean()
        return 100 * dm_plus_smooth / tr_smooth
    
    @staticmethod
    def MINUS_DI(high, low, close, timeperiod=14):
        """Minus Directional Indicator"""
        tr = TALibReplacement.TRANGE(high, low, close)
        dm_minus = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        tr_smooth = pd.Series(tr).rolling(timeperiod).mean()
        dm_minus_smooth = pd.Series(dm_minus).rolling(timeperiod).mean()
        return 100 * dm_minus_smooth / tr_smooth
    
    @staticmethod
    def SAR(high, low, acceleration=0.02, maximum=0.2):
        """Parabolic SAR"""
        # Simplified implementation
        sar = pd.Series(index=high.index, dtype=float)
        sar.iloc[0] = low.iloc[0]
        
        for i in range(1, len(high)):
            if i < len(sar):
                sar.iloc[i] = sar.iloc[i-1] + acceleration * (high.iloc[i-1] - sar.iloc[i-1])
                if sar.iloc[i] > low.iloc[i]:
                    sar.iloc[i] = low.iloc[i]
        
        return sar
    
    @staticmethod
    def OBV(close, volume):
        """On Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def AD(high, low, close, volume):
        """Accumulation/Distribution Line"""
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        mfv = mfm * volume
        ad = mfv.cumsum()
        return ad

# Use the custom implementation
talib = TALibReplacement()
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import Config

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Comprehensive technical analysis with pattern detection for each indicator"""
    
    def __init__(self):
        self.config = Config()
        
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Bollinger Bands with squeeze and band walking pattern detection"""
        
        period = self.config.BOLLINGER_PERIOD
        close = df['Close']
        
        # Standard Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=period)
        
        # Band width for squeeze detection
        bb_width = (bb_upper - bb_lower) / bb_middle * 100
        bb_squeeze = bb_width < bb_width.rolling(20).quantile(0.2)  # Bottom 20%
        
        # Band position
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Band walking patterns
        bb_upper_walk = (close > bb_upper).rolling(3).sum() >= 2
        bb_lower_walk = (close < bb_lower).rolling(3).sum() >= 2
        
        # Expansion patterns
        bb_expansion = bb_width > bb_width.shift(1)
        
        return {
            'BB_Upper': bb_upper,
            'BB_Middle': bb_middle,
            'BB_Lower': bb_lower,
            'BB_Width': bb_width,
            'BB_Squeeze': bb_squeeze.astype(int),
            'BB_Position': bb_position,
            'BB_Upper_Walk': bb_upper_walk.astype(int),
            'BB_Lower_Walk': bb_lower_walk.astype(int),
            'BB_Expansion': bb_expansion.astype(int)
        }
    
    def _calculate_vwap(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """VWAP with flat VWAP significance and reversion patterns"""
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # VWAP slope analysis
        vwap_slope = vwap.diff().rolling(5).mean()
        vwap_flat = (abs(vwap_slope) < vwap.std() * 0.1).astype(int)
        
        # Distance from VWAP
        vwap_distance = (df['Close'] - vwap) / vwap * 100
        
        # Reversion signals
        vwap_far_above = (vwap_distance > 2).astype(int)
        vwap_far_below = (vwap_distance < -2).astype(int)
        
        # Support/Resistance strength
        vwap_support = ((df['Low'] <= vwap) & (df['Close'] > vwap)).rolling(5).sum()
        vwap_resistance = ((df['High'] >= vwap) & (df['Close'] < vwap)).rolling(5).sum()
        
        return {
            'VWAP': vwap,
            'VWAP_Slope': vwap_slope,
            'VWAP_Flat': vwap_flat,
            'VWAP_Distance': vwap_distance,
            'VWAP_Far_Above': vwap_far_above,
            'VWAP_Far_Below': vwap_far_below,
            'VWAP_Support_Strength': vwap_support,
            'VWAP_Resistance_Strength': vwap_resistance
        }
    
    def _calculate_rsi_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """RSI with divergence detection and cluster formation"""
        
        period = self.config.RSI_PERIOD
        rsi = talib.RSI(df['Close'], timeperiod=period)
        
        # RSI levels
        rsi_overbought = (rsi > 70).astype(int)
        rsi_oversold = (rsi < 30).astype(int)
        rsi_neutral = ((rsi >= 40) & (rsi <= 60)).astype(int)
        
        # RSI divergence detection
        close_peaks = argrelextrema(df['Close'].values, np.greater, order=5)[0]
        close_troughs = argrelextrema(df['Close'].values, np.less, order=5)[0]
        rsi_peaks = argrelextrema(rsi.values, np.greater, order=5)[0]
        rsi_troughs = argrelextrema(rsi.values, np.less, order=5)[0]
        
        # Bullish divergence: price makes lower lows, RSI makes higher lows
        bullish_divergence = pd.Series(0, index=df.index)
        bearish_divergence = pd.Series(0, index=df.index)
        
        for i in range(1, len(close_troughs)):
            if len(rsi_troughs) > i:
                price_idx1, price_idx2 = close_troughs[i-1], close_troughs[i]
                
                # Find corresponding RSI troughs
                rsi_idx1 = rsi_troughs[rsi_troughs <= price_idx1]
                rsi_idx2 = rsi_troughs[rsi_troughs <= price_idx2]
                
                if len(rsi_idx1) > 0 and len(rsi_idx2) > 0:
                    rsi_idx1, rsi_idx2 = rsi_idx1[-1], rsi_idx2[-1]
                    
                    if (df['Close'].iloc[price_idx2] < df['Close'].iloc[price_idx1] and
                        rsi.iloc[rsi_idx2] > rsi.iloc[rsi_idx1]):
                        bullish_divergence.iloc[price_idx2] = 1
        
        # Similar for bearish divergence
        for i in range(1, len(close_peaks)):
            if len(rsi_peaks) > i:
                price_idx1, price_idx2 = close_peaks[i-1], close_peaks[i]
                
                rsi_idx1 = rsi_peaks[rsi_peaks <= price_idx1]
                rsi_idx2 = rsi_peaks[rsi_peaks <= price_idx2]
                
                if len(rsi_idx1) > 0 and len(rsi_idx2) > 0:
                    rsi_idx1, rsi_idx2 = rsi_idx1[-1], rsi_idx2[-1]
                    
                    if (df['Close'].iloc[price_idx2] > df['Close'].iloc[price_idx1] and
                        rsi.iloc[rsi_idx2] < rsi.iloc[rsi_idx1]):
                        bearish_divergence.iloc[price_idx2] = 1
        
        # RSI momentum
        rsi_momentum = rsi.diff().rolling(3).mean()
        
        return {
            'RSI': rsi,
            'RSI_Overbought': rsi_overbought,
            'RSI_Oversold': rsi_oversold,
            'RSI_Neutral': rsi_neutral,
            'RSI_Bullish_Divergence': bullish_divergence,
            'RSI_Bearish_Divergence': bearish_divergence,
            'RSI_Momentum': rsi_momentum
        }
    
    def _calculate_macd_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """MACD with histogram patterns and momentum shifts"""
        
        fast = self.config.MACD_FAST
        slow = self.config.MACD_SLOW
        signal = self.config.MACD_SIGNAL
        
        macd, signal_line, histogram = talib.MACD(df['Close'], fastperiod=fast, 
                                                 slowperiod=slow, signalperiod=signal)
        
        # MACD crossovers
        macd_bullish_cross = ((macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))).astype(int)
        macd_bearish_cross = ((macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))).astype(int)
        
        # Zero line crossovers
        macd_bull_zero = ((macd > 0) & (macd.shift(1) <= 0)).astype(int)
        macd_bear_zero = ((macd < 0) & (macd.shift(1) >= 0)).astype(int)
        
        # Histogram patterns
        hist_increasing = (histogram > histogram.shift(1)).astype(int)
        hist_decreasing = (histogram < histogram.shift(1)).astype(int)
        
        # Momentum acceleration
        macd_acceleration = histogram.diff()
        
        # MACD convergence/divergence
        macd_convergence = (abs(macd - signal_line) < abs(macd.shift(1) - signal_line.shift(1))).astype(int)
        
        return {
            'MACD': macd,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram,
            'MACD_Bull_Cross': macd_bullish_cross,
            'MACD_Bear_Cross': macd_bearish_cross,
            'MACD_Bull_Zero': macd_bull_zero,
            'MACD_Bear_Zero': macd_bear_zero,
            'MACD_Hist_Inc': hist_increasing,
            'MACD_Hist_Dec': hist_decreasing,
            'MACD_Acceleration': macd_acceleration,
            'MACD_Convergence': macd_convergence
        }
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Volume indicators with accumulation patterns and breakout detection"""
        
        # On Balance Volume
        obv = talib.OBV(df['Close'], df['Volume'])
        
        # Accumulation/Distribution Line
        ad_line = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume Rate of Change
        volume_roc = df['Volume'].pct_change(periods=10) * 100
        
        # Volume moving average
        volume_ma = df['Volume'].rolling(20).mean()
        volume_ratio = df['Volume'] / volume_ma
        
        # High volume patterns
        high_volume = (volume_ratio > 2).astype(int)
        low_volume = (volume_ratio < 0.5).astype(int)
        
        # Volume trend
        volume_trend = df['Volume'].rolling(10).apply(lambda x: stats.linregress(range(len(x)), x)[0])
        
        # Price-Volume confirmation
        price_up_volume_up = ((df['Close'] > df['Close'].shift(1)) & 
                             (df['Volume'] > volume_ma)).astype(int)
        price_down_volume_up = ((df['Close'] < df['Close'].shift(1)) & 
                               (df['Volume'] > volume_ma)).astype(int)
        
        return {
            'OBV': obv,
            'AD_Line': ad_line,
            'Volume_ROC': volume_roc,
            'Volume_Ratio': volume_ratio,
            'High_Volume': high_volume,
            'Low_Volume': low_volume,
            'Volume_Trend': volume_trend,
            'Price_Up_Vol_Up': price_up_volume_up,
            'Price_Down_Vol_Up': price_down_volume_up
        }
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Moving averages with dynamic support/resistance detection"""
        
        close = df['Close']
        
        # Multiple timeframe MAs
        sma_5 = talib.SMA(close, timeperiod=5)
        sma_10 = talib.SMA(close, timeperiod=10)
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        sma_200 = talib.SMA(close, timeperiod=200)
        
        ema_12 = talib.EMA(close, timeperiod=12)
        ema_26 = talib.EMA(close, timeperiod=26)
        
        # MA crossovers
        golden_cross = ((sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))).astype(int)
        death_cross = ((sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))).astype(int)
        
        # Price relative to MAs
        above_all_ma = ((close > sma_5) & (close > sma_10) & 
                       (close > sma_20) & (close > sma_50)).astype(int)
        below_all_ma = ((close < sma_5) & (close < sma_10) & 
                       (close < sma_20) & (close < sma_50)).astype(int)
        
        # MA slope analysis
        sma_20_slope = sma_20.diff().rolling(3).mean()
        sma_50_slope = sma_50.diff().rolling(5).mean()
        
        # Support/Resistance at MAs
        sma_20_support = ((df['Low'] <= sma_20) & (df['Close'] > sma_20)).rolling(3).sum()
        sma_20_resistance = ((df['High'] >= sma_20) & (df['Close'] < sma_20)).rolling(3).sum()
        
        return {
            'SMA_5': sma_5,
            'SMA_10': sma_10,
            'SMA_20': sma_20,
            'SMA_50': sma_50,
            'SMA_200': sma_200,
            'EMA_12': ema_12,
            'EMA_26': ema_26,
            'Golden_Cross': golden_cross,
            'Death_Cross': death_cross,
            'Above_All_MA': above_all_ma,
            'Below_All_MA': below_all_ma,
            'SMA_20_Slope': sma_20_slope,
            'SMA_50_Slope': sma_50_slope,
            'SMA_20_Support': sma_20_support,
            'SMA_20_Resistance': sma_20_resistance
        }
    
    def _calculate_momentum_oscillators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Advanced momentum oscillators with pattern detection"""
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close)
        
        # Williams %R
        willr = talib.WILLR(high, low, close, timeperiod=14)
        
        # Commodity Channel Index
        cci = talib.CCI(high, low, close, timeperiod=14)
        
        # Rate of Change
        roc = talib.ROC(close, timeperiod=10)
        
        # Momentum
        momentum = talib.MOM(close, timeperiod=10)
        
        # Stochastic patterns
        stoch_oversold = ((slowk < 20) & (slowd < 20)).astype(int)
        stoch_overbought = ((slowk > 80) & (slowd > 80)).astype(int)
        stoch_bull_cross = ((slowk > slowd) & (slowk.shift(1) <= slowd.shift(1))).astype(int)
        stoch_bear_cross = ((slowk < slowd) & (slowk.shift(1) >= slowd.shift(1))).astype(int)
        
        # Williams %R patterns
        willr_oversold = (willr < -80).astype(int)
        willr_overbought = (willr > -20).astype(int)
        
        # CCI patterns
        cci_extreme_bullish = (cci > 100).astype(int)
        cci_extreme_bearish = (cci < -100).astype(int)
        
        return {
            'Stoch_K': slowk,
            'Stoch_D': slowd,
            'Williams_R': willr,
            'CCI': cci,
            'ROC': roc,
            'Momentum': momentum,
            'Stoch_Oversold': stoch_oversold,
            'Stoch_Overbought': stoch_overbought,
            'Stoch_Bull_Cross': stoch_bull_cross,
            'Stoch_Bear_Cross': stoch_bear_cross,
            'WillR_Oversold': willr_oversold,
            'WillR_Overbought': willr_overbought,
            'CCI_Extreme_Bull': cci_extreme_bullish,
            'CCI_Extreme_Bear': cci_extreme_bearish
        }
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Volatility indicators with expansion/contraction patterns"""
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Average True Range
        atr = talib.ATR(high, low, close, timeperiod=14)
        
        # True Range
        tr = talib.TRANGE(high, low, close)
        
        # Volatility expansion/contraction
        atr_ma = atr.rolling(20).mean()
        atr_expansion = (atr > atr_ma * 1.5).astype(int)
        atr_contraction = (atr < atr_ma * 0.5).astype(int)
        
        # Volatility trend
        atr_trend = atr.rolling(10).apply(lambda x: stats.linregress(range(len(x)), x)[0])
        
        # Historical volatility
        returns = close.pct_change()
        hist_volatility = returns.rolling(20).std() * np.sqrt(252) * 100
        
        # Volatility percentile
        vol_percentile = hist_volatility.rolling(252).rank(pct=True) * 100
        
        return {
            'ATR': atr,
            'True_Range': tr,
            'ATR_Expansion': atr_expansion,
            'ATR_Contraction': atr_contraction,
            'ATR_Trend': atr_trend,
            'Hist_Volatility': hist_volatility,
            'Vol_Percentile': vol_percentile
        }
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Trend indicators with strength and direction analysis"""
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Average Directional Index
        adx = talib.ADX(high, low, close, timeperiod=14)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Parabolic SAR
        sar = talib.SAR(high, low)
        
        # Trend strength categories
        strong_trend = (adx > 25).astype(int)
        weak_trend = (adx < 20).astype(int)
        
        # Trend direction
        bullish_trend = (plus_di > minus_di).astype(int)
        bearish_trend = (plus_di < minus_di).astype(int)
        
        # SAR signals
        sar_bullish = (close > sar).astype(int)
        sar_bearish = (close < sar).astype(int)
        sar_flip_bull = ((close > sar) & (close.shift(1) <= sar.shift(1))).astype(int)
        sar_flip_bear = ((close < sar) & (close.shift(1) >= sar.shift(1))).astype(int)
        
        return {
            'ADX': adx,
            'Plus_DI': plus_di,
            'Minus_DI': minus_di,
            'SAR': sar,
            'Strong_Trend': strong_trend,
            'Weak_Trend': weak_trend,
            'Bullish_Trend': bullish_trend,
            'Bearish_Trend': bearish_trend,
            'SAR_Bullish': sar_bullish,
            'SAR_Bearish': sar_bearish,
            'SAR_Flip_Bull': sar_flip_bull,
            'SAR_Flip_Bear': sar_flip_bear
        }
    
    def _calculate_custom_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Custom composite indicators with proprietary calculations"""
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # 1. Price-Volume Momentum Indicator
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        pv_momentum = (price_change * volume_change).rolling(5).mean()
        
        # 2. Dynamic Support/Resistance Levels
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        support_strength = ((low - rolling_low) / rolling_low * 100).rolling(5).min()
        resistance_strength = ((rolling_high - high) / rolling_high * 100).rolling(5).min()
        
        # 3. Trend Consistency Index
        ma_5 = close.rolling(5).mean()
        ma_10 = close.rolling(10).mean()
        ma_20 = close.rolling(20).mean()
        
        trend_consistency = (
            ((ma_5 > ma_10) & (ma_10 > ma_20)).astype(int) * 2 +
            ((ma_5 < ma_10) & (ma_10 < ma_20)).astype(int) * -2 +
            ((ma_5 > ma_10) & (ma_10 < ma_20)).astype(int) * 1 +
            ((ma_5 < ma_10) & (ma_10 > ma_20)).astype(int) * -1
        )
        
        # 4. Volatility-Adjusted Momentum
        returns = close.pct_change()
        volatility = returns.rolling(14).std()
        vol_adj_momentum = returns.rolling(5).mean() / (volatility + 1e-6)
        
        # 5. Multi-Timeframe Confluence
        sma_alignment = (
            (close > close.rolling(5).mean()).astype(int) +
            (close > close.rolling(10).mean()).astype(int) +
            (close > close.rolling(20).mean()).astype(int) +
            (close > close.rolling(50).mean()).astype(int)
        )
        
        # 6. Breakout Probability Indicator
        atr = talib.ATR(high, low, close, timeperiod=14)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        breakout_probability = (
            (abs(close - bb_middle) / (bb_width * bb_middle)) * 100
        ).rolling(5).mean()
        
        return {
            'PV_Momentum': pv_momentum,
            'Support_Strength': support_strength,
            'Resistance_Strength': resistance_strength,
            'Trend_Consistency': trend_consistency,
            'Vol_Adj_Momentum': vol_adj_momentum,
            'SMA_Alignment': sma_alignment,
            'Breakout_Probability': breakout_probability
        }
    
    def _calculate_confluence_score(self, indicators_df: pd.DataFrame) -> pd.Series:
        """Calculate confluence score across all indicators"""
        
        # Select key bullish signals
        bullish_signals = [
            'RSI_Oversold', 'MACD_Bull_Cross', 'Stoch_Oversold', 'Golden_Cross',
            'Above_All_MA', 'Strong_Trend', 'Bullish_Trend', 'SAR_Flip_Bull',
            'BB_Lower_Walk', 'High_Volume', 'Price_Up_Vol_Up'
        ]
        
        # Select key bearish signals
        bearish_signals = [
            'RSI_Overbought', 'MACD_Bear_Cross', 'Stoch_Overbought', 'Death_Cross',
            'Below_All_MA', 'Bearish_Trend', 'SAR_Flip_Bear',
            'BB_Upper_Walk', 'Price_Down_Vol_Up'
        ]
        
        # Calculate bullish confluence
        bullish_score = indicators_df[
            [col for col in bullish_signals if col in indicators_df.columns]
        ].sum(axis=1)
        
        # Calculate bearish confluence
        bearish_score = indicators_df[
            [col for col in bearish_signals if col in indicators_df.columns]
        ].sum(axis=1)
        
        # Net confluence score
        confluence_score = bullish_score - bearish_score
        
        return confluence_score
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all 40+ technical indicators with pattern analysis"""
        
        logger.info("Calculating advanced technical indicators...")
        
        if len(df) < 50:
            logger.warning("Insufficient data for technical analysis")
            return pd.DataFrame()
        
        # Start with original data
        result_df = df.copy()
        
        try:
            # Calculate all indicator groups
            bollinger_data = self._calculate_bollinger_bands(df)
            vwap_data = self._calculate_vwap(df)
            rsi_data = self._calculate_rsi_patterns(df)
            macd_data = self._calculate_macd_patterns(df)
            volume_data = self._calculate_volume_indicators(df)
            ma_data = self._calculate_moving_averages(df)
            momentum_data = self._calculate_momentum_oscillators(df)
            volatility_data = self._calculate_volatility_indicators(df)
            trend_data = self._calculate_trend_indicators(df)
            custom_data = self._calculate_custom_indicators(df)
            
            # Combine all indicators
            all_indicators = {
                **bollinger_data,
                **vwap_data,
                **rsi_data,
                **macd_data,
                **volume_data,
                **ma_data,
                **momentum_data,
                **volatility_data,
                **trend_data,
                **custom_data
            }
            
            # Add indicators to dataframe
            for name, series in all_indicators.items():
                if len(series) == len(result_df):
                    result_df[name] = series
                else:
                    # Handle length mismatches
                    if len(series) > len(result_df):
                        result_df[name] = series.iloc[:len(result_df)]
                    else:
                        # Pad with NaN
                        padded_series = pd.Series(np.nan, index=result_df.index)
                        padded_series.iloc[-len(series):] = series.values
                        result_df[name] = padded_series
            
            # Calculate confluence score
            result_df['Confluence_Score'] = self._calculate_confluence_score(result_df)
            
            # Generate overall signal
            result_df['Overall_Signal'] = np.where(
                result_df['Confluence_Score'] > 3, 1,  # Bullish
                np.where(result_df['Confluence_Score'] < -3, -1, 0)  # Bearish or Neutral
            )
            
            logger.info(f"Successfully calculated {len(all_indicators)} technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
        
        return result_df
    
    def get_indicator_summary(self, indicators_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary of current indicator states"""
        
        if indicators_df.empty:
            return {}
        
        latest = indicators_df.iloc[-1]
        
        summary = {
            'trend_indicators': {
                'adx': latest.get('ADX', 0),
                'trend_strength': 'Strong' if latest.get('ADX', 0) > 25 else 'Weak',
                'trend_direction': 'Bullish' if latest.get('Bullish_Trend', 0) else 'Bearish',
                'sar_signal': 'Bullish' if latest.get('SAR_Bullish', 0) else 'Bearish'
            },
            'momentum_indicators': {
                'rsi': latest.get('RSI', 50),
                'rsi_signal': ('Overbought' if latest.get('RSI', 50) > 70 else
                              'Oversold' if latest.get('RSI', 50) < 30 else 'Neutral'),
                'macd_signal': ('Bullish' if latest.get('MACD_Bull_Cross', 0) else
                               'Bearish' if latest.get('MACD_Bear_Cross', 0) else 'Neutral'),
                'stochastic_signal': ('Overbought' if latest.get('Stoch_Overbought', 0) else
                                     'Oversold' if latest.get('Stoch_Oversold', 0) else 'Neutral')
            },
            'volatility_indicators': {
                'atr': latest.get('ATR', 0),
                'volatility_state': ('Expanding' if latest.get('ATR_Expansion', 0) else
                                    'Contracting' if latest.get('ATR_Contraction', 0) else 'Normal'),
                'bb_squeeze': bool(latest.get('BB_Squeeze', 0))
            },
            'volume_indicators': {
                'volume_trend': 'High' if latest.get('High_Volume', 0) else 'Normal',
                'price_volume_confirmation': bool(latest.get('Price_Up_Vol_Up', 0)),
                'accumulation_signal': latest.get('AD_Line', 0)
            },
            'confluence': {
                'confluence_score': latest.get('Confluence_Score', 0),
                'overall_signal': ('Bullish' if latest.get('Overall_Signal', 0) > 0 else
                                  'Bearish' if latest.get('Overall_Signal', 0) < 0 else 'Neutral'),
                'signal_strength': abs(latest.get('Confluence_Score', 0))
            }
        }
        
        return summary
    
    def detect_indicator_patterns(self, indicators_df: pd.DataFrame, 
                                lookback_periods: int = 20) -> Dict[str, List[str]]:
        """Detect patterns within individual indicators"""
        
        if len(indicators_df) < lookback_periods:
            return {}
        
        recent_data = indicators_df.tail(lookback_periods)
        patterns = {
            'bullish_patterns': [],
            'bearish_patterns': [],
            'neutral_patterns': []
        }
        
        try:
            # RSI patterns
            if 'RSI' in recent_data.columns:
                rsi_recent = recent_data['RSI']
                if len(rsi_recent.dropna()) > 5:
                    if rsi_recent.iloc[-1] < 30 and rsi_recent.iloc[-3:].min() == rsi_recent.iloc[-1]:
                        patterns['bullish_patterns'].append('RSI Double Bottom in Oversold')
                    elif rsi_recent.iloc[-1] > 70 and rsi_recent.iloc[-3:].max() == rsi_recent.iloc[-1]:
                        patterns['bearish_patterns'].append('RSI Double Top in Overbought')
            
            # Bollinger Band patterns
            if all(col in recent_data.columns for col in ['BB_Squeeze', 'BB_Expansion']):
                if recent_data['BB_Squeeze'].iloc[-3:].sum() >= 2 and recent_data['BB_Expansion'].iloc[-1]:
                    patterns['bullish_patterns'].append('Bollinger Band Squeeze Breakout')
            
            # MACD patterns
            if 'MACD_Histogram' in recent_data.columns:
                macd_hist = recent_data['MACD_Histogram']
                if len(macd_hist.dropna()) > 5:
                    # Histogram divergence
                    if (macd_hist.iloc[-1] > macd_hist.iloc[-2] > macd_hist.iloc[-3] and
                        macd_hist.iloc[-3] < 0):
                        patterns['bullish_patterns'].append('MACD Histogram Bullish Divergence')
            
            # Volume patterns
            if 'Volume_Trend' in recent_data.columns:
                vol_trend = recent_data['Volume_Trend']
                if len(vol_trend.dropna()) > 3:
                    if vol_trend.iloc[-3:].mean() > 0:
                        patterns['bullish_patterns'].append('Increasing Volume Trend')
                    elif vol_trend.iloc[-3:].mean() < 0:
                        patterns['bearish_patterns'].append('Decreasing Volume Trend')
            
            # Moving average patterns
            if recent_data['Golden_Cross'].sum() > 0:
                patterns['bullish_patterns'].append('Golden Cross Detected')
            elif recent_data['Death_Cross'].sum() > 0:
                patterns['bearish_patterns'].append('Death Cross Detected')
            
            # Confluence patterns
            if 'Confluence_Score' in recent_data.columns:
                conf_scores = recent_data['Confluence_Score']
                if len(conf_scores.dropna()) > 3:
                    if conf_scores.iloc[-3:].mean() > 4:
                        patterns['bullish_patterns'].append('Strong Bullish Confluence')
                    elif conf_scores.iloc[-3:].mean() < -4:
                        patterns['bearish_patterns'].append('Strong Bearish Confluence')
                    else:
                        patterns['neutral_patterns'].append('Mixed Signals')
        
        except Exception as e:
            logger.error(f"Error detecting indicator patterns: {e}")
        
        return patterns
