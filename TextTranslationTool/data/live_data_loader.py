"""
Live Data Loader with Multi-Timeframe Support
Fetches real-time and historical market data for Indian stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp
import pytz
from typing import Dict, List, Optional, Any
import logging
from functools import lru_cache
import time

from config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """Advanced data loader for Indian market data with intelligent caching"""
    
    def __init__(self):
        self.config = Config()
        self.cache = {}
        self.cache_timestamps = {}
        self.ist_tz = pytz.timezone(self.config.TIMEZONE)
        
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours (IST)"""
        now_ist = datetime.now(self.ist_tz)
        market_start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Check if it's a weekday and within market hours
        is_weekday = now_ist.weekday() < 5
        is_market_time = market_start <= now_ist <= market_end
        
        return is_weekday and is_market_time
    
    def _should_use_cache(self, cache_key: str) -> bool:
        """Determine if cached data should be used"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache_timestamps.get(cache_key, 0)
        current_time = time.time()
        
        # Use longer cache during market hours for real-time data
        cache_duration = 60 if self._is_market_hours() else 300  # 1 min vs 5 min
        
        return (current_time - cache_time) < cache_duration
    
    def _cache_data(self, cache_key: str, data: Any) -> None:
        """Cache data with timestamp"""
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = time.time()
    
    def _clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data"""
        if df.empty:
            return df
        
        # Remove rows with NaN values in critical columns
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Ensure OHLC relationships are valid
        df = df[df['High'] >= df['Low']]
        df = df[df['High'] >= df['Open']]
        df = df[df['High'] >= df['Close']]
        df = df[df['Low'] <= df['Open']]
        df = df[df['Low'] <= df['Close']]
        
        # Remove zero or negative prices
        df = df[df['Open'] > 0]
        df = df[df['High'] > 0]
        df = df[df['Low'] > 0]
        df = df[df['Close'] > 0]
        
        # Remove unrealistic volume (negative or extremely high)
        median_volume = df['Volume'].median()
        df = df[df['Volume'] >= 0]
        df = df[df['Volume'] <= median_volume * 100]  # Remove outliers
        
        return df
    
    def _apply_corporate_actions(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Apply corporate action adjustments"""
        try:
            # Get corporate actions from yfinance
            ticker = yf.Ticker(symbol)
            actions = ticker.actions
            
            if not actions.empty:
                # Apply dividend adjustments
                for date, row in actions.iterrows():
                    if date in df.index:
                        if 'Dividends' in row and row['Dividends'] > 0:
                            # Adjust prices before dividend date
                            mask = df.index < date
                            adjustment_factor = 1 - (row['Dividends'] / df.loc[date, 'Close'])
                            df.loc[mask, ['Open', 'High', 'Low', 'Close']] *= adjustment_factor
                        
                        if 'Stock Splits' in row and row['Stock Splits'] > 0:
                            # Adjust for stock splits
                            mask = df.index < date
                            split_factor = row['Stock Splits']
                            df.loc[mask, ['Open', 'High', 'Low', 'Close']] /= split_factor
                            df.loc[mask, 'Volume'] *= split_factor
            
            return df
        except Exception as e:
            logger.warning(f"Could not apply corporate actions for {symbol}: {e}")
            return df
    
    async def _fetch_single_timeframe(self, symbol: str, timeframe: str, 
                                    period: str = "2y") -> Optional[pd.DataFrame]:
        """Fetch data for a single timeframe with retries"""
        
        cache_key = f"{symbol}_{timeframe}_{period}"
        
        if self._should_use_cache(cache_key):
            logger.info(f"Using cached data for {symbol} {timeframe}")
            return self.cache[cache_key]
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                logger.info(f"Fetching {symbol} data for {timeframe} (attempt {attempt + 1})")
                
                # Use yfinance to fetch data
                ticker = yf.Ticker(symbol)
                
                # Map timeframe to yfinance interval
                interval_map = {
                    '5m': '5m',
                    '15m': '15m',
                    '1d': '1d',
                    '1w': '1wk'
                }
                
                interval = interval_map.get(timeframe, '1d')
                
                # Fetch data
                df = ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    prepost=False,
                    threads=True
                )
                
                if df.empty:
                    logger.warning(f"No data received for {symbol} {timeframe}")
                    continue
                
                # Clean the data
                df = self._clean_ohlcv_data(df)
                
                if df.empty:
                    logger.warning(f"No valid data after cleaning for {symbol} {timeframe}")
                    continue
                
                # Apply corporate actions
                df = self._apply_corporate_actions(df, symbol)
                
                # Convert timezone to IST
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(self.ist_tz)
                else:
                    df.index = df.index.tz_localize('UTC').tz_convert(self.ist_tz)
                
                # Cache the data
                self._cache_data(cache_key, df)
                
                logger.info(f"Successfully fetched {len(df)} rows for {symbol} {timeframe}")
                return df
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} {timeframe} (attempt {attempt + 1}): {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    await asyncio.sleep(self.config.RETRY_DELAY_SECONDS * (attempt + 1))
                else:
                    logger.error(f"Failed to fetch {symbol} {timeframe} after all retries")
        
        return None
    
    async def fetch_multi_timeframe_data(self, symbol: str, 
                                       timeframes: List[str],
                                       lookback_days: int = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes simultaneously"""
        
        if not self.config.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        symbol = self.config.get_symbol_with_exchange(symbol)
        lookback_days = lookback_days or self.config.HISTORICAL_DAYS
        
        # Calculate period string
        if lookback_days <= 7:
            period = "7d"
        elif lookback_days <= 30:
            period = "1mo"
        elif lookback_days <= 90:
            period = "3mo"
        elif lookback_days <= 180:
            period = "6mo"
        elif lookback_days <= 365:
            period = "1y"
        else:
            period = "2y"
        
        logger.info(f"Fetching multi-timeframe data for {symbol}")
        
        # Create tasks for concurrent fetching
        tasks = [
            self._fetch_single_timeframe(symbol, tf, period)
            for tf in timeframes
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_dict = {}
        for timeframe, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {timeframe}: {result}")
                continue
            
            if result is not None and not result.empty:
                # Ensure minimum data requirements
                if len(result) >= self.config.MIN_TRAINING_SAMPLES:
                    data_dict[timeframe] = result
                else:
                    logger.warning(f"Insufficient data for {timeframe}: {len(result)} rows")
            else:
                logger.warning(f"No data available for {timeframe}")
        
        if not data_dict:
            raise ValueError(f"No valid data fetched for {symbol}")
        
        logger.info(f"Successfully fetched data for {len(data_dict)} timeframes")
        return data_dict
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        try:
            symbol = self.config.get_symbol_with_exchange(symbol)
            ticker = yf.Ticker(symbol)
            
            # Try to get real-time price
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if current_price:
                return float(current_price)
            
            # Fallback to last close price
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        now_ist = datetime.now(self.ist_tz)
        
        market_start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_weekday = now_ist.weekday() < 5
        is_open = is_weekday and market_start <= now_ist <= market_end
        
        if is_open:
            status = "OPEN"
            next_event = "Market closes"
            next_time = market_end
        elif is_weekday and now_ist < market_start:
            status = "PRE_MARKET"
            next_event = "Market opens"
            next_time = market_start
        elif is_weekday and now_ist > market_end:
            status = "AFTER_MARKET"
            next_event = "Market opens"
            next_time = market_start + timedelta(days=1)
        else:
            # Weekend
            status = "CLOSED"
            next_event = "Market opens"
            days_ahead = (7 - now_ist.weekday()) % 7
            if days_ahead == 0:  # Sunday
                days_ahead = 1
            next_time = market_start + timedelta(days=days_ahead)
        
        return {
            'status': status,
            'is_open': is_open,
            'current_time': now_ist.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'next_event': next_event,
            'next_time': next_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'market_hours': f"{market_start.strftime('%H:%M')} - {market_end.strftime('%H:%M')} IST"
        }
    
    def validate_data_quality(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate the quality of fetched data"""
        quality_report = {
            'overall_score': 0,
            'timeframe_scores': {},
            'issues': [],
            'recommendations': []
        }
        
        total_score = 0
        valid_timeframes = 0
        
        for timeframe, df in data.items():
            score = 100
            issues = []
            
            # Check data completeness
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                score -= 30
                issues.append(f"Insufficient data: {len(df)} rows")
            
            # Check for missing values
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            if missing_pct > 5:
                score -= 20
                issues.append(f"High missing data: {missing_pct:.1f}%")
            
            # Check data freshness
            latest_date = df.index[-1]
            days_old = (datetime.now(self.ist_tz) - latest_date).days
            if days_old > 1:
                score -= 15
                issues.append(f"Stale data: {days_old} days old")
            
            # Check price consistency
            price_changes = df['Close'].pct_change().abs()
            extreme_moves = (price_changes > 0.2).sum()  # More than 20% moves
            if extreme_moves > len(df) * 0.05:  # More than 5% of data
                score -= 10
                issues.append(f"Unusual price movements: {extreme_moves} extreme moves")
            
            quality_report['timeframe_scores'][timeframe] = {
                'score': max(0, score),
                'issues': issues,
                'data_points': len(df),
                'date_range': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
            }
            
            total_score += max(0, score)
            valid_timeframes += 1
        
        if valid_timeframes > 0:
            quality_report['overall_score'] = total_score / valid_timeframes
        
        # Generate recommendations
        if quality_report['overall_score'] < 70:
            quality_report['recommendations'].append("Consider using alternative data sources")
        
        if quality_report['overall_score'] < 50:
            quality_report['recommendations'].append("Data quality is poor - results may be unreliable")
        
        return quality_report
