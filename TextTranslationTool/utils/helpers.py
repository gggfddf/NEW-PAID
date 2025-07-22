"""
Helper Utility Functions
Common utilities for formatting, calculations, and data processing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
import re
from datetime import datetime, timedelta
import pytz
import logging
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)

def format_currency(amount: float, currency: str = "₹", decimal_places: int = 2) -> str:
    """Format amount as currency with proper Indian number formatting"""
    
    if pd.isna(amount) or amount is None:
        return f"{currency}0.00"
    
    try:
        # Handle very large numbers with appropriate suffixes
        if abs(amount) >= 10_000_000:  # 1 crore
            formatted = f"{currency}{amount/10_000_000:.{decimal_places}f}Cr"
        elif abs(amount) >= 100_000:  # 1 lakh
            formatted = f"{currency}{amount/100_000:.{decimal_places}f}L"
        elif abs(amount) >= 1000:  # 1 thousand
            formatted = f"{currency}{amount/1000:.{decimal_places}f}K"
        else:
            formatted = f"{currency}{amount:.{decimal_places}f}"
        
        return formatted
        
    except (ValueError, TypeError):
        return f"{currency}0.00"

def format_percentage(value: float, decimal_places: int = 2, include_sign: bool = True) -> str:
    """Format value as percentage"""
    
    if pd.isna(value) or value is None:
        return "0.00%"
    
    try:
        sign = "+" if value > 0 and include_sign else ""
        return f"{sign}{value:.{decimal_places}f}%"
    except (ValueError, TypeError):
        return "0.00%"

def format_number(value: float, decimal_places: int = 2, 
                 use_indian_format: bool = True) -> str:
    """Format number with appropriate separators"""
    
    if pd.isna(value) or value is None:
        return "0"
    
    try:
        if use_indian_format:
            # Indian number formatting (lakhs, crores)
            if abs(value) >= 10_000_000:
                return f"{value/10_000_000:.{decimal_places}f}Cr"
            elif abs(value) >= 100_000:
                return f"{value/100_000:.{decimal_places}f}L"
            elif abs(value) >= 1000:
                return f"{value/1000:.{decimal_places}f}K"
            else:
                return f"{value:.{decimal_places}f}"
        else:
            # Standard formatting with commas
            return f"{value:,.{decimal_places}f}"
            
    except (ValueError, TypeError):
        return "0"

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """Calculate returns from price series"""
    
    if len(prices) < 2:
        return pd.Series(dtype=float)
    
    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError("Method must be 'simple' or 'log'")
    
    return returns.dropna()

def calculate_volatility(returns: pd.Series, period: int = 252) -> float:
    """Calculate annualized volatility"""
    
    if len(returns) < 2:
        return 0.0
    
    return returns.std() * np.sqrt(period)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06,
                          period: int = 252) -> float:
    """Calculate Sharpe ratio"""
    
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns.mean() * period - risk_free_rate
    volatility = calculate_volatility(returns, period)
    
    if volatility == 0:
        return 0.0
    
    return excess_returns / volatility

def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, datetime, datetime]:
    """Calculate maximum drawdown and its duration"""
    
    if len(prices) < 2:
        return 0.0, None, None
    
    # Calculate running maximum
    peak = prices.expanding().max()
    
    # Calculate drawdown
    drawdown = (prices - peak) / peak
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # Find peak before max drawdown
    peak_date = peak.loc[:max_dd_date].idxmax()
    
    return max_dd, peak_date, max_dd_date

def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk"""
    
    if len(returns) < 10:
        return 0.0
    
    return np.percentile(returns, confidence_level * 100)

def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    
    if len(returns) < 10:
        return 0.0
    
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def validate_symbol(symbol: str) -> bool:
    """Validate Indian stock symbol format"""
    
    if not symbol:
        return False
    
    # Remove exchange suffix for validation
    base_symbol = symbol.replace('.NS', '').replace('.BSE', '')
    
    # Check length (1-20 characters)
    if len(base_symbol) < 1 or len(base_symbol) > 20:
        return False
    
    # Check for valid characters (alphanumeric, hyphens, ampersands)
    pattern = r'^[A-Za-z0-9\-&]+$'
    if not re.match(pattern, base_symbol):
        return False
    
    return True

def get_exchange_from_symbol(symbol: str) -> str:
    """Extract exchange from symbol"""
    
    if symbol.endswith('.NS'):
        return 'NSE'
    elif symbol.endswith('.BSE'):
        return 'BSE'
    else:
        return 'NSE'  # Default to NSE

def normalize_symbol(symbol: str, exchange: str = 'NSE') -> str:
    """Normalize symbol with proper exchange suffix"""
    
    # Remove existing exchange suffix
    base_symbol = symbol.replace('.NS', '').replace('.BSE', '').upper()
    
    # Add appropriate suffix
    if exchange.upper() == 'BSE':
        return f"{base_symbol}.BSE"
    else:
        return f"{base_symbol}.NS"

def is_market_open(timezone: str = 'Asia/Kolkata') -> Dict[str, Any]:
    """Check if Indian stock market is currently open"""
    
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    
    # Market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
    is_trading_hours = market_start <= now <= market_end
    is_open = is_weekday and is_trading_hours
    
    # Calculate next market session
    if is_open:
        next_event = "Market closes"
        next_time = market_end
    elif is_weekday and now < market_start:
        next_event = "Market opens"
        next_time = market_start
    elif is_weekday and now > market_end:
        next_event = "Market opens"
        next_time = market_start + timedelta(days=1)
    else:
        # Weekend - find next Monday
        days_ahead = (7 - now.weekday()) % 7
        if days_ahead == 0:  # Sunday
            days_ahead = 1
        next_time = market_start + timedelta(days=days_ahead)
        next_event = "Market opens"
    
    return {
        'is_open': is_open,
        'current_time': now.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'market_status': 'OPEN' if is_open else 'CLOSED',
        'next_event': next_event,
        'next_time': next_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'trading_hours': '09:15 - 15:30 IST'
    }

def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix with proper handling of missing values"""
    
    if data.empty:
        return pd.DataFrame()
    
    # Calculate returns if price data is provided
    if data.min().min() > 1:  # Assuming prices are > 1
        returns_data = data.pct_change().dropna()
    else:
        returns_data = data.dropna()
    
    if len(returns_data) < 10:
        return pd.DataFrame()
    
    return returns_data.corr()

def detect_outliers(data: pd.Series, method: str = 'iqr', 
                   threshold: float = 1.5) -> pd.Series:
    """Detect outliers in data series"""
    
    if len(data) < 4:
        return pd.Series(False, index=data.index)
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = z_scores > threshold
        
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return outliers

def safe_divide(numerator: float, denominator: float, 
               default: float = 0.0) -> float:
    """Safely divide two numbers, handling division by zero"""
    
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def round_to_tick_size(price: float, tick_size: float = 0.05) -> float:
    """Round price to valid tick size"""
    
    try:
        if tick_size <= 0:
            return price
        
        # Use Decimal for precise rounding
        price_decimal = Decimal(str(price))
        tick_decimal = Decimal(str(tick_size))
        
        rounded = (price_decimal / tick_decimal).quantize(
            Decimal('1'), rounding=ROUND_HALF_UP
        ) * tick_decimal
        
        return float(rounded)
        
    except (ValueError, TypeError):
        return price

def calculate_position_value(price: float, quantity: int, 
                           lot_size: int = 1) -> float:
    """Calculate total position value"""
    
    try:
        return price * quantity * lot_size
    except (TypeError, ValueError):
        return 0.0

def calculate_brokerage(trade_value: float, brokerage_pct: float = 0.01,
                       min_brokerage: float = 0.0, 
                       max_brokerage: float = float('inf')) -> float:
    """Calculate brokerage with min/max limits"""
    
    try:
        brokerage = trade_value * (brokerage_pct / 100)
        return max(min_brokerage, min(brokerage, max_brokerage))
    except (TypeError, ValueError):
        return 0.0

def time_to_market_close() -> Dict[str, Any]:
    """Calculate time remaining until market close"""
    
    market_info = is_market_open()
    
    if not market_info['is_open']:
        return {
            'hours': 0,
            'minutes': 0,
            'seconds': 0,
            'total_seconds': 0,
            'market_closed': True
        }
    
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    time_diff = market_close - now
    total_seconds = int(time_diff.total_seconds())
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    return {
        'hours': hours,
        'minutes': minutes,
        'seconds': seconds,
        'total_seconds': total_seconds,
        'market_closed': False
    }

def categorize_volume(current_volume: float, avg_volume: float) -> str:
    """Categorize volume relative to average"""
    
    if avg_volume == 0:
        return "Unknown"
    
    ratio = current_volume / avg_volume
    
    if ratio >= 3.0:
        return "Extremely High"
    elif ratio >= 2.0:
        return "Very High"
    elif ratio >= 1.5:
        return "High"
    elif ratio >= 0.8:
        return "Normal"
    elif ratio >= 0.5:
        return "Low"
    else:
        return "Very Low"

def get_trading_session_info() -> Dict[str, Any]:
    """Get detailed trading session information"""
    
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Define session times
    pre_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    post_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Determine current session
    if now.weekday() >= 5:  # Weekend
        session = "Weekend"
        status = "CLOSED"
    elif now < pre_open:
        session = "Pre-Market"
        status = "CLOSED"
    elif pre_open <= now < market_open:
        session = "Pre-Open"
        status = "PRE_OPEN"
    elif market_open <= now <= market_close:
        session = "Regular Trading"
        status = "OPEN"
    elif market_close < now <= post_close:
        session = "Post-Market"
        status = "POST_MARKET"
    else:
        session = "After Hours"
        status = "CLOSED"
    
    return {
        'current_session': session,
        'market_status': status,
        'current_time': now.strftime('%H:%M:%S'),
        'is_trading_day': now.weekday() < 5,
        'sessions': {
            'pre_open': '09:00 - 09:15',
            'regular': '09:15 - 15:30',
            'post_market': '15:30 - 16:00'
        }
    }

def validate_price_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate OHLCV data quality"""
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Missing columns: {missing_columns}")
        return validation_results
    
    if len(df) == 0:
        validation_results['is_valid'] = False
        validation_results['errors'].append("No data available")
        return validation_results
    
    # Check for negative prices
    price_columns = ['Open', 'High', 'Low', 'Close']
    negative_prices = (df[price_columns] <= 0).any()
    if negative_prices.any():
        validation_results['errors'].append("Negative or zero prices detected")
        validation_results['is_valid'] = False
    
    # Check OHLC relationships
    invalid_ohlc = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) |
        (df['High'] < df['Close']) |
        (df['Low'] > df['Open']) |
        (df['Low'] > df['Close'])
    ).any()
    
    if invalid_ohlc:
        validation_results['errors'].append("Invalid OHLC relationships")
        validation_results['is_valid'] = False
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        validation_results['warnings'].append(f"Missing values: {missing_values.to_dict()}")
    
    # Calculate statistics
    validation_results['statistics'] = {
        'total_rows': len(df),
        'date_range': {
            'start': df.index.min().strftime('%Y-%m-%d') if not df.empty else None,
            'end': df.index.max().strftime('%Y-%m-%d') if not df.empty else None
        },
        'price_range': {
            'min': df['Close'].min(),
            'max': df['Close'].max()
        },
        'avg_volume': df['Volume'].mean(),
        'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    return validation_results

def format_time_duration(seconds: int) -> str:
    """Format duration in seconds to human readable format"""
    
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{hours}h {remaining_minutes}m"

def get_market_holidays(year: int = None) -> List[datetime]:
    """Get list of Indian stock market holidays"""
    
    if year is None:
        year = datetime.now().year
    
    # Common holidays (approximate dates - would need to be updated annually)
    holidays = [
        datetime(year, 1, 26),   # Republic Day
        datetime(year, 8, 15),   # Independence Day
        datetime(year, 10, 2),   # Gandhi Jayanti
    ]
    
    # Note: This is a simplified list. In production, you would want to
    # fetch this from an official source or maintain a comprehensive database
    
    return holidays

def is_trading_day(date: datetime = None) -> bool:
    """Check if given date is a trading day"""
    
    if date is None:
        date = datetime.now()
    
    # Check if weekend
    if date.weekday() >= 5:
        return False
    
    # Check if holiday (simplified check)
    holidays = get_market_holidays(date.year)
    return date.date() not in [h.date() for h in holidays]
