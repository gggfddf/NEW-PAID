"""
Configuration module for the Ultimate Market AI Engine
The only user-editable parameter is the stock symbol.
"""

import os
from typing import List

class Config:
    """Configuration class with stock symbol as the only editable parameter"""
    
    # The ONLY user-editable parameter
    DEFAULT_SYMBOL: str = "RELIANCE.NS"
    
    # Fixed system parameters (not user-editable)
    TIMEFRAMES: List[str] = ["5m", "15m", "1d", "1w"]
    HISTORICAL_DAYS: int = 730  # 2 years of data for ML training
    
    # Market timing (IST)
    MARKET_START_TIME: str = "09:15"
    MARKET_END_TIME: str = "15:30"
    TIMEZONE: str = "Asia/Kolkata"
    
    # Data source configuration
    PRIMARY_DATA_SOURCE: str = "yfinance"
    CACHE_DURATION_MINUTES: int = 5
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 2
    
    # ML Model parameters
    MIN_TRAINING_SAMPLES: int = 100
    PATTERN_SEQUENCE_LENGTHS: List[int] = [3, 5, 8, 13, 20]  # Fibonacci-based
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # Technical Analysis
    TOTAL_INDICATORS: int = 40
    BOLLINGER_PERIOD: int = 20
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    # Deep Learning
    LSTM_LOOKBACK: int = 60
    CNN_KERNEL_SIZE: int = 3
    ATTENTION_HEADS: int = 8
    ENSEMBLE_MODELS: int = 5
    
    # Risk Management
    MAX_POSITION_SIZE_PCT: float = 5.0
    DEFAULT_STOP_LOSS_PCT: float = 2.0
    DEFAULT_TAKE_PROFIT_PCT: float = 6.0
    
    # Portfolio Optimization
    MIN_WEIGHT: float = 0.01
    MAX_WEIGHT: float = 0.30
    RISK_FREE_RATE: float = 0.06  # 6% risk-free rate for India
    
    # Backtesting
    INITIAL_CAPITAL: float = 100000.0
    COMMISSION_PCT: float = 0.1
    SLIPPAGE_PCT: float = 0.05
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # API Configuration (from environment)
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    QUANDL_API_KEY: str = os.getenv("QUANDL_API_KEY", "")
    
    @classmethod
    def get_symbol_with_exchange(cls, symbol: str) -> str:
        """Ensure symbol has proper NSE exchange suffix"""
        if not symbol.endswith('.NS') and not symbol.endswith('.BSE'):
            return f"{symbol}.NS"
        return symbol
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """Basic validation for Indian stock symbols"""
        if not symbol:
            return False
        
        # Remove exchange suffix for validation
        base_symbol = symbol.replace('.NS', '').replace('.BSE', '')
        
        # Basic checks
        if len(base_symbol) < 1 or len(base_symbol) > 20:
            return False
        
        if not base_symbol.replace('-', '').replace('&', '').isalnum():
            return False
        
        return True
    
    @classmethod
    def get_market_hours(cls) -> dict:
        """Get market trading hours in IST"""
        return {
            'start': cls.MARKET_START_TIME,
            'end': cls.MARKET_END_TIME,
            'timezone': cls.TIMEZONE
        }
    
    @classmethod
    def get_ml_config(cls) -> dict:
        """Get ML model configuration"""
        return {
            'min_samples': cls.MIN_TRAINING_SAMPLES,
            'pattern_lengths': cls.PATTERN_SEQUENCE_LENGTHS,
            'confidence_threshold': cls.CONFIDENCE_THRESHOLD,
            'lstm_lookback': cls.LSTM_LOOKBACK,
            'cnn_kernel_size': cls.CNN_KERNEL_SIZE,
            'attention_heads': cls.ATTENTION_HEADS,
            'ensemble_size': cls.ENSEMBLE_MODELS
        }
    
    @classmethod
    def get_risk_config(cls) -> dict:
        """Get risk management configuration"""
        return {
            'max_position_pct': cls.MAX_POSITION_SIZE_PCT,
            'stop_loss_pct': cls.DEFAULT_STOP_LOSS_PCT,
            'take_profit_pct': cls.DEFAULT_TAKE_PROFIT_PCT,
            'risk_free_rate': cls.RISK_FREE_RATE
        }
