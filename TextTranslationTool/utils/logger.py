"""
Logging Configuration and Setup
Provides comprehensive logging for the Ultimate Market AI Engine
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional
import sys

from config import Config

def setup_logging(log_level: Optional[str] = None, 
                 log_file: Optional[str] = None,
                 max_bytes: int = 10*1024*1024,  # 10MB
                 backup_count: int = 5) -> None:
    """Setup comprehensive logging configuration"""
    
    config = Config()
    
    # Use config level if not specified
    if log_level is None:
        log_level = config.LOG_LEVEL
    
    # Convert string level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    numeric_level = level_map.get(log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Default log file with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_dir, f'market_ai_{timestamp}.log')
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=config.LOG_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Error file handler (errors and above only)
    error_log_file = os.path.join(log_dir, f'market_ai_errors_{datetime.now().strftime("%Y%m%d")}.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # Log the startup
    logger.info("="*60)
    logger.info("Ultimate Market AI Engine - Logging Initialized")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Main Log File: {log_file}")
    logger.info(f"Error Log File: {error_log_file}")
    logger.info("="*60)

def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module"""
    return logging.getLogger(name)

class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return logging.getLogger(self.__class__.__name__)

class PerformanceLogger:
    """Special logger for performance metrics and trading results"""
    
    def __init__(self, log_file: str = None):
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d')
            log_file = f'logs/performance_{timestamp}.log'
        
        # Ensure logs directory exists
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create performance logger
        self.logger = logging.getLogger('performance')
        
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_trade(self, symbol: str, action: str, price: float, 
                 quantity: int, pnl: float = None, **kwargs):
        """Log trade execution details"""
        
        trade_info = f"TRADE | {symbol} | {action} | Price: ₹{price:.2f} | Qty: {quantity}"
        
        if pnl is not None:
            trade_info += f" | P&L: ₹{pnl:.2f}"
        
        # Add any additional info
        for key, value in kwargs.items():
            trade_info += f" | {key}: {value}"
        
        self.logger.info(trade_info)
    
    def log_portfolio_performance(self, total_value: float, daily_return: float,
                                total_return: float, positions: int, **kwargs):
        """Log portfolio performance metrics"""
        
        perf_info = (f"PORTFOLIO | Value: ₹{total_value:,.2f} | "
                    f"Daily Return: {daily_return:.2f}% | "
                    f"Total Return: {total_return:.2f}% | "
                    f"Positions: {positions}")
        
        for key, value in kwargs.items():
            perf_info += f" | {key}: {value}"
        
        self.logger.info(perf_info)
    
    def log_prediction(self, symbol: str, direction: str, confidence: float,
                      target_price: float, model: str = "Ensemble", **kwargs):
        """Log ML prediction details"""
        
        pred_info = (f"PREDICTION | {symbol} | {direction} | "
                    f"Confidence: {confidence:.1f}% | "
                    f"Target: ₹{target_price:.2f} | Model: {model}")
        
        for key, value in kwargs.items():
            pred_info += f" | {key}: {value}"
        
        self.logger.info(pred_info)
    
    def log_pattern_discovery(self, symbol: str, timeframe: str, 
                            patterns_found: int, success_rate: float, **kwargs):
        """Log pattern discovery results"""
        
        pattern_info = (f"PATTERNS | {symbol} | {timeframe} | "
                       f"Found: {patterns_found} | "
                       f"Avg Success Rate: {success_rate:.1f}%")
        
        for key, value in kwargs.items():
            pattern_info += f" | {key}: {value}"
        
        self.logger.info(pattern_info)

class AlertLogger:
    """Logger for alerts and notifications"""
    
    def __init__(self, log_file: str = None):
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d')
            log_file = f'logs/alerts_{timestamp}.log'
        
        self.logger = logging.getLogger('alerts')
        
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def risk_alert(self, message: str, severity: str = "MEDIUM"):
        """Log risk management alerts"""
        alert_msg = f"RISK ALERT [{severity}] | {message}"
        
        if severity == "HIGH":
            self.logger.error(alert_msg)
        elif severity == "MEDIUM":
            self.logger.warning(alert_msg)
        else:
            self.logger.info(alert_msg)
    
    def market_alert(self, symbol: str, message: str, alert_type: str = "INFO"):
        """Log market-related alerts"""
        alert_msg = f"MARKET ALERT | {symbol} | {message}"
        
        if alert_type == "CRITICAL":
            self.logger.error(alert_msg)
        elif alert_type == "WARNING":
            self.logger.warning(alert_msg)
        else:
            self.logger.info(alert_msg)
    
    def system_alert(self, component: str, message: str, severity: str = "INFO"):
        """Log system alerts"""
        alert_msg = f"SYSTEM ALERT | {component} | {message}"
        
        if severity == "ERROR":
            self.logger.error(alert_msg)
        elif severity == "WARNING":
            self.logger.warning(alert_msg)
        else:
            self.logger.info(alert_msg)

def log_execution_time(func):
    """Decorator to log function execution time"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed successfully in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper

def log_exceptions(func):
    """Decorator to automatically log exceptions"""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
            raise
    
    return wrapper

# Global logger instances
performance_logger = PerformanceLogger()
alert_logger = AlertLogger()
