<<<<<<< HEAD
# monitoring/logger.py

import logging
import logging.handlers
import os
from datetime import datetime


def setup_logger(config: dict) -> logging.Logger:
    """
    Setup structured logging with file rotation
    """
    # Create logs directory if not exists
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('TradingSystem')
    logger.setLevel(getattr(logging, config.get('level', 'INFO')))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=config.get('file', 'logs/trading.log'),
        maxBytes=config.get('max_bytes', 10485760),  # 10MB
        backupCount=config.get('backup_count', 5),
        encoding='utf-8'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
=======
# monitoring/logger.py

import logging
import logging.handlers
import os
from datetime import datetime


def setup_logger(config: dict) -> logging.Logger:
    """
    Setup structured logging with file rotation
    """
    # Create logs directory if not exists
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('TradingSystem')
    logger.setLevel(getattr(logging, config.get('level', 'INFO')))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=config.get('file', 'logs/trading.log'),
        maxBytes=config.get('max_bytes', 10485760),  # 10MB
        backupCount=config.get('backup_count', 5),
        encoding='utf-8'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
>>>>>>> da5aa461dbb0f1589a9d5c3ad9fb3a175f2f7bf6
    return logger