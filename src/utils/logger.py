"""
Logging configuration for the Engine Health MLOps pipeline.

Provides structured logging with file and console handlers,
configurable log levels, and consistent formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger(__name__, 'logs/training.log')
        >>> logger.info("Training started")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Default logger for the package
default_logger = setup_logger('engine_health', level=logging.INFO)
