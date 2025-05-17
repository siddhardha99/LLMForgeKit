"""Logging system for LLMForgeKit."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Set up the logging system.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to a log file
        log_format: Optional custom format for log messages
        
    Returns:
        The configured logger
    """
    # Set default format if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("llmforgekit")
    logger.setLevel(numeric_level)
    
    # Remove any existing handlers to avoid duplicates if reconfigured
    if logger.handlers:
        logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # Create file handler if log_file provided
    if log_file:
        file_path = Path(log_file) if isinstance(log_file, str) else log_file
        file_path = file_path.expanduser()
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


# Create the default logger
logger = setup_logging()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger with an optional name.
    
    Args:
        name: Optional name to append to the base logger name
        
    Returns:
        A configured logger
    """
    if name:
        return logging.getLogger(f"llmforgekit.{name}")
    return logging.getLogger("llmforgekit")