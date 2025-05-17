"""Tests for the logging system."""

import os
import tempfile
from pathlib import Path

# Add the src directory to the Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from llmforgekit.core.logging import setup_logging, get_logger


def test_console_logging():
    """Test logging to the console."""
    # Configure logger
    logger = setup_logging(log_level="DEBUG")
    
    print("\nTesting console logging:")
    
    # Log messages at different levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")


def test_file_logging():
    """Test logging to a file."""
    # Create a temporary directory for the log file
    temp_dir = tempfile.mkdtemp()
    log_file = Path(temp_dir) / "test.log"
    
    # Configure logger with file output
    logger = setup_logging(log_level="DEBUG", log_file=log_file)
    
    print(f"\nTesting file logging to {log_file}:")
    
    # Log messages
    logger.debug("Debug message in file")
    logger.info("Info message in file")
    logger.warning("Warning message in file")
    
    # Check file exists and has content
    assert log_file.exists(), "Log file was not created"
    
    # Read the log file content
    log_content = log_file.read_text()
    print(f"Log file content:\n{log_content}")
    
    # Clean up
    os.unlink(log_file)
    os.rmdir(temp_dir)


def test_named_loggers():
    """Test creating named loggers."""
    # Configure root logger
    setup_logging(log_level="INFO")
    
    print("\nTesting named loggers:")
    
    # Get named loggers
    config_logger = get_logger("config")
    api_logger = get_logger("api")
    
    # Log with different loggers
    config_logger.info("Message from config logger")
    api_logger.info("Message from API logger")


if __name__ == "__main__":
    test_console_logging()
    test_file_logging()
    test_named_loggers()
    print("\nAll logging tests completed!")