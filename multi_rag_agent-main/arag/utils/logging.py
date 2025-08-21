"""
Logging utilities for the ARAG system.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler
import time

from arag.config import LOG_LEVEL, LOG_FORMAT

def setup_logging(
    log_dir: str = "logs",
    log_level: str = LOG_LEVEL,
    log_format: str = LOG_FORMAT
) -> None:
    """
    Set up logging for the ARAG system.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        log_format: Format for log messages
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up root logger
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set up file handler with timestamp in filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"arag-{timestamp}.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Suppress excessive logging from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized with level {log_level}")
    logging.info(f"Log file: {log_file}")


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds session context to log messages.
    
    Attributes:
        logger: Logger to adapt
        session_id: Session ID for context
    """
    
    def __init__(self, logger, session_id):
        """
        Initialize the logger adapter.
        
        Args:
            logger: Logger to adapt
            session_id: Session ID for context
        """
        super().__init__(logger, {"session_id": session_id})
    
    def process(self, msg, kwargs):
        """
        Process log messages to add session context.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments for logging
            
        Returns:
            Tuple of (modified message, kwargs)
        """
        return f"[Session: {self.extra['session_id']}] {msg}", kwargs


def get_session_logger(session_id: str) -> LoggerAdapter:
    """
    Get a logger with session context.
    
    Args:
        session_id: Session ID for context
        
    Returns:
        Logger adapter with session context
    """
    logger = logging.getLogger("arag")
    return LoggerAdapter(logger, session_id)