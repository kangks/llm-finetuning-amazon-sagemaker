import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logging(log_file: Optional[str] = None, 
                 log_level: int = logging.INFO,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> None:
    """
    Configure logging to output to both console and file (if specified)
    
    Args:
        log_file: Path to log file. If None, only console logging is enabled
        log_level: Logging level (default: INFO)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log initial configuration
    logging.info(f"Logging configured with level: {logging.getLevelName(log_level)}")
    if log_file:
        logging.info(f"Log file location: {log_file}")
