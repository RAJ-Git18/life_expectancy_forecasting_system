import logging
import os
from datetime import datetime


def get_logger(
    name: str = "sales_forecaster", log_level: int = logging.INFO
) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger (usually __name__).
        log_level (int): The logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding multiple handlers if get_logger is called multiple times
    if logger.hasHandlers():
        return logger

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Define log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Console Handler (StreamHandler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Rotating logs optionally, here simple FileHandler)
    timestamp = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"app_{timestamp}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Create a default logger instance for direct import
logger = get_logger()
