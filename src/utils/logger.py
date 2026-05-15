"""
Centralized logging configuration using loguru.
"""

import sys
from loguru import logger


def get_logger(name: str = "chocolate_analysis") -> logger:
    """
    Returns a configured logger instance.

    Args:
        name: Logger name/module identifier

    Returns:
        Configured loguru logger
    """
    logger.remove()  # Remove default handler

    # Console handler - INFO and above
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # File handler - DEBUG and above
    logger.add(
        f"logs/{name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    return logger.bind(module=name)
