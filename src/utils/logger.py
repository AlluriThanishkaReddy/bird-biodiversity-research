"""
Structured logging utility for research workflows.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str = "bird_research", level: str = "INFO") -> logging.Logger:
    """
    Configure and return a structured logger.
    
    Args:
        name: Name of the logger.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
