"""
Reproducibility utilities for deterministic experiment execution.
"""

import os
import random
from typing import Optional


def seed_everything(seed: int = 42) -> int:
    """
    Set deterministic seeds across Python standard library, NumPy, and PyTorch (if available).
    
    Args:
        seed: Integer seed value (default: 42).
        
    Returns:
        The applied seed integer.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    
    # NumPy seeding
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
        
    # PyTorch seeding (if installed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
        
    return seed
