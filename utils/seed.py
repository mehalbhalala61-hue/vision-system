# =============================================================================
# utils/seed.py — Fixed Seed Utility
# =============================================================================
import os
import random
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for full reproducibility.
    Call once at the top of train.py before anything else.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic ops — slight speed cost, full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    logger.info(f"Seeds fixed: {seed}")
