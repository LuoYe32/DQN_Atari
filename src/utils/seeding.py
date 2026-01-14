import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set global random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
