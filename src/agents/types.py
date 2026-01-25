from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


Obs = np.ndarray


@dataclass(frozen=True)
class Transition:
    """A single environment transition stored in replay buffer."""
    obs: Obs
    action: int
    reward: float
    next_obs: Obs
    terminated: bool
    truncated: bool
    info: Optional[Dict[str, Any]] = None

    @property
    def done(self) -> bool:
        """Convenience flag: True if episode ended (terminated or truncated)."""
        return bool(self.terminated or self.truncated)
