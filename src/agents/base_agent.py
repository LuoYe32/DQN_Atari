from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseAgent(ABC):
    """
    Minimal agent interface for Atari DQN-style training/evaluation.

    The agent consumes preprocessed observations from the environment:
      - obs shape: (4, 84, 84)
      - dtype: uint8
      - value range: [0, 255]
    and outputs an integer action from a discrete action space.
    """

    @abstractmethod
    def act(self, obs: np.ndarray, *, explore: bool = True) -> int:
        """
        Select an action given the current observation.

        Args:
            obs: Current observation, shape (4, 84, 84), dtype uint8.
            explore: If True, allow exploration (e.g., epsilon-greedy).

        Returns:
            Discrete action index.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset internal episodic state (optional).

        For DQN this is usually a no-op, but it is useful for agents
        with recurrent state or episode-dependent logic.
        """
        return

    def save(self, path: str) -> None:
        """Save agent parameters/checkpoint (optional)."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load agent parameters/checkpoint (optional)."""
        raise NotImplementedError
