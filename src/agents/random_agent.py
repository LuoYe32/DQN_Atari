import numpy as np
from gymnasium import spaces

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that samples random actions from a discrete action space."""

    def __init__(self, action_space: spaces.Discrete):
        assert isinstance(action_space, spaces.Discrete)
        self.action_space = action_space

    def act(self, obs: np.ndarray, *, explore: bool = True) -> int:
        return int(self.action_space.sample())

    def save(self, path: str) -> None:
        raise NotImplementedError("RandomAgent has no parameters to save.")

    def load(self, path: str) -> None:
        raise NotImplementedError("RandomAgent has no parameters to load.")
