from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class ReplayBufferConfig:
    """
    Replay buffer configuration.

    Args:
        capacity: Max number of transitions stored.
        obs_shape: Observation shape (e.g. (4, 84, 84)).
        device: Torch device used when sampling batches.
    """
    capacity: int
    obs_shape: Tuple[int, ...] = (4, 84, 84)
    device: str = "mps"
    # device: str = "cuda"


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.

    Stores transitions: (obs, action, reward, next_obs, done).
    Designed for Atari DQN preprocessing output:
      - obs: uint8, shape (4, 84, 84)
      - next_obs: uint8, shape (4, 84, 84)

    Sampling returns torch tensors ready for training.
    """

    def __init__(self, cfg: ReplayBufferConfig):
        self.cfg = cfg
        self.capacity = int(cfg.capacity)
        self.obs_shape = tuple(cfg.obs_shape)
        self.device = torch.device(cfg.device)

        self.pos = 0
        self.size = 0

        self.obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)

        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)

    def __len__(self) -> int:
        """Number of currently stored transitions."""
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add one transition to replay buffer.

        Args:
            obs: Observation at time t, uint8 (4,84,84).
            action: Action taken at time t.
            reward: Reward received after action.
            next_obs: Observation at time t+1, uint8 (4,84,84).
            done: Episode ended flag (terminated or truncated).
        """
        assert obs.shape == self.obs_shape, f"obs shape mismatch: {obs.shape} != {self.obs_shape}"
        assert next_obs.shape == self.obs_shape, f"next_obs shape mismatch: {next_obs.shape} != {self.obs_shape}"
        assert obs.dtype == np.uint8, f"obs dtype must be uint8, got {obs.dtype}"
        assert next_obs.dtype == np.uint8, f"next_obs dtype must be uint8, got {next_obs.dtype}"

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = bool(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    @torch.no_grad()
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a random minibatch.

        Returns dict of torch tensors:
            obs: float32 in [0,1], shape (B, 4, 84, 84)
            actions: int64 (B,)
            rewards: float32 (B,)
            next_obs: float32 in [0,1], shape (B, 4, 84, 84)
            dones: float32 (B,) where done=1.0 if terminal else 0.0
        """
        assert self.size >= batch_size, f"Not enough samples: size={self.size}, batch={batch_size}"

        idxs = np.random.randint(0, self.size, size=batch_size)

        obs = torch.from_numpy(self.obs[idxs]).to(self.device).float() / 255.0
        next_obs = torch.from_numpy(self.next_obs[idxs]).to(self.device).float() / 255.0

        actions = torch.from_numpy(self.actions[idxs]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idxs]).to(self.device)
        dones = torch.from_numpy(self.dones[idxs].astype(np.float32)).to(self.device)

        return {
            "obs": obs,                 # (B,4,84,84)
            "actions": actions,         # (B,)
            "rewards": rewards,         # (B,)
            "next_obs": next_obs,       # (B,4,84,84)
            "dones": dones,             # (B,)
        }
