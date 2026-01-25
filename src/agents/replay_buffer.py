from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class ReplayBufferConfig:
    """
    Replay Buffer config.

    Args:
        capacity: Number of *frames* stored (not stacked observations).
        frame_shape: Shape of a single frame (H, W).
        stack_size: Number of frames in observation stack (usually 4).
        device: Torch device used when sampling batches.
    """
    capacity: int
    frame_shape: Tuple[int, int] = (84, 84)
    stack_size: int = 4
    device: str = "mps"
    # device: str = "cpu"


class ReplayBuffer:
    """
    Memory-efficient Replay Buffer for Atari DQN.

    Stores:
        - frames (uint8) of shape (H, W)
        - actions
        - rewards
        - done flags

    Stacked observations are reconstructed at sampling time.
    """

    def __init__(self, cfg: ReplayBufferConfig):
        self.cfg = cfg
        self.capacity = int(cfg.capacity)
        self.frame_shape = tuple(cfg.frame_shape)
        self.stack_size = int(cfg.stack_size)
        self.device = torch.device(cfg.device)

        self.pos = 0
        self.size = 0

        self.frames = np.zeros((self.capacity, *self.frame_shape), dtype=np.uint8)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs_stack: np.ndarray,
        action: int,
        reward: float,
        next_obs_stack: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store only the last frame of next_obs_stack to ensure correct reconstruction of next state.
        """
        assert obs_stack.dtype == np.uint8
        assert next_obs_stack.dtype == np.uint8

        last_frame = next_obs_stack[-1]

        self.frames[self.pos] = last_frame
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = bool(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_stack(self, idx: int) -> np.ndarray:
        """Reconstruct stacked observation ending at idx."""
        frames = []
        for i in range(self.stack_size):
            cur_idx = idx - (self.stack_size - 1 - i)

            if cur_idx < 0:
                frames.append(np.zeros(self.frame_shape, dtype=np.uint8))
            else:
                frames.append(self.frames[cur_idx])

            if cur_idx > 0 and self.dones[cur_idx - 1]:
                frames = [np.zeros(self.frame_shape, dtype=np.uint8)] * (self.stack_size - len(frames)) + frames
                break

        return np.stack(frames, axis=0)

    @torch.no_grad()
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        assert self.size > self.stack_size, "Not enough frames to sample"

        idxs = np.random.randint(self.stack_size, self.size - 1, size=batch_size)

        obs_batch = np.stack([self._get_stack(i - 1) for i in idxs])
        next_obs_batch = np.stack([self._get_stack(i) for i in idxs])

        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs].astype(np.float32)

        return {
            "obs": torch.from_numpy(obs_batch).to(self.device).float() / 255.0,
            "actions": torch.from_numpy(actions).to(self.device),
            "rewards": torch.from_numpy(rewards).to(self.device),
            "next_obs": torch.from_numpy(next_obs_batch).to(self.device).float() / 255.0,
            "dones": torch.from_numpy(dones).to(self.device),
        }
