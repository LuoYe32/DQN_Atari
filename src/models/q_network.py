from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class QNetworkConfig:
    """Config for Atari DQN CNN Q-network."""
    in_channels: int = 4
    num_actions: int = 6
    hidden_dim: int = 512


class QNetwork(nn.Module):
    """
    Classic DQN CNN for Atari (Nature DQN).
    Architecture:
        Conv(32, 8x8, s4) -> Conv(64, 4x4, s2) -> Conv(64, 3x3, s1)
        -> FC(512) -> FC(num_actions)

    Expects input x as float tensor in [0, 1], shape (B, 4, 84, 84).
    """

    def __init__(self, cfg: QNetworkConfig):
        super().__init__()
        self.cfg = cfg

        self.features = nn.Sequential(
            nn.Conv2d(cfg.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.num_actions),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values. Returns shape (B, num_actions)."""
        x = self.features(x)
        q = self.head(x)
        return q

    def _init_weights(self) -> None:
        """Kaiming init for conv/linear layers (common default)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
