from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.q_network import QNetwork, QNetworkConfig
from src.agents.target_update import TargetUpdater, TargetUpdateConfig, hard_update
from src.agents.epsilon_scheduler import EpsilonSchedule


@dataclass
class DQNConfig:
    """
    DQN training hyperparameters.
    """
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 32

    grad_clip_norm: Optional[float] = 10.0

    epsilon: EpsilonSchedule = field(default_factory=EpsilonSchedule)

    target_update: TargetUpdateConfig = field(
        default_factory=lambda: TargetUpdateConfig(
            mode="hard",
            period=10_000
        )
    )


class DQNAgent:
    """
    DQN Agent:
      - online network (trainable)
      - target network (for stable TD targets)
      - epsilon-greedy action selection
      - one gradient update step from replay buffer batch
    """

    def __init__(
        self,
        num_actions: int,
        cfg: DQNConfig,
        device: str = "cuda",
        # device: str = "mps",
    ):
        self.cfg = cfg
        self.num_actions = int(num_actions)
        self.device = torch.device(device)

        self.online = QNetwork(QNetworkConfig(in_channels=4, num_actions=self.num_actions)).to(self.device)
        self.target = QNetwork(QNetworkConfig(in_channels=4, num_actions=self.num_actions)).to(self.device)
        self.target.eval()

        hard_update(self.target, self.online)

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)

        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.target_updater = TargetUpdater(self.cfg.target_update)

    @torch.no_grad()
    def act(self, obs: np.ndarray, global_step: int, explore: bool = True) -> int:
        """
        Choose an action using epsilon-greedy policy.

        obs: uint8 (4,84,84)
        """
        if explore:
            eps = self.cfg.epsilon.value(global_step)
            if np.random.rand() < eps:
                return int(np.random.randint(self.num_actions))

        x = torch.from_numpy(obs).to(self.device).float().unsqueeze(0) / 255.0  # (1,4,84,84)
        q_values = self.online(x)  # (1, A)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one gradient step on DQN loss.

        batch contains:
          obs: (B,4,84,84) float in [0,1]
          actions: (B,)
          rewards: (B,)
          next_obs: (B,4,84,84) float in [0,1]
          dones: (B,) float {0,1}
        """
        self.online.train()

        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        q = self.online(obs)  # (B,A)
        q_sa = q.gather(1, actions.view(-1, 1)).squeeze(1)  # (B,)

        with torch.no_grad():
            q_next = self.target(next_obs)  # (B,A)
            max_q_next = q_next.max(dim=1).values  # (B,)
            td_target = rewards + self.cfg.gamma * (1.0 - dones) * max_q_next

        loss = self.loss_fn(q_sa, td_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=float(self.cfg.grad_clip_norm))

        self.optimizer.step()

        updated = self.target_updater.maybe_update(self.target, self.online)

        return {
            "loss": float(loss.item()),
            "target_updated": float(updated),
            "q_mean": float(q.mean().item()),
            "q_max": float(q.max().item()),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
