from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TargetUpdateConfig:
    """
    Target network update settings.

    Args:
        mode: "hard" or "soft"
        tau: soft update coefficient (used if mode == "soft"), typical 0.005
        period: hard update period in gradient steps (used if mode == "hard"), typical 10_000
    """
    mode: str = "hard"
    tau: float = 0.005
    period: int = 10_000


@torch.no_grad()
def hard_update(target: nn.Module, online: nn.Module) -> None:
    """Copy all parameters from online network to target network."""
    target.load_state_dict(online.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """
    Polyak averaging update:
        target = (1 - tau) * target + tau * online
    """
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


class TargetUpdater:
    """Small helper to apply hard/soft target updates during training."""

    def __init__(self, cfg: TargetUpdateConfig):
        self.cfg = cfg
        self.step_count = 0

    def maybe_update(self, target: nn.Module, online: nn.Module) -> bool:
        """
        Apply update depending on config. Returns True if an update happened.
        """
        self.step_count += 1

        if self.cfg.mode == "soft":
            soft_update(target, online, tau=self.cfg.tau)
            return True

        if self.cfg.mode == "hard":
            if self.step_count % int(self.cfg.period) == 0:
                hard_update(target, online)
                return True
            return False

        raise ValueError(f"Unknown target update mode: {self.cfg.mode}")
