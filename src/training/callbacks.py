import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CheckpointConfig:
    dirpath: str
    save_every_steps: int = 50_000
    keep_last: int = 3


class CheckpointManager:
    """Save periodic checkpoints and keep only last N files."""

    def __init__(self, cfg: CheckpointConfig):
        self.cfg = cfg
        os.makedirs(cfg.dirpath, exist_ok=True)

    def maybe_save(self, step: int, agent) -> Optional[str]:
        if step % int(self.cfg.save_every_steps) != 0:
            return None

        path = os.path.join(self.cfg.dirpath, f"ckpt_step_{step}.pt")
        agent.save(path)

        ckpts = sorted([f for f in os.listdir(self.cfg.dirpath) if f.startswith("ckpt_step_")])
        if len(ckpts) > self.cfg.keep_last:
            for f in ckpts[: len(ckpts) - self.cfg.keep_last]:
                try:
                    os.remove(os.path.join(self.cfg.dirpath, f))
                except OSError:
                    pass
        return path


class EpisodeCSVLogger:
    """Append episode stats to CSV for reward-vs-steps plots."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("step,episode,reward,length\n")

    def log(self, step: int, episode: int, reward: float, length: int) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{step},{episode},{reward:.6f},{length}\n")
