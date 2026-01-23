import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter


@dataclass
class LoggerConfig:
    """
    TensorBoard logger config.

    Args:
        log_dir: Directory where logs will be stored.
        run_name: Optional run name (subfolder inside log_dir).
    """
    log_dir: str = "results/tensorboard"
    run_name: Optional[str] = None


class TBLogger:
    """
    Simple TensorBoard logger wrapper.

    Usage:
        logger.log_scalar("train/loss", loss, step)
        logger.log_scalars({"a": 1.0, "b": 2.0}, step)
    """

    def __init__(self, cfg: LoggerConfig):
        self.cfg = cfg
        self.start_time = time.time()

        run_dir = cfg.log_dir
        if cfg.run_name is not None:
            run_dir = os.path.join(cfg.log_dir, cfg.run_name)

        os.makedirs(run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=run_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, float(value), int(step))

    def log_scalars(self, scalars: Dict[str, float], step: int) -> None:
        for k, v in scalars.items():
            self.log_scalar(k, v, step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
