import time


class FPSMeter:
    """Utility class for tracking frames per second."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.t0 = time.time()
        self.frames = 0

    def step(self, n: int = 1) -> None:
        self.frames += int(n)

    def fps(self) -> float:
        dt = time.time() - self.t0
        if dt <= 1e-9:
            return 0.0
        return float(self.frames / dt)
