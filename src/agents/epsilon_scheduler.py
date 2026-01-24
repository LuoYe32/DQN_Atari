from dataclasses import dataclass


@dataclass
class EpsilonSchedule:
    """
    Linear epsilon decay schedule for epsilon-greedy exploration.
    """
    eps_start: float = 1.0
    eps_end: float = 0.1
    decay_steps: int = 1_000_000

    def value(self, step: int) -> float:
        """
        Return epsilon at a given global environment step.
        """
        step = max(0, int(step))
        if step >= self.decay_steps:
            return float(self.eps_end)
        frac = step / float(self.decay_steps)
        return float(self.eps_start + frac * (self.eps_end - self.eps_start))
