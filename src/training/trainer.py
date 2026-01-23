from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from src.agents.dqn_agent import DQNAgent
from src.agents.replay_buffer import ReplayBuffer


@dataclass
class TrainerConfig:
    """
    DQN training loop settings.
    """
    total_steps: int = 2_000_000

    learning_starts: int = 50_000
    train_frequency: int = 4
    buffer_fill_steps: int = 10_000

    eval_every: int = 200_000
    log_every: int = 10_000


class DQNTrainer:
    """
    Trainer that runs:
      env_step -> store transition -> (if ready) train_step(s)
    """

    def __init__(
        self,
        env,
        agent: DQNAgent,
        replay_buffer: ReplayBuffer,
        cfg: TrainerConfig,
    ):
        self.env = env
        self.agent = agent
        self.rb = replay_buffer
        self.cfg = cfg

    def train(self) -> None:
        obs, info = self.env.reset()

        episode_reward = 0.0
        episode_len = 0
        episode_idx = 0

        for step in range(1, self.cfg.total_steps + 1):
            action = self.agent.act(obs, global_step=step, explore=True)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            self.rb.add(obs, action, float(reward), next_obs, done)

            obs = next_obs
            episode_reward += float(reward)
            episode_len += 1

            if done:
                episode_idx += 1
                print(
                    f"[ep {episode_idx}] step={step} "
                    f"reward={episode_reward:.2f} len={episode_len}"
                )
                obs, info = self.env.reset()
                episode_reward = 0.0
                episode_len = 0

            ready_to_train = (
                step >= self.cfg.learning_starts
                and len(self.rb) >= max(self.cfg.buffer_fill_steps, self.agent.cfg.batch_size)
                and (step % self.cfg.train_frequency == 0)
            )

            if ready_to_train:
                batch = self.rb.sample(self.agent.cfg.batch_size)
                metrics = self.agent.train_step(batch)

                if step % self.cfg.log_every == 0:
                    eps = self.agent.cfg.epsilon.value(step)
                    print(
                        f"[train] step={step} "
                        f"eps={eps:.3f} "
                        f"loss={metrics['loss']:.4f} "
                        f"q_mean={metrics['q_mean']:.3f} "
                        f"q_max={metrics['q_max']:.3f}"
                    )
