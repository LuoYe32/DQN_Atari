from dataclasses import dataclass
from typing import Optional

from src.agents.dqn_agent import DQNAgent
from src.agents.replay_buffer import ReplayBuffer
from src.utils.logger import TBLogger
from src.utils.timers import FPSMeter


@dataclass
class TrainerConfig:
    total_steps: int = 2_000_000

    learning_starts: int = 50_000
    train_frequency: int = 4
    buffer_fill_steps: int = 10_000

    log_every: int = 5_000
    episode_log_every: int = 1


class DQNTrainer:
    """
    DQN training loop.

    env_step -> store transition -> (if ready) sample batch -> gradient step
    """

    def __init__(
        self,
        env,
        agent: DQNAgent,
        replay_buffer: ReplayBuffer,
        cfg: TrainerConfig,
        logger: Optional[TBLogger] = None,
    ):
        self.env = env
        self.agent = agent
        self.rb = replay_buffer
        self.cfg = cfg
        self.logger = logger

        self.fps_meter = FPSMeter()

    def train(self) -> None:
        obs, info = self.env.reset()

        episode_reward = 0.0
        episode_len = 0
        episode_idx = 0

        self.fps_meter.reset()

        for step in range(1, self.cfg.total_steps + 1):
            self.fps_meter.step(1)

            action = self.agent.act(obs, global_step=step, explore=True)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            self.rb.add(obs, action, float(reward), next_obs, done)

            obs = next_obs
            episode_reward += float(reward)
            episode_len += 1

            if done:
                episode_idx += 1

                if self.logger is not None and (episode_idx % self.cfg.episode_log_every == 0):
                    self.logger.log_scalar("episode/reward", episode_reward, step)
                    self.logger.log_scalar("episode/length", episode_len, step)

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

                if self.logger is not None and (step % self.cfg.log_every == 0):
                    eps = self.agent.cfg.epsilon.value(step)
                    fps = self.fps_meter.fps()

                    self.logger.log_scalars(
                        {
                            "train/loss": metrics["loss"],
                            "train/epsilon": eps,
                            "train/q_mean": metrics["q_mean"],
                            "train/q_max": metrics["q_max"],
                            "train/fps": fps,
                        },
                        step,
                    )

                    print(
                        f"[train] step={step} eps={eps:.3f} "
                        f"loss={metrics['loss']:.4f} "
                        f"q_mean={metrics['q_mean']:.3f} q_max={metrics['q_max']:.3f} "
                        f"fps={fps:.1f}"
                    )
