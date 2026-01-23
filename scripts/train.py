import argparse
import time

from src.envs import AtariEnvConfig, make_atari_env
from src.agents.dqn_agent import DQNAgent, DQNConfig
from src.agents.replay_buffer import ReplayBuffer, ReplayBufferConfig
from src.training.trainer import DQNTrainer, TrainerConfig
from src.utils.logger import TBLogger, LoggerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"{args.env_id.replace('/', '_')}_{int(time.time())}"

    env = make_atari_env(AtariEnvConfig(env_id=args.env_id))
    num_actions = env.action_space.n

    agent = DQNAgent(
        num_actions=num_actions,
        cfg=DQNConfig(),
        device=args.device,
    )

    rb = ReplayBuffer(
        ReplayBufferConfig(
            capacity=200_000,
            obs_shape=(4, 84, 84),
            device=args.device,
        )
    )

    logger = TBLogger(LoggerConfig(log_dir="results/tensorboard", run_name=args.run_name))

    trainer = DQNTrainer(
        env=env,
        agent=agent,
        replay_buffer=rb,
        cfg=TrainerConfig(total_steps=args.total_steps),
        logger=logger,
    )

    try:
        trainer.train()
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
