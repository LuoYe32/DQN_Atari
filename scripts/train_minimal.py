import argparse

from src.envs import AtariEnvConfig, make_atari_env
from src.agents.dqn_agent import DQNAgent, DQNConfig
from src.agents.replay_buffer import ReplayBuffer, ReplayBufferConfig
from src.training.trainer import DQNTrainer, TrainerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

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

    trainer = DQNTrainer(
        env=env,
        agent=agent,
        replay_buffer=rb,
        cfg=TrainerConfig(total_steps=args.total_steps),
    )

    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
