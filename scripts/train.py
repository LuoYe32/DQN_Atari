import argparse
import os
import time

from src.envs import AtariEnvConfig, make_atari_env
from src.agents.dqn_agent import DQNAgent, DQNConfig
from src.agents.replay_buffer import ReplayBuffer, ReplayBufferConfig
from src.training.trainer import DQNTrainer, TrainerConfig
from src.utils.logger import TBLogger, LoggerConfig
from src.training.callbacks import CheckpointManager, CheckpointConfig, EpisodeCSVLogger
from src.utils.seeding import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.run_name is None:
        args.run_name = f"{args.env_id.replace('/', '_')}_{int(time.time())}"

    game_name = args.env_id.split("/")[1].split("-")[0].lower()

    run_dir = os.path.join(
        "results",
        game_name,
        f"seed_{args.seed}"
    )
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tb")
    csv_path = os.path.join(run_dir, "episodes.csv")

    env = make_atari_env(AtariEnvConfig(env_id=args.env_id))
    num_actions = env.action_space.n

    agent = DQNAgent(num_actions=num_actions, cfg=DQNConfig(), device=args.device)

    rb = ReplayBuffer(
        ReplayBufferConfig(
            capacity=1_000_000,
            frame_shape=(84, 84),
            stack_size=4,
            device=args.device
        )
    )

    logger = TBLogger(LoggerConfig(log_dir=tb_dir, run_name=None))
    ckpt_mgr = CheckpointManager(
        CheckpointConfig(
            dirpath=ckpt_dir,
            save_every_steps=100_000,
            keep_last=25,
        )
    )
    episode_csv = EpisodeCSVLogger(csv_path)

    trainer = DQNTrainer(
        env=env,
        agent=agent,
        replay_buffer=rb,
        cfg=TrainerConfig(total_steps=args.total_steps),
        logger=logger,
        checkpoint_mgr=ckpt_mgr,
        episode_csv=episode_csv,
    )

    try:
        trainer.train()
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
