import argparse
import os

import numpy as np

from src.envs.atari_env import AtariEnvConfig, make_atari_env
from src.envs.atari_wrappers import AtariWrapperConfig
from src.agents.dqn_agent import DQNAgent, DQNConfig
from src.utils.video import save_mp4, save_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/eval")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    eval_wrappers = AtariWrapperConfig(
        episodic_life=False,
        reward_clipping=False,
        frame_stack=4,
        frame_skip=4,
        noop_max=30,
        fire_reset=True,
        width=84,
        height=84,
    )

    env = make_atari_env(
        AtariEnvConfig(env_id=args.env_id, render_mode="rgb_array", wrappers=eval_wrappers)
    )
    num_actions = env.action_space.n

    agent = DQNAgent(num_actions=num_actions, cfg=DQNConfig(), device=args.device)
    agent.load(args.ckpt)

    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(args.episodes):
        frames = []
        obs, info = env.reset()
        terminated = truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            action = agent.act(obs, global_step=10**9, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)

        mp4_path = os.path.join(args.out_dir, f"rollout_ep{ep}.mp4")
        gif_path = os.path.join(args.out_dir, f"rollout_ep{ep}.gif")
        save_mp4(frames, mp4_path, fps=args.fps)
        save_gif(frames, gif_path, fps=args.fps)

        print(f"[eval] ep={ep} reward={ep_reward:.2f}")
        print(f"[saved] {mp4_path}")
        print(f"[saved] {gif_path}")

    env.close()


if __name__ == "__main__":
    main()
