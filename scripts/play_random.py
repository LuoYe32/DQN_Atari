import argparse
import time

from src.envs import AtariEnvConfig, make_atari_env
from src.envs.atari_wrappers import AtariWrapperConfig
from src.utils.seeding import set_global_seed


def main():
    """
    Run a simple smoke test for the Atari environment pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Render human mode")
    args = parser.parse_args()

    set_global_seed(args.seed)

    wrappers_cfg = AtariWrapperConfig(
        noop_max=30,
        frame_skip=4,
        episodic_life=True,
        reward_clipping=True,
        frame_stack=4,
        fire_reset=True,
        width=84,
        height=84,
    )

    env_cfg = AtariEnvConfig(
        env_id=args.env_id,
        render_mode="human" if args.render else None,
        wrappers=wrappers_cfg,
        frameskip=1,
        repeat_action_probability=0.0,
    )

    env = make_atari_env(env_cfg)
    obs, info = env.reset(seed=args.seed)

    print("Smoke test")
    print("Observation shape:", obs.shape, "dtype:", obs.dtype)
    print("Obs space:", env.observation_space)
    print("Action space:", env.action_space)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        steps = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)
            steps += 1

            if steps == 1:
                assert obs.shape == (4, 84, 84), f"Unexpected obs shape: {obs.shape}"

            if args.render:
                time.sleep(0.01)

        print(f"Episode {ep}: reward={ep_reward:.2f}, steps={steps}")

    env.close()
    print("Smoke test OK")


if __name__ == "__main__":
    main()
