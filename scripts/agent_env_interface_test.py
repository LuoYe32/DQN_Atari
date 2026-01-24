import argparse

from src.envs import AtariEnvConfig, make_atari_env
from src.envs.atari_wrappers import AtariWrapperConfig
from src.utils.seeding import set_global_seed
from src.agents.random_agent import RandomAgent


def main():
    """
    End-to-end interface test: environment + agent loop.

    Verifies that:
      - env returns obs in DQN format (4,84,84) uint8
      - agent can consume obs and return valid discrete actions
      - the step loop runs without errors
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_global_seed(args.seed)

    env_cfg = AtariEnvConfig(
        env_id=args.env_id,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=None,
        wrappers=AtariWrapperConfig(),
    )
    env = make_atari_env(env_cfg)

    agent = RandomAgent(env.action_space)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        agent.reset()

        assert obs.shape == (4, 84, 84), f"Unexpected obs shape: {obs.shape}"
        assert obs.dtype.name == "uint8", f"Unexpected obs dtype: {obs.dtype}"

        ep_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.act(obs, explore=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)

        print(f"[Episode {ep}] reward={ep_reward:.2f}")

    env.close()
    print("Agent-env interface test OK")


if __name__ == "__main__":
    main()
