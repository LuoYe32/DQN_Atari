import numpy as np

from src.agents.replay_buffer import ReplayBuffer, ReplayBufferConfig


def main():
    cfg = ReplayBufferConfig(
        capacity=1000,
        obs_shape=(4, 84, 84),
        device="mps",
        # device="cuda",
    )
    rb = ReplayBuffer(cfg)

    for i in range(300):
        obs = np.random.randint(0, 256, size=(4, 84, 84), dtype=np.uint8)
        next_obs = np.random.randint(0, 256, size=(4, 84, 84), dtype=np.uint8)
        action = np.random.randint(0, 6)
        reward = float(np.random.randn())
        done = bool(np.random.rand() < 0.05)

        rb.add(obs, action, reward, next_obs, done)

    batch = rb.sample(batch_size=32)
    print("ReplayBuffer size:", len(rb))
    print("obs:", batch["obs"].shape, batch["obs"].dtype, batch["obs"].min().item(), batch["obs"].max().item())
    print("actions:", batch["actions"].shape, batch["actions"].dtype)
    print("rewards:", batch["rewards"].shape, batch["rewards"].dtype)
    print("dones:", batch["dones"].shape, batch["dones"].dtype)


if __name__ == "__main__":
    main()
