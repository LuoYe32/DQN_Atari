import subprocess
import sys

GAMES = {
    "ALE/Pong-v5": 3_000_000,
    # "ALE/Breakout-v5": 5_000_000,
    # "ALE/SpaceInvaders-v5": 5_000_000,
}

LR = {
    "ALE/Pong-v5": 2.5e-4,
    # "ALE/Breakout-v5": 5e-5,
    # "ALE/SpaceInvaders-v5": 5e-5,
}

SEEDS = [0, 1]
# DEVICE = "mps"
DEVICE = "cuda"

for game, steps in GAMES.items():
    for seed in SEEDS:
        cmd = [
            sys.executable, "-m", "scripts.train",
            "--env_id", game,
            "--total_steps", str(steps),
            "--seed", str(seed),
            "--device", DEVICE,
            "--lr", str(LR[game]),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)
