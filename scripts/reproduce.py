import subprocess

GAMES = {
    # "ALE/Pong-v5": 2_500_000,
    "ALE/Breakout-v5": 3_000_000,
    # "ALE/SpaceInvaders-v5": 3_000_000,
}

SEEDS = [0, 1, 2]
DEVICE = "mps"
# DEVICE = "cuda"

for game, steps in GAMES.items():
    for seed in SEEDS:
        cmd = [
            "python", "-m", "scripts.train",
            "--env_id", game,
            "--total_steps", str(steps),
            "--seed", str(seed),
            "--device", DEVICE,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)
