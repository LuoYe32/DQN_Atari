import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def load_and_align(csv_paths, smooth):
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df = df.sort_values("step")
        df["reward_smooth"] = df["reward"].rolling(smooth, min_periods=1).mean()
        dfs.append(df[["step", "reward_smooth"]].rename(columns={"reward_smooth": os.path.basename(path)}))

    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = pd.merge_asof(df_merged, df, on="step")

    df_merged = df_merged.set_index("step")
    mean = df_merged.mean(axis=1)
    std = df_merged.std(axis=1)
    return mean, std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True,
                        help="Glob pattern to episodes.csv for different seeds")
    parser.add_argument("--out", type=str, default="multi_seed_curve.png")
    parser.add_argument("--smooth", type=int, default=20)
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(args.pattern))
    mean, std = load_and_align(csv_paths, args.smooth)

    plt.figure()
    plt.plot(mean.index, mean.values, label="mean reward")
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.3, label="± std")
    plt.xlabel("Environment steps")
    plt.ylabel("Episode reward")
    plt.title("Learning curve (mean ± std over seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
