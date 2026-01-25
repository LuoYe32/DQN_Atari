import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_curve(path, smooth):
    df = pd.read_csv(path)
    df = df.sort_values("step")
    df["reward_smooth"] = df["reward"].rolling(smooth, min_periods=1).mean()
    return df["step"], df["reward_smooth"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="List of CSV files to compare")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each curve (same order as csvs)")
    parser.add_argument("--out", type=str, default="lr_comparison.png")
    parser.add_argument("--smooth", type=int, default=20)
    args = parser.parse_args()

    plt.figure()

    for csv, label in zip(args.csvs, args.labels):
        steps, rewards = load_curve(csv, args.smooth)
        plt.plot(steps, rewards, label=label)

    plt.xlabel("Environment steps")
    plt.ylabel("Episode reward")
    plt.title("Learning curves comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
