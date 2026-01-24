import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="learning_curve.png")
    parser.add_argument("--smooth", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["reward_smooth"] = df["reward"].rolling(args.smooth, min_periods=1).mean()

    plt.figure()
    plt.plot(df["step"], df["reward_smooth"])
    plt.xlabel("Environment steps")
    plt.ylabel(f"Episode reward (moving avg, window={args.smooth})")
    plt.title("Pong learning curve")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
