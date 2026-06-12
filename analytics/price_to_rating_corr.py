import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_paths import PRICE_TO_RATING_CORR_PLOT_PATH, STEAM_GAMES_DATASET_CSV_PATH

DATASET_PATH = STEAM_GAMES_DATASET_CSV_PATH
PRICE_BRACKET_ORDER = [
    "Free",
    "Under $5",
    "$5 – $15",
    "$15 – $30",
    "$30 – $60",
    "$60+",
]


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna().copy()

    df["total_reviews"] = df["total_positive"] + df["total_negative"]
    threshold = df["total_reviews"].quantile(0.75)
    df = df[df["total_reviews"] >= threshold]

    df["positive_share"] = df["total_positive"] / df["total_reviews"]
    df["price_dollars"] = df["price_final"] / 100.0

    return df


def assign_price_bracket(price: float, is_free: bool) -> str:
    if is_free or price == 0:
        return "Free"
    if price < 5:
        return "Under $5"
    if price < 15:
        return "$5 – $15"
    if price < 30:
        return "$15 – $30"
    if price < 60:
        return "$30 – $60"
    return "$60+"


def compute_bracket_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_bracket"] = df.apply(
        lambda r: assign_price_bracket(r["price_dollars"], r["is_free"]), axis=1
    )

    stats = (
        df.groupby("price_bracket")["positive_share"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    stats.columns = ["price_bracket", "count", "mean", "median", "std"]
    stats["ci"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
    return stats


def plot_price_to_rating(df: pd.DataFrame, bracket_stats: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # left: scatter with trend line
    ax = axes[0]
    price = df["price_dollars"]
    rating = df["positive_share"]

    ax.scatter(price, rating, alpha=0.15, s=8, c="#4C72B0", edgecolors="none")

    max_price = price.quantile(0.99)
    mask = price <= max_price
    sort_idx = np.argsort(price[mask])
    x_sorted = price[mask].values[sort_idx]
    y_sorted = rating[mask].values[sort_idx]

    window = max(len(x_sorted) // 50, 20)
    running_mean = pd.Series(y_sorted).rolling(window, center=True).mean()

    ax.plot(x_sorted, running_mean, color="red", linewidth=2, label="Rolling avg")
    ax.axhline(y=rating.mean(), color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Positive review share")
    ax.set_title("Price vs Review Rating")
    ax.legend(fontsize=8)
    ax.set_xlim(-2, max_price + 5)

    # right: bar chart by price bracket
    ax = axes[1]
    brackets = bracket_stats.set_index("price_bracket").reindex(PRICE_BRACKET_ORDER)
    x = range(len(brackets))

    ax.bar(
        x,
        brackets["mean"],
        yerr=brackets["ci"],
        capsize=4,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )

    for i, (_, row) in enumerate(brackets.iterrows()):
        ax.text(i, row["mean"] + 0.012, f"n={row['count']}", ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(brackets.index, fontsize=8)
    ax.set_ylabel("Mean positive review share")
    ax.set_title("Average rating by price bracket")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    PRICE_TO_RATING_CORR_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PRICE_TO_RATING_CORR_PLOT_PATH, dpi=150)
    print(f"Plot saved to {PRICE_TO_RATING_CORR_PLOT_PATH}")


def main() -> None:
    df = load_and_prepare(DATASET_PATH)
    print(f"Loaded {len(df)} games (after filtering)")

    price = df["price_dollars"]
    rating = df["positive_share"]

    pearson_r, pearson_p = pearsonr(price, rating)
    spearman_r, spearman_p = spearmanr(price, rating)

    print(f"\nPearson  r = {pearson_r:.4f}  (p = {pearson_p:.2e})")
    print(f"Spearman ρ = {spearman_r:.4f}  (p = {spearman_p:.2e})")

    bracket_stats = compute_bracket_stats(df)

    print("\n--- Price bracket stats ---")
    print(bracket_stats.to_string(index=False))

    plot_price_to_rating(df, bracket_stats)


if __name__ == "__main__":
    main()
