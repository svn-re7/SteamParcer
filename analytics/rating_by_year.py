import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from project_paths import STEAM_GAMES_DATASET_CSV_PATH, TOP_GENRES_BY_YEAR_PLOT_PATH

DATASET_PATH = STEAM_GAMES_DATASET_CSV_PATH
MIN_TOTAL_REVIEWS = 500


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().copy()


def parse_genres(genres_str: str) -> list[str]:
    try:
        return json.loads(genres_str)
    except (json.JSONDecodeError, TypeError):
        return []


def compute_top_genres_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["genres_list"] = df["genres"].apply(parse_genres)
    df = df.explode("genres_list")
    df = df.dropna(subset=["genres_list"])

    df["year"] = pd.to_datetime(df["release_date"]).dt.year

    df = df[(df["year"] >= 2010) & (df["year"] <= 2025)]

    df["total_reviews_game"] = df["total_positive"] + df["total_negative"]
    review_threshold = df["total_reviews_game"].quantile(0.75)
    df = df[df["total_reviews_game"] >= review_threshold].copy()
    print(f"Reviews threshold (75th percentile): {review_threshold:.0f}")

    year_game_counts = df.groupby("year")["appid"].nunique().to_dict()

    grouped = (
        df.groupby(["year", "genres_list"])
        .agg(
            total_positive=("total_positive", "sum"),
            total_negative=("total_negative", "sum"),
        )
        .reset_index()
    )

    grouped["total_reviews"] = grouped["total_positive"] + grouped["total_negative"]
    grouped = grouped[grouped["total_reviews"] >= MIN_TOTAL_REVIEWS].copy()
    grouped["positive_share"] = grouped["total_positive"] / grouped["total_reviews"]

    top5 = (
        grouped.sort_values(["year", "positive_share"], ascending=[True, False])
        .groupby("year")
        .head(5)
        .reset_index(drop=True)
    )

    return top5, year_game_counts


def plot_top_genres(top5: pd.DataFrame, year_game_counts: dict) -> None:
    years = sorted(top5["year"].unique())
    colors = plt.cm.Set2.colors

    fig, ax = plt.subplots(figsize=(10, 12))

    y_pos = []
    genre_labels = []
    values = []
    bar_colors = []
    year_ticks = []
    year_labels = []

    current_y = 0
    for year in years:
        year_rows = (
            top5[top5["year"] == year]
            .sort_values("positive_share", ascending=False)
            .head(5)
        )

        start_y = current_y
        for rank, (_, row) in enumerate(year_rows.iterrows()):
            y_pos.append(current_y)
            genre_labels.append(row["genres_list"])
            values.append(row["positive_share"])
            bar_colors.append(colors[rank % len(colors)])
            current_y += 1

        year_ticks.append((start_y + current_y - 1) / 2)
        year_labels.append(f"{year} ({year_game_counts.get(year, 0)})")
        current_y += 1  # gap between year groups

    ax.barh(y_pos, values, height=0.7, color=bar_colors, zorder=2)
    ax.set_yticks(year_ticks)
    ax.set_yticklabels(year_labels, fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_xlabel("Positive review share", fontsize=9)

    for y, val, label in zip(y_pos, values, genre_labels):
        ax.text(
            val + 0.012,
            y,
            f"{label}  {val:.1%}",
            va="center",
            fontsize=6.5,
        )

    n_items_per_year = 6
    for i in range(1, len(years)):
        sep = i * n_items_per_year - 1
        ax.axhline(y=sep, color="gray", linewidth=0.8, alpha=0.6)

    for i in range(len(years)):
        start = i * n_items_per_year
        end = start + 4
        if i % 2 == 1:
            ax.axhspan(
                start - 0.5,
                end + 0.5,
                color="lightgray",
                alpha=0.15,
                zorder=0,
            )

    ax.set_title("Top 5 most loved genres by year (2010 - 2025)", fontsize=13)
    plt.tight_layout()
    TOP_GENRES_BY_YEAR_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TOP_GENRES_BY_YEAR_PLOT_PATH, dpi=150)
    print(f"Plot saved to {TOP_GENRES_BY_YEAR_PLOT_PATH}")


def main() -> None:
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df)} rows")

    df = drop_null_rows(df)
    print(f"After dropping nulls: {len(df)} rows")

    top5, year_game_counts = compute_top_genres_by_year(df)
    print(f"Top genres computed: {len(top5)} rows")
    print(top5.to_string(index=False))

    plot_top_genres(top5, year_game_counts)


if __name__ == "__main__":
    main()
