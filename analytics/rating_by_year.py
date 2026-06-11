import json

import matplotlib.pyplot as plt
import pandas as pd

DATASET_PATH = "data/processed/steam_games_dataset.csv"
OUTPUT_DIR = "analytics/graphs"


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().copy()


def parse_genres(genres_str: str) -> list[str]:
    try:
        return json.loads(genres_str)
    except (json.JSONDecodeError, TypeError):
        return []


def compute_top_genres_by_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["genres_list"] = df["genres"].apply(parse_genres)
    df = df.explode("genres_list")
    df = df.dropna(subset=["genres_list"])

    df["year"] = pd.to_datetime(df["release_date"]).dt.year

    df = df[(df["year"] >= 2010) & (df["year"] <= 2025)]

    grouped = (
        df.groupby(["year", "genres_list"])
        .agg(
            total_positive=("total_positive", "sum"),
            total_negative=("total_negative", "sum"),
        )
        .reset_index()
    )

    grouped["total_reviews"] = grouped["total_positive"] + grouped["total_negative"]
    grouped = grouped[grouped["total_reviews"] > 0].copy()
    grouped["positive_share"] = grouped["total_positive"] / grouped["total_reviews"]

    top5 = (
        grouped.sort_values(["year", "positive_share"], ascending=[True, False])
        .groupby("year")
        .head(5)
        .reset_index(drop=True)
    )

    return top5


def plot_top_genres(top5: pd.DataFrame) -> None:
    years = sorted(top5["year"].unique())
    n_years = len(years)
    mid = (n_years + 1) // 2
    year_groups = [years[:mid], years[mid:]]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=True)

    bar_width = 0.17
    colors = plt.cm.Set2.colors

    for row_idx, year_group in enumerate(year_groups):
        ax = axes[row_idx]
        x = range(len(year_group))

        for rank in range(5):
            offset = (rank - 2) * bar_width
            bars = []
            labels = []

            for year in year_group:
                year_rows = top5[top5["year"] == year]
                if rank < len(year_rows):
                    row = year_rows.iloc[rank]
                    bars.append(row["positive_share"])
                    labels.append(row["genres_list"])
                else:
                    bars.append(0)
                    labels.append("")

            ax.bar(
                [xi + offset for xi in x],
                bars,
                bar_width,
                label=f"Rank {rank + 1}",
                color=colors[rank % len(colors)],
            )

            for xi, (bar_val, label) in enumerate(zip(bars, labels)):
                if bar_val > 0:
                    ax.text(
                        xi + offset,
                        bar_val + 0.005,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=45,
                    )

        ax.set_xticks(list(x))
        ax.set_xticklabels([str(y) for y in year_group], fontsize=9)
        ax.set_ylabel("Positive review share", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=8, ncol=5)
    fig.suptitle("Top 5 most loved genres by year", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/top_genres_by_year.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


def main() -> None:
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df)} rows")

    df = drop_null_rows(df)
    print(f"After dropping nulls: {len(df)} rows")

    top5 = compute_top_genres_by_year(df)
    print(f"Top genres computed: {len(top5)} rows")
    print(top5.to_string(index=False))

    plot_top_genres(top5)


if __name__ == "__main__":
    main()
