import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from project_paths import (
    BAYESIAN_TOP_ALL_TIME_PLOT_PATH,
    BAYESIAN_TOP_BY_YEAR_PLOT_PATH,
    STEAM_GAMES_DATASET_CSV_PATH,
)


DATASET_PATH = STEAM_GAMES_DATASET_CSV_PATH

# фильтры и размеры топов
REVIEW_QUANTILE = 0.25
MIN_TOTAL_REVIEWS = 500
MIN_YEAR = 2010
MAX_YEAR = 2025
TOP_ALL_TIME_N = 10
TOP_BY_YEAR_N = 3


def load_dataset(path) -> pd.DataFrame:
    # чтение датасета
    return pd.read_csv(path)


def prepare_bayesian_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float]:
    # расчет оценки
    # оставляем только строки, где есть данные по отзывам
    df = df.dropna(subset=["total_positive", "total_negative"]).copy()

    # приводим отзывы к числам на случай строковых значений в csv
    df["total_positive"] = pd.to_numeric(df["total_positive"], errors="coerce")
    df["total_negative"] = pd.to_numeric(df["total_negative"], errors="coerce")
    df = df.dropna(subset=["total_positive", "total_negative"]).copy()

    # считаем общий объем отзывов и убираем слишком маленькие игры
    df["total_reviews"] = df["total_positive"] + df["total_negative"]
    df = df[df["total_reviews"] >= MIN_TOTAL_REVIEWS].copy()

    # после абсолютного порога берем верхние 75% игр по объему отзывов
    min_reviews = df["total_reviews"].quantile(REVIEW_QUANTILE)
    df = df[df["total_reviews"] >= min_reviews].copy()

    # обычная доля положительных отзывов без байесовской поправки
    df["positive_ratio"] = df["total_positive"] / df["total_reviews"]

    # m — средняя доля положительных, c — типичный объем отзывов
    m = df["positive_ratio"].mean()
    c = df["total_reviews"].median()

    # сглаживаем оценку: игры с малым числом отзывов тянутся к среднему m
    df["bayesian_score"] = (
        df["total_positive"] + c * m
    ) / (
        df["total_reviews"] + c
    )

    return df, m, c, min_reviews


def add_release_year(df: pd.DataFrame) -> pd.DataFrame:
    # год релиза
    df = df.copy()
    # некорректные даты превращаются в NaT и потом удаляются
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # ограничиваем период, чтобы графики не смешивали старые и будущие релизы
    df = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)].copy()

    return df


def get_top_by_year(df: pd.DataFrame) -> pd.DataFrame:
    # топ по годам
    df = add_release_year(df)

    # сортируем внутри каждого года и берем лучшие игры
    return (
        df.sort_values(["year", "bayesian_score"], ascending=[True, False])
        .groupby("year")
        .head(TOP_BY_YEAR_N)
        .reset_index(drop=True)
    )


def get_top_all_time(df: pd.DataFrame) -> pd.DataFrame:
    # топ за все время
    df = add_release_year(df)

    # берем лучшие игры за весь выбранный период
    return (
        df.sort_values("bayesian_score", ascending=False)
        .head(TOP_ALL_TIME_N)
        .reset_index(drop=True)
    )


def shorten_name(name: object, max_length: int = 42) -> str:
    # короткое название
    name = str(name)

    if len(name) <= max_length:
        return name

    return f"{name[:max_length - 3]}..."


def parse_genres(genres_str: object) -> list[str]:
    # жанры из json-строки
    try:
        genres = json.loads(genres_str)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(genres, list):
        return []

    return [str(genre) for genre in genres if genre]


def format_game_with_genres(row: pd.Series) -> str:
    # название и первые жанры
    genres = parse_genres(row.get("genres"))
    game_name = shorten_name(row["name"], max_length=34)

    if not genres:
        return game_name

    return f"{game_name} | {', '.join(genres[:3])}"


def format_reviews(value: float) -> str:
    # формат отзывов
    return f"{int(value):,}".replace(",", " ")


def plot_top_all_time(top_games: pd.DataFrame) -> None:
    # график за все время
    plot_df = top_games.sort_values("bayesian_score", ascending=True)

    # matplotlib рисует снизу вверх, поэтому сортировка выше разворачивает порядок
    labels = [format_game_with_genres(row) for _, row in plot_df.iterrows()]
    values = plot_df["bayesian_score"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(labels, values, color="#4c78a8", zorder=2)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_xlabel("Bayesian score", fontsize=9)
    ax.set_title(f"Топ-10 игр по байесовской оценке ({MIN_YEAR} - {MAX_YEAR})", fontsize=13)

    # рядом с баром пишем score и количество отзывов
    for y, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            row["bayesian_score"] + 0.008,
            y,
            f"{row['bayesian_score']:.1%} ({format_reviews(row['total_reviews'])})",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    BAYESIAN_TOP_ALL_TIME_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(BAYESIAN_TOP_ALL_TIME_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {BAYESIAN_TOP_ALL_TIME_PLOT_PATH}")


def plot_top_by_year(top_games: pd.DataFrame) -> None:
    # график по годам
    years = sorted(top_games["year"].unique())
    colors = plt.cm.tab10.colors

    # списки нужны, чтобы собрать позиции баров вручную по группам лет
    y_pos = []
    values = []
    labels = []
    bar_colors = []
    year_ticks = []
    year_labels = []
    spans = []
    separators = []

    current_y = 0

    for year_index, year in enumerate(years):
        year_rows = (
            top_games[top_games["year"] == year]
            .sort_values("bayesian_score", ascending=False)
            .head(TOP_BY_YEAR_N)
        )

        if year_rows.empty:
            continue

        start_y = current_y

        # добавляем бары текущего года
        for rank, (_, row) in enumerate(year_rows.iterrows()):
            y_pos.append(current_y)
            values.append(row["bayesian_score"])
            labels.append(
                f"{shorten_name(row['name'])}  "
                f"{row['bayesian_score']:.1%}  "
                f"({format_reviews(row['total_reviews'])})"
            )
            bar_colors.append(colors[rank % len(colors)])
            current_y += 1

        end_y = current_y - 1
        year_ticks.append((start_y + end_y) / 2)
        year_labels.append(str(year))

        # чередуем легкую заливку, чтобы годы легче читались
        if year_index % 2 == 1:
            spans.append((start_y - 0.5, end_y + 0.5))

        separators.append(end_y + 0.5)
        current_y += 1

    # высота зависит от числа баров, чтобы подписи не налезали друг на друга
    fig_height = max(10, len(y_pos) * 0.32)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.barh(y_pos, values, height=0.72, color=bar_colors, zorder=2)
    ax.set_yticks(year_ticks)
    ax.set_yticklabels(year_labels, fontsize=8, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_xlabel("Bayesian score", fontsize=9)

    for y, val, label in zip(y_pos, values, labels):
        ax.text(
            val + 0.008,
            y,
            label,
            va="center",
            fontsize=6.5,
        )

    for start, end in spans:
        ax.axhspan(start, end, color="lightgray", alpha=0.15, zorder=0)

    for sep in separators[:-1]:
        ax.axhline(y=sep, color="gray", linewidth=0.7, alpha=0.5)

    ax.set_title(f"Топ-3 игры по байесовской оценке в каждом году ({MIN_YEAR} - {MAX_YEAR})", fontsize=13)
    plt.tight_layout()
    BAYESIAN_TOP_BY_YEAR_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(BAYESIAN_TOP_BY_YEAR_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {BAYESIAN_TOP_BY_YEAR_PLOT_PATH}")


def main() -> None:
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df)} rows")

    scored_df, m, c, min_reviews = prepare_bayesian_scores(df)
    print(f"After review filter: {len(scored_df)} rows")
    print(f"Minimum reviews: {MIN_TOTAL_REVIEWS}")
    print(f"Review threshold: {min_reviews:.0f}")
    print(f"m: {m:.4f}")
    print(f"C: {c:.0f}")

    top_all_time = get_top_all_time(scored_df)
    top_by_year = get_top_by_year(scored_df)

    print("Top 10 all-time:")
    print(
        top_all_time[
            ["name", "release_date", "bayesian_score", "total_reviews"]
        ].to_string(index=False)
    )

    plot_top_all_time(top_all_time)
    plot_top_by_year(top_by_year)


if __name__ == "__main__":
    main()
