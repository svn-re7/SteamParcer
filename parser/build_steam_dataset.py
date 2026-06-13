import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from project_paths import (
    STEAM_APP_DETAILS_CSV_PATH,
    STEAM_APP_REVIEWS_CSV_PATH,
    STEAM_GAMES_DATASET_CSV_PATH,
)


def main() -> None:
    # чтение таблиц
    details_df = pd.read_csv(STEAM_APP_DETAILS_CSV_PATH)
    reviews_df = pd.read_csv(STEAM_APP_REVIEWS_CSV_PATH)

    # сбор датасета
    dataset_df = details_df.merge(reviews_df, on="appid", how="left")

    STEAM_GAMES_DATASET_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(STEAM_GAMES_DATASET_CSV_PATH, index=False, encoding="utf-8")

    print(f"строк в csv: {len(dataset_df)}")
    print(f"файл: {STEAM_GAMES_DATASET_CSV_PATH}")


if __name__ == "__main__":
    main()
