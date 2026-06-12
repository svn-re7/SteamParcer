import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from models.data_preparation import prepare_model_data
from models.reporting import plot_confusion_matrices, plot_metrics
from models.training import create_models, get_metrics_table, train_models
from project_paths import (
    STEAM_GAMES_DATASET_CSV_PATH,
    USER_SCORE_CLASSIFICATION_CONFUSION_PLOT_PATH,
    USER_SCORE_CLASSIFICATION_METRICS_PLOT_PATH,
)


DATASET_PATH = STEAM_GAMES_DATASET_CSV_PATH
METRICS_PLOT_PATH = USER_SCORE_CLASSIFICATION_METRICS_PLOT_PATH
CONFUSION_PLOT_PATH = USER_SCORE_CLASSIFICATION_CONFUSION_PLOT_PATH

TOP_N_GENRES = 20
TOP_N_CATEGORIES = 30
POSITIVE_THRESHOLD = 0.65
CURRENT_YEAR = 2026
TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_LABELS = ["not_positive", "positive"]

TARGET_MAPPING = {
    "Overwhelmingly Positive": "positive",
    "Very Positive": "positive",
    "Positive": "positive",
    "Mostly Positive": "positive",
    "Mixed": "not_positive",
    "Mostly Negative": "not_positive",
    "Negative": "not_positive",
    "Very Negative": "not_positive",
    "Overwhelmingly Negative": "not_positive",
}


def main() -> None:
    # чтение финального датасета
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} rows")

    # подготовка таргета и факторов модели
    (
        df,
        X,
        y,
        excluded_rows,
    ) = prepare_model_data(
        df=df,
        target_mapping=TARGET_MAPPING,
        top_n_genres=TOP_N_GENRES,
        top_n_categories=TOP_N_CATEGORIES,
        current_year=CURRENT_YEAR,
    )
    print(f"Rows after target mapping: {len(df)}")
    print(f"Excluded rows: {excluded_rows}")

    # одинаковое разбиение для обеих моделей
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # обучение и оценка моделей
    models = create_models(random_state=RANDOM_STATE)
    results = train_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        positive_threshold=POSITIVE_THRESHOLD,
    )
    metrics_df = get_metrics_table(
        results=results,
        y_test=y_test,
        class_labels=CLASS_LABELS,
    )

    # графики качества и ошибок
    plot_metrics(metrics_df, METRICS_PLOT_PATH)
    plot_confusion_matrices(
        results=results,
        y_test=y_test,
        class_labels=CLASS_LABELS,
        positive_threshold=POSITIVE_THRESHOLD,
        path=CONFUSION_PLOT_PATH,
    )

    print(metrics_df.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Metrics plot saved to {METRICS_PLOT_PATH}")
    print(f"Confusion plot saved to {CONFUSION_PLOT_PATH}")


if __name__ == "__main__":
    main()
