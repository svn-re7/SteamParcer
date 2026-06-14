import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from models.data_preparation import prepare_model_data
from models.diagnostics import (
    compute_vif_table,
    get_high_correlation_pairs,
    get_model_feature_scores,
    plot_feature_correlation_matrix,
    plot_logistic_vif,
    plot_model_feature_scores,
    plot_model_features,
)
from models.reporting import (
    plot_confusion_matrices,
    plot_metrics,
    plot_roc_pr_curves,
    plot_threshold_metrics,
)
from models.training import create_models, get_metrics_table, train_models
from project_paths import (
    STEAM_GAMES_DATASET_CSV_PATH,
    USER_SCORE_FEATURE_CORRELATION_PLOT_PATH,
    USER_SCORE_CLASSIFICATION_CONFUSION_PLOT_PATH,
    USER_SCORE_CLASSIFICATION_METRICS_PLOT_PATH,
    USER_SCORE_LOGISTIC_VIF_PLOT_PATH,
    USER_SCORE_MODEL_FEATURES_PLOT_PATH,
    USER_SCORE_MODEL_IMPORTANCE_PLOT_PATH,
    USER_SCORE_ROC_PR_PLOT_PATH,
    USER_SCORE_THRESHOLD_METRICS_PLOT_PATH,
)


DATASET_PATH = STEAM_GAMES_DATASET_CSV_PATH
METRICS_PLOT_PATH = USER_SCORE_CLASSIFICATION_METRICS_PLOT_PATH
CONFUSION_PLOT_PATH = USER_SCORE_CLASSIFICATION_CONFUSION_PLOT_PATH
CORRELATION_PLOT_PATH = USER_SCORE_FEATURE_CORRELATION_PLOT_PATH
LOGISTIC_VIF_PLOT_PATH = USER_SCORE_LOGISTIC_VIF_PLOT_PATH
MODEL_FEATURES_PLOT_PATH = USER_SCORE_MODEL_FEATURES_PLOT_PATH
MODEL_IMPORTANCE_PLOT_PATH = USER_SCORE_MODEL_IMPORTANCE_PLOT_PATH
ROC_PR_PLOT_PATH = USER_SCORE_ROC_PR_PLOT_PATH
THRESHOLD_METRICS_PLOT_PATH = USER_SCORE_THRESHOLD_METRICS_PLOT_PATH

TOP_N_GENRES = 20
TOP_N_CATEGORIES = 30
POSITIVE_THRESHOLD = 0.65
CURRENT_YEAR = 2026
TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_LABELS = ["not_positive", "positive"]
HIGH_CORRELATION_THRESHOLD = 0.8
MULTICOLLINEAR_FEATURES_TO_DROP = [
    "game_age_years",
    "genre_Audio Production",
    "category_Family Sharing",
    "genre_Free To Play",
    "category_Shared/Split Screen",
    "category_Tracked Controller Support",
    "category_Online PvP",
]

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

    # диагностика факторов до удаления
    vif_df = compute_vif_table(X)
    high_correlations_df = get_high_correlation_pairs(
        X=X,
        threshold=HIGH_CORRELATION_THRESHOLD,
    )
    plot_feature_correlation_matrix(X, CORRELATION_PLOT_PATH)
    plot_logistic_vif(vif_df, LOGISTIC_VIF_PLOT_PATH)

    # удаление факторов с сильной мультиколлинеарностью
    dropped_multicollinear_features = [
        feature for feature in MULTICOLLINEAR_FEATURES_TO_DROP if feature in X.columns
    ]
    X = X.drop(columns=dropped_multicollinear_features)
    print(f"Dropped multicollinear features: {dropped_multicollinear_features}")

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
    plot_roc_pr_curves(
        results=results,
        y_test=y_test,
        path=ROC_PR_PLOT_PATH,
    )
    plot_threshold_metrics(
        results=results,
        y_test=y_test,
        positive_threshold=POSITIVE_THRESHOLD,
        path=THRESHOLD_METRICS_PLOT_PATH,
    )

    # диагностика финальных факторов модели
    feature_scores = get_model_feature_scores(
        results=results,
        feature_names=list(X.columns),
    )

    plot_model_features(list(X.columns), MODEL_FEATURES_PLOT_PATH)
    plot_model_feature_scores(feature_scores, MODEL_IMPORTANCE_PLOT_PATH)

    print(metrics_df.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nHigh correlation pairs before drop:")
    if high_correlations_df.empty:
        print("No pairs found")
    else:
        print(
            high_correlations_df.head(20).to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )
    print("\nTop LogisticRegression VIF before drop:")
    print(
        vif_df.head(20).to_string(
            index=False,
            float_format=lambda value: "inf" if value == float("inf") else f"{value:.2f}",
        )
    )
    print(f"Metrics plot saved to {METRICS_PLOT_PATH}")
    print(f"Confusion plot saved to {CONFUSION_PLOT_PATH}")
    print(f"ROC/PR plot saved to {ROC_PR_PLOT_PATH}")
    print(f"Threshold metrics plot saved to {THRESHOLD_METRICS_PLOT_PATH}")
    print(f"Correlation plot saved to {CORRELATION_PLOT_PATH}")
    print(f"Logistic VIF plot saved to {LOGISTIC_VIF_PLOT_PATH}")
    print(f"Model features plot saved to {MODEL_FEATURES_PLOT_PATH}")
    print(f"Model importance plot saved to {MODEL_IMPORTANCE_PLOT_PATH}")


if __name__ == "__main__":
    main()
