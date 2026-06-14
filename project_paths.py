from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANALYTICS_DIR = PROJECT_DIR / "analytics"
ANALYTICS_GRAPHS_DIR = ANALYTICS_DIR / "graphs"

STEAM_APP_LIST_PATH = RAW_DATA_DIR / "steam_app_list.json"
STEAM_APP_DETAILS_RAW_PATH = RAW_DATA_DIR / "steam_app_details.jsonl"
STEAM_APP_REVIEWS_RAW_PATH = RAW_DATA_DIR / "steam_app_reviews.jsonl"
STEAM_APP_DETAILS_CSV_PATH = PROCESSED_DATA_DIR / "steam_app_details.csv"
STEAM_APP_REVIEWS_CSV_PATH = PROCESSED_DATA_DIR / "steam_app_reviews.csv"
STEAM_GAMES_DATASET_CSV_PATH = PROCESSED_DATA_DIR / "steam_games_dataset.csv"
TOP_GENRES_BY_YEAR_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "top_genres_by_year.png"
TOP_GENRES_BY_POPULARITY_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "top_genres_by_popularity.png"
PRICE_TO_RATING_CORR_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "price_to_rating_corr.png"
BAYESIAN_TOP_BY_YEAR_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "bayesian_top_by_year.png"
BAYESIAN_TOP_ALL_TIME_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "bayesian_top_all_time.png"
USER_SCORE_CLASSIFICATION_METRICS_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "user_score_classification_metrics.png"
USER_SCORE_CLASSIFICATION_CONFUSION_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "user_score_classification_confusion_matrix.png"
USER_SCORE_FEATURE_CORRELATION_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "user_score_feature_correlation.png"
USER_SCORE_LOGISTIC_VIF_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "user_score_logistic_vif.png"
USER_SCORE_MODEL_FEATURES_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "user_score_model_features.png"
USER_SCORE_MODEL_IMPORTANCE_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "user_score_model_feature_importance.png"
USER_SCORE_ROC_PR_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "user_score_roc_pr_curves.png"
USER_SCORE_THRESHOLD_METRICS_PLOT_PATH = ANALYTICS_GRAPHS_DIR / "user_score_threshold_metrics.png"
