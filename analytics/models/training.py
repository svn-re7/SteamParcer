import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_models(random_state: int) -> dict[str, object]:
    # модели для сравнения
    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "RandomForest": random_forest,
        "LogisticRegression": logistic_regression,
    }


def get_estimator(model: object) -> object:
    # модель внутри pipeline
    if isinstance(model, Pipeline):
        return model.named_steps["model"]

    return model


def get_positive_probabilities(model: object, X_test: pd.DataFrame) -> pd.Series:
    # вероятность класса positive
    estimator = get_estimator(model)
    probabilities = model.predict_proba(X_test)
    positive_index = list(estimator.classes_).index("positive")

    return pd.Series(probabilities[:, positive_index], index=X_test.index)


def apply_positive_threshold(probabilities: pd.Series, positive_threshold: float) -> pd.Series:
    # повышенный порог делает модель осторожнее
    return pd.Series(
        ["positive" if value >= positive_threshold else "not_positive" for value in probabilities],
        index=probabilities.index,
    )


def train_models(
    models: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    positive_threshold: float,
) -> dict[str, dict[str, object]]:
    # обучение и предсказания через вероятность positive
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        positive_probability = get_positive_probabilities(model, X_test)
        y_pred = apply_positive_threshold(positive_probability, positive_threshold)
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "positive_probability": positive_probability,
        }

    return results


def compute_model_metrics(
    y_test: pd.Series,
    y_pred: pd.Series,
    class_labels: list[str],
) -> dict[str, float]:
    # основные метрики и риск ложного positive
    matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
    tn, fp, fn, tp = matrix.ravel()
    predicted_positive = tp + fp
    actual_not_positive = tn + fp

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "positive_precision": precision_score(
            y_test,
            y_pred,
            pos_label="positive",
            zero_division=0,
        ),
        "positive_recall": recall_score(
            y_test,
            y_pred,
            pos_label="positive",
            zero_division=0,
        ),
        "positive_f1": f1_score(
            y_test,
            y_pred,
            pos_label="positive",
            zero_division=0,
        ),
        "not_positive_precision": precision_score(
            y_test,
            y_pred,
            pos_label="not_positive",
            zero_division=0,
        ),
        "not_positive_recall": recall_score(
            y_test,
            y_pred,
            pos_label="not_positive",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_test,
            y_pred,
            labels=class_labels,
            average="macro",
            zero_division=0,
        ),
        "positive_wrong_rate": fp / predicted_positive if predicted_positive else 0,
        "false_positive_rate": fp / actual_not_positive if actual_not_positive else 0,
    }


def get_metrics_table(
    results: dict[str, dict[str, object]],
    y_test: pd.Series,
    class_labels: list[str],
) -> pd.DataFrame:
    # таблица метрик по всем моделям
    rows = []

    for model_name, result in results.items():
        metrics = compute_model_metrics(y_test, result["y_pred"], class_labels)
        metrics["model"] = model_name
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("model")
