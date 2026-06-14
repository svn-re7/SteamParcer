from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metrics(metrics_df: pd.DataFrame, path: Path) -> None:
    # график итоговых метрик
    plot_columns = [
        "accuracy",
        "balanced_accuracy",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "not_positive_precision",
        "not_positive_recall",
        "macro_f1",
        "positive_wrong_rate",
    ]
    plot_df = metrics_df[plot_columns].T
    plot_df.index = [
        "accuracy\n(TP+TN)/(TP+TN+FP+FN)",
        "balanced accuracy\n(TPR+TNR)/2",
        "positive precision\nTP/(TP+FP)",
        "positive recall\nTP/(TP+FN)",
        "positive f1\n2PR/(P+R)",
        "not positive precision\nTN/(TN+FN)",
        "not positive recall\nTN/(TN+FP)",
        "macro f1\n(F1_pos+F1_not)/2",
        "positive wrong rate\nFP/(TP+FP)",
    ]

    fig, ax = plt.subplots(figsize=(16, 7))
    plot_df.plot(kind="bar", ax=ax, rot=25, width=0.8)
    ax.set_title("Сравнение моделей классификации оценки пользователей", fontsize=13)
    ax.set_ylabel("Значение метрики", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(title="Модель")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)

    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(
    results: dict[str, dict[str, object]],
    y_test: pd.Series,
    class_labels: list[str],
    positive_threshold: float,
    path: Path,
) -> None:
    # heatmap ошибок
    matrices = {
        model_name: confusion_matrix(y_test, result["y_pred"], labels=class_labels)
        for model_name, result in results.items()
    }
    row_shares = {
        model_name: pd.DataFrame(matrix).div(matrix.sum(axis=1), axis=0).fillna(0).to_numpy()
        for model_name, matrix in matrices.items()
    }
    column_shares = {
        model_name: pd.DataFrame(matrix).div(matrix.sum(axis=0), axis=1).fillna(0).to_numpy()
        for model_name, matrix in matrices.items()
    }
    max_share = max(share.max() for share in row_shares.values())
    cell_names = [["TN", "FP"], ["FN", "TP"]]

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(7 * len(results), 5.5),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, (model_name, matrix) in zip(axes[0], matrices.items()):
        row_share = row_shares[model_name]
        column_share = column_shares[model_name]
        ax.imshow(row_share, cmap="Blues", vmin=0, vmax=max_share)

        ax.set_title(f"{model_name}, threshold={positive_threshold}")
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_xticks(range(len(class_labels)), class_labels)
        ax.set_yticks(range(len(class_labels)), class_labels)

        for row_index in range(2):
            for column_index in range(2):
                row_value = row_share[row_index, column_index]
                column_value = column_share[row_index, column_index]
                count = matrix[row_index, column_index]
                color = "white" if row_value > max_share / 2 else "black"
                ax.text(
                    column_index,
                    row_index,
                    f"{cell_names[row_index][column_index]}\nn={count}\nrow {row_value:.1%}\ncol {column_value:.1%}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(
    results: dict[str, dict[str, object]],
    y_test: pd.Series,
    path: Path,
) -> None:
    # roc и precision-recall по вероятностям positive
    y_true = (y_test == "positive").astype(int)
    positive_rate = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    roc_ax, pr_ax = axes

    for model_name, result in results.items():
        probabilities = result["positive_probability"]

        fpr, tpr, _ = roc_curve(y_true, probabilities)
        roc_auc = roc_auc_score(y_true, probabilities)
        roc_ax.plot(fpr, tpr, label=f"{model_name}, AUC={roc_auc:.3f}")

        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        average_precision = average_precision_score(y_true, probabilities)
        pr_ax.plot(recall, precision, label=f"{model_name}, AP={average_precision:.3f}")

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="random")
    roc_ax.set_title("ROC-кривая")
    roc_ax.set_xlabel("false positive rate")
    roc_ax.set_ylabel("true positive rate")
    roc_ax.grid(alpha=0.3)
    roc_ax.legend()

    pr_ax.axhline(
        positive_rate,
        linestyle="--",
        color="#777777",
        label=f"baseline={positive_rate:.3f}",
    )
    pr_ax.set_title("Precision-Recall кривая")
    pr_ax.set_xlabel("recall")
    pr_ax.set_ylabel("precision")
    pr_ax.grid(alpha=0.3)
    pr_ax.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def get_threshold_metrics(
    y_test: pd.Series,
    probabilities: pd.Series,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    # метрики при разных порогах positive
    rows = []

    for threshold in thresholds:
        y_pred = pd.Series(
            np.where(probabilities >= threshold, "positive", "not_positive"),
            index=y_test.index,
        )
        matrix = confusion_matrix(
            y_test,
            y_pred,
            labels=["not_positive", "positive"],
        )
        tn, fp, fn, tp = matrix.ravel()
        predicted_positive = tp + fp
        actual_not_positive = tn + fp

        rows.append(
            {
                "threshold": threshold,
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
                "positive_wrong_rate": fp / predicted_positive if predicted_positive else 0,
                "false_positive_rate": fp / actual_not_positive if actual_not_positive else 0,
            }
        )

    return pd.DataFrame(rows)


def plot_threshold_metrics(
    results: dict[str, dict[str, object]],
    y_test: pd.Series,
    positive_threshold: float,
    path: Path,
) -> None:
    # как меняются метрики при разных threshold
    thresholds = np.arange(0.05, 0.91, 0.05)
    plot_columns = [
        "positive_precision",
        "positive_recall",
        "positive_wrong_rate",
        "false_positive_rate",
    ]
    plot_labels = {
        "positive_precision": "positive precision",
        "positive_recall": "positive recall",
        "positive_wrong_rate": "wrong among predicted positive",
        "false_positive_rate": "false positive rate",
    }

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(7 * len(results), 5.5),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, (model_name, result) in zip(axes[0], results.items()):
        metrics_df = get_threshold_metrics(
            y_test=y_test,
            probabilities=result["positive_probability"],
            thresholds=thresholds,
        )

        for column in plot_columns:
            ax.plot(
                metrics_df["threshold"],
                metrics_df[column],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=plot_labels[column],
            )

        ax.axvline(
            positive_threshold,
            color="#333333",
            linestyle="--",
            linewidth=1,
            label=f"current threshold={positive_threshold}",
        )
        ax.set_title(model_name)
        ax.set_xlabel("positive threshold")
        ax.set_ylabel("metric value")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Метрики при разных порогах positive", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
