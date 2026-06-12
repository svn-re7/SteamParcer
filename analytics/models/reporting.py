from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.metrics import confusion_matrix


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
        "false_positive_rate",
    ]
    plot_df = metrics_df[plot_columns].T
    plot_df.index = [
        "accuracy",
        "balanced accuracy",
        "positive precision",
        "positive recall",
        "positive f1",
        "not positive precision",
        "not positive recall",
        "macro f1",
        "positive wrong rate",
        "false positive rate",
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_df.plot(kind="bar", ax=ax, rot=30, width=0.8)
    ax.set_title("Сравнение моделей классификации оценки пользователей", fontsize=13)
    ax.set_ylabel("Значение метрики", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(title="Модель")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)

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
