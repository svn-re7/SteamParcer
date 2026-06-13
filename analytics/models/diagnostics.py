from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_GROUP_COLORS = {
    "base": "#4C78A8",
    "date": "#F2CF5B",
    "genre": "#72B55B",
    "category": "#F28E2B",
}


def get_feature_group(feature_name: str) -> str:
    # тип признака по имени колонки
    if feature_name.startswith("genre_"):
        return "genre"

    if feature_name.startswith("category_"):
        return "category"

    if feature_name in ["release_year", "release_month", "game_age_years"]:
        return "date"

    return "base"


def get_high_correlation_pairs(
    X: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    # пары признаков с сильной корреляцией
    correlation = X.corr(numeric_only=True)
    upper_mask = np.triu(np.ones(correlation.shape), k=1).astype(bool)
    upper = correlation.where(upper_mask)
    rows = []

    for left_feature, row in upper.iterrows():
        for right_feature, value in row.dropna().items():
            if abs(value) >= threshold:
                rows.append(
                    {
                        "feature_1": left_feature,
                        "feature_2": right_feature,
                        "correlation": value,
                        "abs_correlation": abs(value),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=["feature_1", "feature_2", "correlation", "abs_correlation"]
        )

    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    # vif важен прежде всего для logistic regression
    features = list(X.columns)
    rows = []

    for feature in features:
        y = X[feature]

        if y.nunique(dropna=False) <= 1:
            rows.append({"feature": feature, "vif": np.nan})
            continue

        other_features = [name for name in features if name != feature]
        model = LinearRegression()
        model.fit(X[other_features], y)
        r_squared = model.score(X[other_features], y)

        if r_squared >= 0.999999:
            vif = np.inf
        else:
            vif = 1 / (1 - r_squared)

        rows.append({"feature": feature, "vif": vif})

    vif_df = pd.DataFrame(rows)
    vif_df["vif_for_sort"] = vif_df["vif"].replace(np.inf, np.nan)
    vif_df["vif_for_sort"] = vif_df["vif_for_sort"].fillna(vif_df["vif_for_sort"].max() + 1)

    return vif_df.sort_values("vif_for_sort", ascending=False).drop(columns=["vif_for_sort"])


def get_model_feature_scores(
    results: dict[str, dict[str, object]],
    feature_names: list[str],
) -> dict[str, pd.DataFrame]:
    # важности random forest и коэффициенты logistic regression
    scores = {}

    for model_name, result in results.items():
        model = result["model"]

        if model_name == "RandomForest":
            values = model.feature_importances_
            scores[model_name] = pd.DataFrame(
                {
                    "feature": feature_names,
                    "score": values,
                    "abs_score": np.abs(values),
                    "direction": "importance",
                }
            ).sort_values("abs_score", ascending=False)
            continue

        if model_name == "LogisticRegression":
            estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
            positive_index = list(estimator.classes_).index("positive")
            if estimator.coef_.shape[0] == 1:
                values = estimator.coef_[0]
            else:
                values = estimator.coef_[positive_index]
            scores[model_name] = pd.DataFrame(
                {
                    "feature": feature_names,
                    "score": values,
                    "abs_score": np.abs(values),
                    "direction": np.where(values >= 0, "positive", "not_positive"),
                }
            ).sort_values("abs_score", ascending=False)

    return scores


def plot_feature_correlation_matrix(
    X: pd.DataFrame,
    path: Path,
    max_features: int = 60,
) -> None:
    # heatmap корреляций финальных признаков
    variances = X.var(numeric_only=True).sort_values(ascending=False)
    selected_features = list(variances.head(max_features).index)
    correlation = X[selected_features].corr(numeric_only=True)

    figure_size = max(15, len(selected_features) * 0.32)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size * 0.9))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(
        "Корреляция факторов модели",
        fontsize=14,
        pad=16,
    )
    ax.set_xticks(range(len(selected_features)), selected_features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(selected_features)), selected_features, fontsize=7)
    ax.set_xlabel("факторы")
    ax.set_ylabel("факторы")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="correlation")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_logistic_vif(
    vif_df: pd.DataFrame,
    path: Path,
    top_n: int = 25,
) -> None:
    # топ vif для проверки мультиколлинеарности
    plot_df = vif_df.head(top_n).copy()
    finite_values = plot_df["vif"].replace(np.inf, np.nan)
    cap_value = finite_values.max()

    if pd.isna(cap_value):
        cap_value = 10

    plot_df["vif_for_plot"] = plot_df["vif"].replace(np.inf, cap_value * 1.15)
    plot_df = plot_df.iloc[::-1]

    colors = [
        "#D95F02" if value == np.inf or value >= 10 else "#4C78A8"
        for value in plot_df["vif"]
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(plot_df["feature"], plot_df["vif_for_plot"], color=colors)
    ax.axvline(5, color="#777777", linestyle="--", linewidth=1, label="vif 5")
    ax.axvline(10, color="#333333", linestyle="--", linewidth=1, label="vif 10")
    ax.set_title("VIF для факторов LogisticRegression", fontsize=14, pad=16)
    ax.set_xlabel("vif")
    ax.set_ylabel("фактор")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()

    for bar, value in zip(bars, plot_df["vif"]):
        label = "inf" if value == np.inf else f"{value:.1f}"
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {label}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_features(
    feature_names: list[str],
    path: Path,
) -> None:
    # все признаки, которые ушли в модель
    rows = [
        {
            "feature": feature_name,
            "group": get_feature_group(feature_name),
            "value": 1,
        }
        for feature_name in feature_names
    ]
    plot_df = pd.DataFrame(rows)
    group_order = {"base": 0, "date": 1, "genre": 2, "category": 3}
    plot_df["group_order"] = plot_df["group"].map(group_order)
    plot_df = plot_df.sort_values(["group_order", "feature"], ascending=[True, True])
    plot_df = plot_df.iloc[::-1]

    fig_height = max(9, len(plot_df) * 0.24)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    colors = [FEATURE_GROUP_COLORS[group] for group in plot_df["group"]]
    ax.barh(plot_df["feature"], plot_df["value"], color=colors)
    ax.set_title("Финальные признаки модели", fontsize=14, pad=16)
    ax.set_xlabel("признак включен")
    ax.set_ylabel("фактор")
    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    ax.grid(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=group)
        for group, color in FEATURE_GROUP_COLORS.items()
    ]
    ax.legend(handles=handles, title="тип признака", loc="lower right")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_feature_scores(
    feature_scores: dict[str, pd.DataFrame],
    path: Path,
    top_n: int = 20,
) -> None:
    # сравнение того, какие признаки важны для разных моделей
    fig, axes = plt.subplots(
        1,
        len(feature_scores),
        figsize=(8 * len(feature_scores), 8),
        squeeze=False,
    )

    for ax, (model_name, scores_df) in zip(axes[0], feature_scores.items()):
        plot_df = scores_df.head(top_n).iloc[::-1].copy()

        if model_name == "RandomForest":
            colors = ["#4C78A8"] * len(plot_df)
            xlabel = "feature importance"
        else:
            colors = [
                "#4C78A8" if value >= 0 else "#D95F02"
                for value in plot_df["score"]
            ]
            xlabel = "standardized coefficient"

        bars = ax.barh(plot_df["feature"], plot_df["score"], color=colors)
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title(model_name, fontsize=13)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("фактор")
        ax.grid(axis="x", alpha=0.25)

        for bar, value in zip(bars, plot_df["score"]):
            offset = 0.01 if value >= 0 else -0.01
            ha = "left" if value >= 0 else "right"
            ax.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha=ha,
                fontsize=8,
            )

    fig.suptitle("Важность факторов по моделям", fontsize=14, y=1.02)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
