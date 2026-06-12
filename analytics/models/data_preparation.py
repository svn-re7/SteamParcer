import json
from collections import Counter

import pandas as pd


def parse_json_list(value: object) -> list[str]:
    # список из json-строки
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(parsed, list):
        return []

    return [str(item) for item in parsed if item]


def get_top_items(df: pd.DataFrame, column: str, limit: int) -> list[str]:
    # частые элементы спискового поля
    counter = Counter()

    for items in df[column].apply(parse_json_list):
        counter.update(items)

    return [item for item, _ in counter.most_common(limit)]


def to_binary(series: pd.Series) -> pd.Series:
    # bool в 0/1
    if series.dtype == bool:
        return series.astype(int)

    return series.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)


def add_list_features(
    features: pd.DataFrame,
    source_df: pd.DataFrame,
    column: str,
    top_items: list[str],
    prefix: str,
) -> None:
    # one-hot признаки жанров и категорий
    parsed_values = source_df[column].apply(lambda value: set(parse_json_list(value)))

    for item in top_items:
        features[f"{prefix}_{item}"] = parsed_values.apply(lambda values: int(item in values))


def prepare_features(
    df: pd.DataFrame,
    top_genres: list[str],
    top_categories: list[str],
    current_year: int,
) -> pd.DataFrame:
    # базовые числовые и бинарные признаки
    features = pd.DataFrame(index=df.index)
    features["is_free"] = to_binary(df["is_free"])
    features["windows"] = to_binary(df["windows"])
    features["linux"] = to_binary(df["linux"])
    features["mac"] = to_binary(df["mac"])

    # цена: free игры получают 0, платные пропуски получают медиану
    price = pd.to_numeric(df["price_final"], errors="coerce")
    free_mask = features["is_free"] == 1
    paid_price_median = price[(~free_mask) & price.notna()].median()

    if pd.isna(paid_price_median):
        paid_price_median = 0

    price = price.mask(free_mask & price.isna(), 0)
    features["price_final"] = price.fillna(paid_price_median)

    # дата выхода превращается в год, месяц и возраст игры
    release_date = pd.to_datetime(df["release_date"], errors="coerce")
    features["release_year"] = release_date.dt.year
    features["release_month"] = release_date.dt.month
    features["game_age_years"] = current_year - features["release_year"]

    for column in ["release_year", "release_month", "game_age_years"]:
        median_value = features[column].median()

        if pd.isna(median_value):
            median_value = 0

        features[column] = features[column].fillna(median_value)

    # списковые признаки кодируются по самым частым значениям
    add_list_features(features, df, "genres", top_genres, "genre")
    add_list_features(features, df, "categories", top_categories, "category")

    return features


def prepare_model_data(
    df: pd.DataFrame,
    target_mapping: dict[str, str],
    top_n_genres: int,
    top_n_categories: int,
    current_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, int]:
    # целевая переменная positive/not_positive
    df = df.copy()
    df["review_class"] = df["review_score_desc"].map(target_mapping)

    excluded_rows = int(df["review_class"].isna().sum())
    df = df.dropna(subset=["review_class"]).copy()

    # топы считаются после удаления строк без таргета
    top_genres = get_top_items(df, "genres", top_n_genres)
    top_categories = get_top_items(df, "categories", top_n_categories)

    X = prepare_features(
        df=df,
        top_genres=top_genres,
        top_categories=top_categories,
        current_year=current_year,
    )
    y = df["review_class"]

    return df, X, y, excluded_rows
