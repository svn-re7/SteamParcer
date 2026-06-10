import json
from json import JSONDecodeError
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError

import pandas as pd

from pipeline_config import (
    APP_LIMIT,
    ERROR_SLEEP,
    LOAD_ALL_APPS,
    REVIEWS_REQUEST_DELAY as REQUEST_DELAY,
)
from project_paths import (
    STEAM_APP_LIST_PATH,
    STEAM_APP_REVIEWS_CSV_PATH,
    STEAM_APP_REVIEWS_RAW_PATH,
)
from steam_api_client import SteamApiClient


API_URL = "https://store.steampowered.com/appreviews"

REVIEW_COLUMNS = [
    "appid",
    "review_score_desc",
    "total_positive",
    "total_negative",
]


def load_app_ids(path: Path) -> list[int]:
    # загрузка appid
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    apps = data.get("response", {}).get("apps", [])

    return [int(app["appid"]) for app in apps if "appid" in app]


def load_raw_payloads() -> dict[int, dict]:
    # чтение jsonl
    if not STEAM_APP_REVIEWS_RAW_PATH.exists():
        return {}

    payloads = {}

    with STEAM_APP_REVIEWS_RAW_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except JSONDecodeError:
                continue

            payloads[int(record["appid"])] = record["payload"]

    return payloads


def parse_reviews_response(appid: int, payload: dict) -> dict | None:
    # парсинг json
    summary = payload.get("query_summary")

    if not summary:
        return None

    return {
        "appid": appid,
        "review_score_desc": summary.get("review_score_desc"),
        "total_positive": summary.get("total_positive"),
        "total_negative": summary.get("total_negative"),
    }


def build_reviews_dataframe(app_ids: list[int], payloads: dict[int, dict]) -> pd.DataFrame:
    # сбор датафрейма
    rows = []

    for appid in app_ids:
        payload = payloads.get(appid)

        if not payload:
            continue

        row = parse_reviews_response(appid, payload)

        if row:
            rows.append(row)

    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def main() -> None:
    # подготовка данных
    all_app_ids = load_app_ids(STEAM_APP_LIST_PATH)
    app_ids = all_app_ids if LOAD_ALL_APPS else all_app_ids[:APP_LIMIT]
    payloads = load_raw_payloads()
    downloaded_app_ids = set(payloads)
    total_selected = len(app_ids)

    # загрузка отзывов
    for index, appid in enumerate(app_ids, start=1):
        if appid in downloaded_app_ids:
            print(f"пропуск {index}/{total_selected}: {appid}")
            continue

        params = {
            "json": "1",
            "language": "all",
            "purchase_type": "all",
            "num_per_page": "0",
        }

        payload = None
        client = SteamApiClient(f"{API_URL}/{appid}")

        while payload is None:
            try:
                payload = client.get_json(params)
            except (HTTPError, URLError, TimeoutError, ConnectionResetError, OSError, JSONDecodeError) as error:
                print(f"сбой {index}/{total_selected}: {appid}, {error}, сон {ERROR_SLEEP} сек")
                sleep(ERROR_SLEEP)

        STEAM_APP_REVIEWS_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

        with STEAM_APP_REVIEWS_RAW_PATH.open("a", encoding="utf-8") as file:
            json.dump({"appid": appid, "payload": payload}, file, ensure_ascii=False)
            file.write("\n")

        payloads[appid] = payload
        downloaded_app_ids.add(appid)

        print(f"загружено {index}/{total_selected}: {appid}")
        sleep(REQUEST_DELAY)

    # сохранение таблицы
    reviews_df = build_reviews_dataframe(app_ids, payloads)
    STEAM_APP_REVIEWS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    reviews_df.to_csv(STEAM_APP_REVIEWS_CSV_PATH, index=False, encoding="utf-8")

    print(f"строк в csv: {len(reviews_df)}")
    print(f"файл: {STEAM_APP_REVIEWS_CSV_PATH}")


if __name__ == "__main__":
    main()
