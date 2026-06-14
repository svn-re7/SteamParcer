import json
import sys
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from parser.pipeline_config import (
    APP_LIMIT,
    BUILD_FROM_ALL_RAW,
    DETAILS_REQUEST_DELAY as REQUEST_DELAY,
    ERROR_SLEEP,
    LOAD_ALL_APPS,
)
from project_paths import (
    STEAM_APP_DETAILS_CSV_PATH,
    STEAM_APP_DETAILS_RAW_PATH,
    STEAM_APP_LIST_PATH,
)
from parser.steam_api_client import SteamApiClient


API_URL = "https://store.steampowered.com/api/appdetails"
FILTERS = "basic,price_overview,platforms,release_date,genres,categories,developers,publishers"

COLUMNS = [
    "appid",
    "name",
    "is_free",
    "developers",
    "publishers",
    "price_final",
    "price_currency",
    "windows",
    "mac",
    "linux",
    "categories",
    "genres",
    "release_date",
]


def load_app_ids(path: Path) -> np.ndarray:
    # загрузка appid
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    apps = data.get("response", {}).get("apps", [])
    app_ids = [app["appid"] for app in apps if "appid" in app]

    return np.array(app_ids, dtype=np.int64)


def load_raw_payloads(path: Path) -> dict[int, dict]:
    # чтение jsonl
    if not path.exists():
        return {}

    payloads = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except JSONDecodeError:
                continue

            payloads[int(record["appid"])] = record["payload"]

    return payloads


def normalize_list(value: object) -> list[str]:
    # список строк
    if not isinstance(value, list):
        return []

    return [str(item) for item in value if item]


def extract_descriptions(value: object) -> list[str]:
    # список описаний
    if not isinstance(value, list):
        return []

    descriptions = []

    for item in value:
        if isinstance(item, dict) and item.get("description"):
            descriptions.append(str(item["description"]))

    return descriptions


def normalize_release_date(value: object) -> str | None:
    # нормализация даты
    if not isinstance(value, dict) or value.get("coming_soon"):
        return None

    date_text = value.get("date")

    if not date_text:
        return None

    for date_format in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(date_text, date_format).date().isoformat()
        except ValueError:
            continue

    return None


def parse_appdetails_response(appid: int, payload: dict) -> dict | None:
    # парсинг json
    app_payload = payload.get(str(appid))

    if not app_payload or not app_payload.get("success"):
        return None

    data = app_payload.get("data")

    if not data:
        return None

    price = data.get("price_overview") or {}
    platforms = data.get("platforms") or {}

    developers = normalize_list(data.get("developers"))
    publishers = normalize_list(data.get("publishers"))
    categories = extract_descriptions(data.get("categories"))
    genres = extract_descriptions(data.get("genres"))

    return {
        "appid": appid,
        "name": data.get("name"),
        "is_free": data.get("is_free"),
        "developers": json.dumps(developers, ensure_ascii=False),
        "publishers": json.dumps(publishers, ensure_ascii=False),
        "price_final": price.get("final"),
        "price_currency": price.get("currency"),
        "windows": bool(platforms.get("windows", False)),
        "mac": bool(platforms.get("mac", False)),
        "linux": bool(platforms.get("linux", False)),
        "categories": json.dumps(categories, ensure_ascii=False),
        "genres": json.dumps(genres, ensure_ascii=False),
        "release_date": normalize_release_date(data.get("release_date")),
    }


def build_dataframe(app_ids: np.ndarray, payloads: dict[int, dict]) -> pd.DataFrame:
    # сбор датафрейма
    rows = []

    for appid in app_ids:
        appid = int(appid)
        payload = payloads.get(appid)

        if not payload:
            continue

        row = parse_appdetails_response(appid, payload)

        if row:
            rows.append(row)

    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    # подготовка данных
    app_ids = load_app_ids(STEAM_APP_LIST_PATH)
    app_ids = app_ids[::-1]
    payloads = load_raw_payloads(STEAM_APP_DETAILS_RAW_PATH)
    downloaded_app_ids = set(payloads)

    if BUILD_FROM_ALL_RAW:
        selected_app_ids = np.array(sorted(downloaded_app_ids), dtype=np.int64)
    else:
        selected_app_ids = app_ids if LOAD_ALL_APPS else app_ids[:APP_LIMIT]

    client = SteamApiClient(API_URL)
    total_selected = len(selected_app_ids)

    # загрузка деталей
    for index, appid in enumerate(selected_app_ids, start=1):
        appid = int(appid)

        if appid in downloaded_app_ids:
            print(f"пропуск {index}/{total_selected}: {appid}")
            continue

        params = {
            "appids": str(appid),
            "cc": "us",
            "l": "en",
            "filters": FILTERS,
        }

        payload = None

        while payload is None:
            try:
                payload = client.get_json(params)
            except (HTTPError, URLError, TimeoutError, ConnectionResetError, OSError, JSONDecodeError) as error:
                print(f"сбой {index}/{total_selected}: {appid}, {error}, сон {ERROR_SLEEP} сек")
                sleep(ERROR_SLEEP)

        STEAM_APP_DETAILS_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

        with STEAM_APP_DETAILS_RAW_PATH.open("a", encoding="utf-8") as file:
            json.dump({"appid": appid, "payload": payload}, file, ensure_ascii=False)
            file.write("\n")

        payloads[appid] = payload
        downloaded_app_ids.add(appid)

        print(f"загружено {index}/{total_selected}: {appid}")
        sleep(REQUEST_DELAY)

    # сохранение таблицы
    dataframe = build_dataframe(selected_app_ids, payloads)
    STEAM_APP_DETAILS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(STEAM_APP_DETAILS_CSV_PATH, index=False, encoding="utf-8")

    print(f"строк в csv: {len(dataframe)}")
    print(f"файл: {STEAM_APP_DETAILS_CSV_PATH}")


if __name__ == "__main__":
    main()
