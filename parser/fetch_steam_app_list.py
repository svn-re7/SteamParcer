import json
import os
import sys
from pathlib import Path
from urllib.error import URLError

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from project_paths import ENV_PATH, STEAM_APP_LIST_PATH
from parser.steam_api_client import SteamApiClient


API_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
MAX_RESULTS = 50_000


def fetch_app_list_page(
    client: SteamApiClient,
    api_key: str,
    last_appid: int | None = None,
) -> dict:
    # параметры запроса
    params = {
        "include_games": "true",
        "max_results": str(MAX_RESULTS),
        "key": api_key,
    }

    if last_appid is not None:
        params["last_appid"] = str(last_appid)

    return client.get_json(params)


def fetch_app_list(client: SteamApiClient, api_key: str) -> dict:
    # сбор страниц
    all_apps = []
    last_appid = None
    page_number = 1

    while True:
        data = fetch_app_list_page(client, api_key, last_appid)
        response = data.get("response", {})
        apps = response.get("apps", [])

        all_apps.extend(apps)
        print(f"страница {page_number}: {len(apps)} игр")

        if not response.get("have_more_results") or not apps:
            return {
                "response": {
                    "apps": all_apps,
                    "have_more_results": False,
                    "last_appid": response.get("last_appid"),
                    "total_apps": len(all_apps),
                    "pages": page_number,
                }
            }

        last_appid = response.get("last_appid") or apps[-1]["appid"]
        page_number += 1


def save_json(data: dict, path: Path) -> None:
    # сохранение файла
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    # подготовка ключа
    load_dotenv(ENV_PATH)
    api_key = os.getenv("API_KEY")

    if not api_key:
        raise RuntimeError("добавь API_KEY в .env")

    try:
        client = SteamApiClient(API_URL)
        data = fetch_app_list(client, api_key)
    except URLError as error:
        raise RuntimeError(f"ошибка: {error.reason}") from error

    save_json(data, STEAM_APP_LIST_PATH)

    apps_count = len(data.get("response", {}).get("apps", []))
    print(f"сохранено игр: {apps_count}")
    print(f"файл: {STEAM_APP_LIST_PATH}")


if __name__ == "__main__":
    main()
