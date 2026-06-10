from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

STEAM_APP_LIST_PATH = RAW_DATA_DIR / "steam_app_list.json"
STEAM_APP_DETAILS_RAW_PATH = RAW_DATA_DIR / "steam_app_details.jsonl"
STEAM_APP_DETAILS_CSV_PATH = PROCESSED_DATA_DIR / "steam_app_details.csv"
