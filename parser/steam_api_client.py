import json
from json import JSONDecodeError
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class SteamApiClient:
    def __init__(
        self,
        base_url: str,
        retries: int = 3,
        timeout: int = 60,
        sleep_seconds: int = 2,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.retries = retries
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }

    def get_json(self, params: dict[str, str]) -> dict:
        # подготовка запроса
        url = f"{self.base_url}?{urlencode(params)}"

        for attempt in range(1, self.retries + 1):
            try:
                return self._load_json(url)
            except HTTPError as error: # ловим ошибки http
                if not self._can_retry_http(error) or attempt == self.retries:
                    raise

                sleep(self.sleep_seconds * attempt)
            except (URLError, TimeoutError, ConnectionResetError, OSError, JSONDecodeError): # ловим временные сбои
                if attempt == self.retries:
                    raise

                sleep(self.sleep_seconds * attempt)

        raise RuntimeError("не удалось получить json")

    def _load_json(self, url: str) -> dict:
        # чтение ответа
        request = Request(url, headers=self.headers)

        with urlopen(request, timeout=self.timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset))


    @staticmethod
    def _can_retry_http(error: HTTPError) -> bool:
        # проверка статуса
        return error.code == 429 or error.code >= 500
