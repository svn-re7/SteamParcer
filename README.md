# SteamParcer

Проект собирает датасет игр Steam: сначала список AppID, потом метаданные игр, потом оценки из Steam Reviews, затем финальный CSV.

## Установка

Активировать окружение или запускать команды через `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

В `.env` должен быть ключ Steam API:

```env
API_KEY=your_key_here
```

## Где менять настройки

Основные настройки находятся в `pipeline_config.py`:

```python
APP_LIMIT = 110
LOAD_ALL_APPS = False

DETAILS_REQUEST_DELAY = 0.1
REVIEWS_REQUEST_DELAY = 0.1
ERROR_SLEEP = 10
```

- `APP_LIMIT` — сколько AppID брать, если `LOAD_ALL_APPS = False`
- `LOAD_ALL_APPS = True` — обработать весь список AppID
- `DETAILS_REQUEST_DELAY` — пауза между запросами `appdetails`
- `REVIEWS_REQUEST_DELAY` — пауза между запросами `appreviews`
- `ERROR_SLEEP` — сколько секунд ждать после сетевой ошибки

Пути к файлам лежат в `project_paths.py`.

## Порядок запуска

### 1. Скачать список игр

Запускается редко, чтобы получить общий список AppID:

```powershell
.\.venv\Scripts\python.exe fetch_steam_app_list.py
```

Результат:

```text
data/raw/steam_app_list.json
```

### 2. Запустить полный pipeline

Основной запуск на ночь:

```powershell
.\.venv\Scripts\python.exe run_full_pipeline.py
```

Он параллельно запускает:

- `fetch_steam_app_details.py`
- `fetch_steam_app_reviews.py`

После завершения обоих парсеров запускает:

- `build_steam_dataset.py`

## Что делает каждый файл

`steam_api_client.py`  
Общий HTTP-клиент для запросов к Steam. Добавляет headers, timeout и retry.

`fetch_steam_app_list.py`  
Скачивает список игр через `IStoreService/GetAppList` и сохраняет raw JSON.

`fetch_steam_app_details.py`  
Читает `steam_app_list.json`, дергает `appdetails`, сохраняет raw JSONL и собирает таблицу метаданных.

`fetch_steam_app_reviews.py`  
Читает `steam_app_list.json`, дергает `appreviews`, сохраняет raw JSONL и собирает таблицу оценок.

`build_steam_dataset.py`  
Объединяет таблицу метаданных и таблицу оценок по `appid`.

`run_full_pipeline.py`  
Запускает details и reviews в двух потоках, ждёт завершения и собирает финальный датасет.

`pipeline_config.py`  
Настройки лимита, задержек и полного сбора.

`project_paths.py`  
Все пути к raw и processed файлам.

## Выходные файлы

Raw-файлы:

```text
data/raw/steam_app_list.json
data/raw/steam_app_details.jsonl
data/raw/steam_app_reviews.jsonl
```

Промежуточные CSV:

```text
data/processed/steam_app_details.csv
data/processed/steam_app_reviews.csv
```

Финальный датасет:

```text
data/processed/steam_games_dataset.csv
```

## Повторный запуск

Парсеры сохраняют каждый успешный ответ сразу в `.jsonl`. Если запуск оборвался, можно просто снова запустить:

```powershell
.\.venv\Scripts\python.exe run_full_pipeline.py
```

Уже скачанные AppID будут пропущены, а недостающие продолжат скачиваться.

## Полный сбор

Для полного сбора поменять в `pipeline_config.py`:

```python
LOAD_ALL_APPS = True
```

После этого запустить:

```powershell
.\.venv\Scripts\python.exe run_full_pipeline.py
```

Для тестового запуска лучше оставить:

```python
LOAD_ALL_APPS = False
APP_LIMIT = 10
```
