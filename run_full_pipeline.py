from threading import Thread

import build_steam_dataset
import fetch_steam_app_details
import fetch_steam_app_reviews


def run_thread(name: str, target, errors: list[tuple[str, BaseException]]) -> None:
    # запуск шага
    try:
        target()
    except BaseException as error:
        errors.append((name, error))


def main() -> None:
    # запуск парсеров
    errors = []
    threads = [
        Thread(
            target=run_thread,
            args=("details", fetch_steam_app_details.main, errors),
        ),
        Thread(
            target=run_thread,
            args=("reviews", fetch_steam_app_reviews.main, errors),
        ),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    if errors:
        name, error = errors[0]
        raise RuntimeError(f"ошибка в потоке {name}: {error}") from error

    # сбор финального датасета
    build_steam_dataset.main()


if __name__ == "__main__":
    main()
