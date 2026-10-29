"""Чтение, сборка и валидация сабмита соревнования.

Формат (data/submission_sample.csv): одна строка на пользователя, колонки
`user_id,item_id`, где `item_id` — строка из 10 идентификаторов фильмов
через пробел: `0 1 2 3 4 5 6 7 8 9`. Идентификаторы целочисленные:
пользователи 0..6039, фильмы 0..3705 (notes/data.md).

`validate_submission` — sanity-check перед отправкой на Kaggle (план E00):
полное покрытие тестовых пользователей, ровно 10 фильмов на пользователя,
без дубликатов, только известные item_id.
"""

import pandas as pd

COMPETITION = "hse-rec-sys-challenge-2026"
TOP_K = 10


def parse_items(value: object) -> list[int]:
    """Разбирает ячейку `item_id` в список из 10 уникальных идентификаторов."""
    tokens = str(value).split()
    items = [int(token) for token in tokens]
    if len(items) != TOP_K:
        raise ValueError(f"ожидалось {TOP_K} item_id, получено {len(items)}: {value!r}")
    if len(set(items)) != TOP_K:
        raise ValueError(f"дубликаты item_id внутри пользователя: {value!r}")
    return items


def load_submission(path: str) -> pd.DataFrame:
    """Читает файл сабмита, сохраняя `item_id` как исходную строку."""
    return pd.read_csv(path, dtype={"user_id": str, "item_id": str})


def build_submission(recs: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """Собирает сабмит из длинного формата рекомендаций.

    `recs` — колонки `user_id`, `item_id` и опционально `score` (больше —
    лучше). Без `score` сохраняется исходный порядок строк. На пользователя
    берутся первые `top_k` кандидатов; фильтрация уже просмотренных фильмов
    — ответственность модели, а не формата.
    """
    ordered = recs
    if "score" in recs.columns:
        ordered = recs.sort_values(
            ["user_id", "score"], ascending=[True, False], kind="stable"
        )
    top = ordered.groupby("user_id", sort=False).head(top_k)
    wide = top.groupby("user_id", sort=False)["item_id"].agg(
        lambda items: " ".join(map(str, items))
    )
    return wide.reset_index()


def validate_submission(
    df: pd.DataFrame,
    *,
    user_ids: set[int],
    item_ids: set[int],
) -> dict[str, int]:
    """Проверяет сабмит против известных пользователей и фильмов.

    Бросает ValueError со списком всех найденных проблем; возвращает сводку
    по прошедшему проверки файлу.
    """
    problems: list[str] = []

    if list(df.columns) != ["user_id", "item_id"]:
        raise ValueError(
            f"колонки должны быть ['user_id', 'item_id'], получены {list(df.columns)}"
        )

    try:
        users = df["user_id"].map(int)
    except ValueError as e:
        raise ValueError(f"user_id нецелочисленный: {e}") from e

    duplicated = users[users.duplicated()].unique().tolist()
    if duplicated:
        problems.append(f"повторяющиеся пользователи: {duplicated[:5]}")

    got_users = set(users)
    missing = sorted(user_ids - got_users)
    extra = sorted(got_users - user_ids)
    if missing:
        problems.append(f"отсутствуют пользователи: {missing[:5]} (всего {len(missing)})")
    if extra:
        problems.append(f"лишние пользователи: {extra[:5]} (всего {len(extra)})")

    items_used: set[int] = set()
    for user, item_str in zip(users, df["item_id"]):
        try:
            items_used.update(parse_items(item_str))
        except ValueError as e:
            problems.append(f"пользователь {user}: {e}")

    unknown = sorted(items_used - item_ids)
    if unknown:
        problems.append(f"неизвестные item_id: {unknown[:10]} (всего {len(unknown)})")

    if problems:
        raise ValueError("Сабмит невалиден:\n" + "\n".join(problems))

    return {"users": len(df), "unique_items": len(items_used)}


def save_submission(df: pd.DataFrame, path: str) -> None:
    """Записывает сабмит в формате соревнования."""
    df.to_csv(path, index=False)
