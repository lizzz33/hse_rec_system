"""Leakage-safe локальные сплиты (notes/data.md, раздел 6).

Общей календарной оси в данных нет: timestamp — позиция внутри истории
пользователя. Поэтому cutoff задаётся долей истории каждого пользователя,
а не глобальным числом. Оба сплита возвращают ``(train, validation)``,
где validation — хвост каждой пользовательской истории.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.data import DataValidationError


def chronological_split(events: pd.DataFrame, train_fraction: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primary local validation: последние ``ceil(n * (1 - train_fraction))`` событий пользователя.

    Ровно это правило даёт задокументированные числа на текущем events.csv:
    train 802 016 / validation 92 133, все 6 040 пользователей в validation.
    Пользователей с пустым train не возникает (минимальная история — 14 событий).
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction должен быть в (0, 1), получен {train_fraction}")
    rank, size = _timestamp_rank_and_size(events)
    # 1 - 0.7 в float равно 0.30000000000000004: n * fraction может подняться
    # чуть выше точного целого (30 -> 30.000000000000004) и ceil возьмёт лишнее
    # событие. Эпсилон гасит ошибку представления и не задевает настоящие
    # дробные части (проверено против точной рациональной арифметики, n <= 2056).
    validation_size = np.ceil(size * (1.0 - train_fraction) - 1e-9).astype(int)
    if (validation_size >= size).any():
        # ceil(n * (1 - fraction)) >= n возможно при n = 1 или очень малом train_fraction.
        raise ValueError("история из одного события или слишком маленький train_fraction: "
                         "у части пользователей весь history уходит в validation")
    is_validation = rank > size - validation_size
    return events[~is_validation].reset_index(drop=True), events[is_validation].reset_index(drop=True)


def last_n_split(events: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Альтернативный split: последние ``n`` событий каждого пользователя.

    Даёт ровно 10 ground-truth айтемов на пользователя (60 400 validation-строк),
    удобен как robustness-проверка рядом с Recall@10, но сильнее искажает долю
    holdout для коротких и длинных историй (notes/data.md, раздел 6).
    """
    if n < 1:
        raise ValueError(f"n должен быть >= 1, получен {n}")
    rank, size = _timestamp_rank_and_size(events)
    # size == n тоже невозможен: весь history уйдёт в valid, train останется пустым
    # (симметрично запрету пустого train в chronological_split)
    if (size <= n).any():
        shortest = int(size.min())
        raise ValueError(f"у части пользователей не больше {n} событий (минимум {shortest}): train после holdout пуст")
    is_validation = rank > size - n
    return events[~is_validation].reset_index(drop=True), events[is_validation].reset_index(drop=True)


def _timestamp_rank_and_size(events: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Позиция события в истории пользователя (1..n) и длина этой истории."""
    if events.empty:
        raise ValueError("events пуст")
    # rank(method="first") корректен только при уникальном (user_id, timestamp),
    # иначе тай-брейк тихо разрешается порядком строк. Эту гарантию даёт load_events;
    # проверяем и здесь, чтобы сырой read_csv падал громко, а не портил сплит.
    if events.duplicated(subset=["user_id", "timestamp"]).any():
        raise DataValidationError(
            "неуникальный timestamp внутри истории пользователя; "
            "загрузите events через load_events() вместо сырого read_csv"
        )
    group = events.groupby("user_id", sort=False)["timestamp"]
    return group.rank(method="first"), group.transform("size")
