"""Загрузка и проверка данных HSE RecSys Challenge 2026.

Читает четыре файла соревнования из одного каталога, приводит колонки
к фиксированным dtypes и проверяет инварианты схемы из ``notes/data.md``.
Все ноутбуки и эксперименты ходят за данными только через этот модуль,
чтобы типы и проверки не расходились между запусками.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EVENTS_FILE = "events.csv"
USER_FEATURES_FILE = "user_features.csv"
ITEM_FEATURES_FILE = "item_features.csv"
SUBMISSION_SAMPLE_FILE = "submission_sample.csv"

GENRE_COLUMNS = tuple(f"genre_{i}" for i in range(18))

# Документированные значения категориальных признаков (notes/data.md).
GENDER_VALUES = frozenset({"M", "F"})
AGE_VALUES = frozenset({1, 18, 25, 35, 45, 50, 56})

# ids оставляем int64 для стабильных join-ов, остальное сжимаем:
# события занимают основную память, и 894k строк в int8/int32 заметно легче int64.
_EVENT_DTYPES = {"user_id": "int64", "item_id": "int64", "rating": "int8", "timestamp": "int32"}


class DataValidationError(ValueError):
    """Данные не соответствуют схеме, зафиксированной в notes/data.md."""


@dataclass(frozen=True)
class Dataset:
    """Три таблицы соревнования с проверенной схемой."""

    events: pd.DataFrame
    user_features: pd.DataFrame
    item_features: pd.DataFrame

    @property
    def n_users(self) -> int:
        """Число пользователей, встречающихся в events."""
        return self.events["user_id"].nunique()

    @property
    def n_items(self) -> int:
        """Число фильмов, встречающихся в events."""
        return self.events["item_id"].nunique()

    @property
    def n_interactions(self) -> int:
        return len(self.events)


def load_dataset(data_dir: str | Path = "data", *, validate: bool = True) -> Dataset:
    """Загрузить events, user features и item features из ``data_dir``."""
    events = load_events(data_dir, validate=validate)
    user_features = load_user_features(data_dir, validate=validate)
    item_features = load_item_features(data_dir, validate=validate)
    if validate:
        _validate_coverage(events, user_features, item_features)
    return Dataset(events=events, user_features=user_features, item_features=item_features)


def load_events(data_dir: str | Path = "data", *, validate: bool = True) -> pd.DataFrame:
    """История взаимодействий: user_id, item_id, rating (1..5), timestamp.

    ``timestamp`` — позиция внутри истории пользователя, а не общая календарная ось
    (notes/data.md, раздел 5), поэтому сортироваться он может только внутри одного user_id.
    """
    df = _read_csv(data_dir, EVENTS_FILE, dtype=_EVENT_DTYPES)
    if validate:
        _validate_events(df)
    return df


def load_user_features(data_dir: str | Path = "data", *, validate: bool = True) -> pd.DataFrame:
    """Признаки пользователей: user_id, gender (M/F), age (категория 1..56)."""
    df = _read_csv(data_dir, USER_FEATURES_FILE, dtype={"user_id": "int64", "age": "int8"}, columns={"user_id", "gender", "age"})
    if validate:
        _validate_user_features(df)
    return df


def load_item_features(data_dir: str | Path = "data", *, validate: bool = True) -> pd.DataFrame:
    """Признаки фильмов: item_id и 18 бинарных колонок genre_0..genre_17."""
    df = _read_csv(data_dir, ITEM_FEATURES_FILE, dtype={"item_id": "int64", **{c: "int8" for c in GENRE_COLUMNS}}, columns={"item_id", *GENRE_COLUMNS})
    if validate:
        _validate_item_features(df)
    return df


def load_submission_sample(data_dir: str | Path = "data", *, validate: bool = True) -> pd.DataFrame:
    """Формат сабмита: колонки user_id и item_ids (список из 10 item_id).

    В исходном csv десять айтемов лежат одной строкой через пробел в колонке
    ``item_id``; здесь они разбираются в список, чтобы downstream-код не парсил их заново.
    """
    df = _read_csv(data_dir, SUBMISSION_SAMPLE_FILE, dtype={"user_id": "int64"}, columns={"user_id", "item_id"})
    df = df.rename(columns={"item_id": "item_ids"})
    # до разбора строк: NaN в ячейке иначе падает из .split() с AttributeError
    _require_no_nulls(df, SUBMISSION_SAMPLE_FILE)
    df["item_ids"] = df["item_ids"].map(lambda value: [int(x) for x in value.split()])
    if validate:
        _validate_submission_sample(df)
    return df


def _read_csv(data_dir: str | Path, filename: str, *, dtype, columns: set[str] | None = None) -> pd.DataFrame:
    path = Path(data_dir) / filename
    if not path.is_file():
        raise DataValidationError(f"не найден файл {path}")
    df = pd.read_csv(path, dtype=dtype)
    expected = set(dtype) if columns is None else columns
    if set(df.columns) != expected:
        raise DataValidationError(f"{filename}: ожидались колонки {sorted(expected)}, получено {sorted(df.columns)}")
    return df


def _require_no_nulls(df: pd.DataFrame, name: str) -> None:
    nulls = df.columns[df.isna().any()]
    if len(nulls):
        raise DataValidationError(f"{name}: пропуски в колонках {sorted(nulls)}")


def _validate_events(df: pd.DataFrame) -> None:
    _require_no_nulls(df, EVENTS_FILE)
    bad_rating = ~df["rating"].between(1, 5)
    if bad_rating.any():
        raise DataValidationError(f"{EVENTS_FILE}: rating вне диапазона 1..5, например {df.loc[bad_rating, 'rating'].iloc[0]}")
    if (df["timestamp"] < 0).any():
        raise DataValidationError(f"{EVENTS_FILE}: отрицательный timestamp")
    if df.duplicated(subset=["user_id", "item_id"]).any():
        raise DataValidationError(f"{EVENTS_FILE}: дубликаты пары (user_id, item_id)")
    if df.duplicated(subset=["user_id", "timestamp"]).any():
        raise DataValidationError(f"{EVENTS_FILE}: повторяющийся timestamp внутри истории пользователя")


def _validate_user_features(df: pd.DataFrame) -> None:
    _require_no_nulls(df, USER_FEATURES_FILE)
    if df["user_id"].duplicated().any():
        raise DataValidationError(f"{USER_FEATURES_FILE}: дубликаты user_id")
    bad_gender = ~df["gender"].isin(GENDER_VALUES)
    if bad_gender.any():
        raise DataValidationError(f"{USER_FEATURES_FILE}: неизвестные значения gender {sorted(df.loc[bad_gender, 'gender'].unique())}")
    bad_age = ~df["age"].isin(AGE_VALUES)
    if bad_age.any():
        raise DataValidationError(f"{USER_FEATURES_FILE}: незадокументированные категории age {sorted(df.loc[bad_age, 'age'].unique())}")


def _validate_item_features(df: pd.DataFrame) -> None:
    _require_no_nulls(df, ITEM_FEATURES_FILE)
    if df["item_id"].duplicated().any():
        raise DataValidationError(f"{ITEM_FEATURES_FILE}: дубликаты item_id")
    if not df[list(GENRE_COLUMNS)].isin([0, 1]).all().all():
        raise DataValidationError(f"{ITEM_FEATURES_FILE}: жанровые колонки должны быть бинарными 0/1")


def _validate_submission_sample(df: pd.DataFrame) -> None:
    if df["user_id"].duplicated().any():
        raise DataValidationError(f"{SUBMISSION_SAMPLE_FILE}: дубликаты user_id")
    bad_rows = df["item_ids"].map(lambda item_ids: len(item_ids) != 10 or len(set(item_ids)) != 10)
    if bad_rows.any():
        raise DataValidationError(f"{SUBMISSION_SAMPLE_FILE}: строки без 10 уникальных item_id, например user_id={df.loc[bad_rows, 'user_id'].iloc[0]}")


def _validate_coverage(events: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame) -> None:
    missing_users = set(events["user_id"]) - set(user_features["user_id"])
    if missing_users:
        raise DataValidationError(f"в events есть пользователи без user features: {min(missing_users)} и ещё {len(missing_users) - 1}")
    missing_items = set(events["item_id"]) - set(item_features["item_id"])
    if missing_items:
        raise DataValidationError(f"в events есть фильмы без item features: {min(missing_items)} и ещё {len(missing_items) - 1}")
