"""Проверки загрузчика данных: схема, dtypes, валидация, формат сабмита.

Запуск: uv run python tests/test_data.py
(файл совместим с pytest, если он появится позже)

test_real_dataset_numbers сверяет размеры из notes/data.md; пропускается,
если рядом нет data/events.csv.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recsys.data import (
    EVENTS_FILE,
    GENRE_COLUMNS,
    ITEM_FEATURES_FILE,
    SUBMISSION_SAMPLE_FILE,
    USER_FEATURES_FILE,
    DataValidationError,
    Dataset,
    load_dataset,
    load_events,
    load_item_features,
    load_submission_sample,
    load_user_features,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _events_df() -> pd.DataFrame:
    # user 0: 10 событий, user 1: 14 событий (минимальная реальная история)
    return pd.DataFrame(
        {
            "user_id": [0] * 10 + [1] * 14,
            "item_id": [*range(100, 110), *range(200, 214)],
            "rating": [4] * 10 + [3] * 14,
            "timestamp": [*range(10), *range(14)],
        }
    )


def _users_df() -> pd.DataFrame:
    return pd.DataFrame({"user_id": [0, 1], "gender": ["M", "F"], "age": [25, 56]})


def _items_df() -> pd.DataFrame:
    # item 999 есть в item_features, но не в events — как 16 реальных фильмов
    df = pd.DataFrame({"item_id": [*range(100, 110), *range(200, 214), 999]})
    df["genre_0"] = 1
    for column in GENRE_COLUMNS[1:]:
        df[column] = 0
    return df


def _submission_df() -> pd.DataFrame:
    items = "100 101 102 103 104 105 106 107 108 109"
    return pd.DataFrame({"user_id": [0, 1], "item_id": [items] * 2})


def _write_csv(directory: str, filename: str, df: pd.DataFrame) -> None:
    df.to_csv(Path(directory) / filename, index=False)


def _expect_data_error(fn, what: str) -> None:
    try:
        fn()
    except DataValidationError:
        return
    raise AssertionError(f"ожидали DataValidationError: {what}")


def test_load_dataset_happy_path():
    with tempfile.TemporaryDirectory() as tmp:
        _write_csv(tmp, EVENTS_FILE, _events_df())
        _write_csv(tmp, USER_FEATURES_FILE, _users_df())
        _write_csv(tmp, ITEM_FEATURES_FILE, _items_df())

        ds = load_dataset(tmp)
        assert isinstance(ds, Dataset)
        assert (ds.n_interactions, ds.n_users, ds.n_items) == (24, 2, 24)
        assert len(ds.user_features) == 2 and len(ds.item_features) == 25
        # фиксированные dtypes: ids int64, rating int8, timestamp int32
        assert [str(ds.events[c].dtype) for c in _events_df().columns] == [
            "int64", "int64", "int8", "int32",
        ]


def test_events_schema_errors():
    base = _events_df()

    bad_rating = base.copy()
    bad_rating.loc[0, "rating"] = 9

    negative_ts = base.copy()
    negative_ts.loc[0, "timestamp"] = -1

    dup_pair = base.copy()
    dup_pair.loc[1, "item_id"] = dup_pair.loc[0, "item_id"]

    dup_ts = base.copy()
    dup_ts.loc[1, "timestamp"] = dup_ts.loc[0, "timestamp"]

    cases = {
        "rating вне 1..5": bad_rating,
        "отрицательный timestamp": negative_ts,
        "дубликат пары (user, item)": dup_pair,
        "дубликат timestamp внутри истории": dup_ts,
        "лишняя колонка": base.assign(extra=1),
    }
    for what, events in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            _write_csv(tmp, EVENTS_FILE, events)
            _expect_data_error(lambda: load_events(tmp), what)


def test_user_features_schema_errors():
    base = _users_df()

    bad_gender = base.copy()
    bad_gender.loc[0, "gender"] = "X"

    bad_age = base.copy()
    bad_age.loc[0, "age"] = 30

    null_gender = base.copy()
    null_gender.loc[0, "gender"] = None

    cases = {
        "неизвестный gender": bad_gender,
        "незадокументированная age": bad_age,
        "пропуск в gender": null_gender,
        "дубликат user_id": pd.concat([base.iloc[[0]], base]),
        "нет колонки gender": base.drop(columns=["gender"]),
    }
    for what, users in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            _write_csv(tmp, USER_FEATURES_FILE, users)
            _expect_data_error(lambda: load_user_features(tmp), what)


def test_item_features_schema_errors():
    base = _items_df()

    non_binary = base.copy()
    non_binary.loc[0, "genre_0"] = 2

    cases = {
        "дубликат item_id": pd.concat([base.iloc[[0]], base]),
        "небинарный жанр": non_binary,
    }
    for what, items in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            _write_csv(tmp, ITEM_FEATURES_FILE, items)
            _expect_data_error(lambda: load_item_features(tmp), what)


def test_coverage_errors():
    # полный датасет, но из фич выпал пользователь / фильм, присутствующий в events
    with tempfile.TemporaryDirectory() as tmp:
        _write_csv(tmp, EVENTS_FILE, _events_df())
        _write_csv(tmp, ITEM_FEATURES_FILE, _items_df())
        _write_csv(tmp, USER_FEATURES_FILE, _users_df()[lambda df: df["user_id"] == 0])
        _expect_data_error(lambda: load_dataset(tmp), "пользователь без user features")

        _write_csv(tmp, USER_FEATURES_FILE, _users_df())
        _write_csv(tmp, ITEM_FEATURES_FILE, _items_df()[lambda df: df["item_id"] != 200])
        _expect_data_error(lambda: load_dataset(tmp), "фильм без item features")


def test_validate_false_skips_checks():
    events = _events_df().copy()
    events.loc[0, "rating"] = 9
    with tempfile.TemporaryDirectory() as tmp:
        _write_csv(tmp, EVENTS_FILE, events)
        loaded = load_events(tmp, validate=False)
    assert len(loaded) == 24 and (loaded["rating"] == 9).sum() == 1


def test_submission_sample_parsing():
    with tempfile.TemporaryDirectory() as tmp:
        _write_csv(tmp, SUBMISSION_SAMPLE_FILE, _submission_df())
        sub = load_submission_sample(tmp)
        assert list(sub.columns) == ["user_id", "item_ids"]
        assert sub["item_ids"][0] == list(range(100, 110))  # инты, а не строки


def test_submission_sample_schema_errors():
    base = _submission_df()

    nine_items = base.copy()
    nine_items.loc[0, "item_id"] = "100 101 102 103 104 105 106 107 108"

    dup_items = base.copy()
    dup_items.loc[0, "item_id"] = "100 100 102 103 104 105 106 107 108 109"

    empty_cell = base.copy()
    empty_cell.loc[0, "item_id"] = None

    cases = {
        "9 items вместо 10": nine_items,
        "дубликат item в строке": dup_items,
        "пустая ячейка": empty_cell,
    }
    for what, sub in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            _write_csv(tmp, SUBMISSION_SAMPLE_FILE, sub)
            _expect_data_error(lambda: load_submission_sample(tmp), what)


def test_real_dataset_numbers():
    """Размеры из notes/data.md; пропускается без data/."""
    if not (DATA_DIR / EVENTS_FILE).is_file():
        raise unittest.SkipTest("нет data/events.csv")
    ds = load_dataset(DATA_DIR)
    assert ds.n_interactions == 894_149
    assert ds.n_users == 6_040 and ds.n_items == 3_690
    assert len(ds.user_features) == 6_040 and len(ds.item_features) == 3_706
    sub = load_submission_sample(DATA_DIR)
    assert len(sub) == 6_040
    assert sub["item_ids"].map(lambda xs: len(set(xs)) == 10).all()


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except unittest.SkipTest as reason:
                print(f"skip {name} ({reason})")
                continue
            print(f"ok  {name}")
    print("все проверки пройдены")


if __name__ == "__main__":
    main()
