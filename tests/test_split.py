"""Проверки leakage-safe сплитов на синтетических фреймах.

Запуск: uv run python tests/test_split.py
(файл совместим с pytest, если он появится позже)

test_real_data_split_numbers сверяет числа из notes/data.md; пропускается,
если рядом нет data/events.csv.
"""

import unittest
from fractions import Fraction
from math import ceil
from pathlib import Path

import pandas as pd

from recsys.data import DataValidationError
from recsys.split import chronological_split, last_n_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _events_mini() -> pd.DataFrame:
    # user 1: 10 событий, user 2: 14 событий (минимальная реальная история);
    # timestamp уникален внутри user
    return pd.DataFrame(
        {
            "user_id": [1] * 10 + [2] * 14,
            "item_id": [100 + i for i in range(10)]
            + [200 + i for i in range(14)],
            "rating": [4] * 24,
            "timestamp": [*range(10), *range(14)],
        }
    )


def _expect_error(fn, error=ValueError) -> None:
    try:
        fn()
    except error:
        return
    raise AssertionError(f"ожидали {error.__name__}")


def test_chronological_split():
    train, valid = chronological_split(_events_mini(), train_fraction=0.9)
    # user 1: ceil(10 * 0.1) = 1 valid; user 2: ceil(14 * 0.1) = 2 valid
    assert len(train) == 21 and len(valid) == 3
    assert set(train["user_id"]) == {1, 2} and set(valid["user_id"]) == {1, 2}
    # у user 2 в valid ровно два последних события истории
    assert set(valid[valid.user_id == 2]["timestamp"]) == {12, 13}
    # train — строгий префикс истории
    for user in (1, 2):
        assert train[train.user_id == user]["timestamp"].max() < valid[valid.user_id == user]["timestamp"].min()
    # leakage: пересечений пар user-item нет
    assert (
        set(zip(train["user_id"], train["item_id"]))
        & set(zip(valid["user_id"], valid["item_id"]))
    ) == set()


def test_last_n_split():
    train, valid = last_n_split(_events_mini(), n=3)
    assert len(valid) == 6 and valid.groupby("user_id").size().eq(3).all()
    for user in (1, 2):
        assert train[train.user_id == user]["timestamp"].max() < valid[valid.user_id == user]["timestamp"].min()


def test_validation_size_matches_exact_ceil():
    # регрессия epsilon-фикса: float-представление 1 - 0.7 = 0.30000000000000004
    # поднимало ceil выше точного целого (9.000000000000002 -> 10 у истории 30).
    # Сверяем размер valid каждого пользователя с рациональной арифметикой.
    sizes = range(2, 101)
    events = pd.DataFrame(
        [
            {"user_id": uid, "item_id": 1000 * uid + t, "rating": 4, "timestamp": t}
            for uid, n in enumerate(sizes, start=1)
            for t in range(n)
        ]
    )
    for fraction in [0.5, 0.7, 0.75, 0.8, 0.9, 0.95]:
        _, valid = chronological_split(events, train_fraction=fraction)
        got = valid.groupby("user_id").size().to_dict()
        expected = {
            uid: ceil(Fraction(n) * (1 - Fraction(str(fraction))))
            for uid, n in enumerate(sizes, start=1)
        }
        assert got == expected, f"train_fraction={fraction}"


def test_duplicate_timestamp_raises_loudly():
    # сырой read_csv с дубликатом timestamp должен падать,
    # а не разрешать тай-брейк порядком строк
    events = _events_mini()
    broken = pd.concat([events.iloc[[0]], events])
    _expect_error(lambda: chronological_split(broken), DataValidationError)


def test_rejects_impossible_input():
    single = pd.DataFrame({"user_id": [7], "item_id": [42], "rating": [5], "timestamp": [0]})
    _expect_error(lambda: chronological_split(single))
    empty = pd.DataFrame({"user_id": [], "item_id": [], "rating": [], "timestamp": []})
    _expect_error(lambda: chronological_split(empty))

    events = _events_mini()
    _expect_error(lambda: chronological_split(events, train_fraction=0.0))
    _expect_error(lambda: chronological_split(events, train_fraction=1.0))
    _expect_error(lambda: last_n_split(events, n=0))
    # у user 1 только 10 событий — last_n_split(n=15) невозможен
    _expect_error(lambda: last_n_split(events, n=15))
    # у user 2 ровно 14 событий: n=14 оставило бы его без train
    _expect_error(lambda: last_n_split(events, n=14))


def test_real_data_split_numbers():
    """Числа из notes/data.md и докстрингов split.py; пропускается без data/."""
    if not (DATA_DIR / "events.csv").is_file():
        raise unittest.SkipTest("нет data/events.csv")
    from recsys.data import load_events

    events = load_events(DATA_DIR)
    assert len(events) == 894_149
    train, valid = chronological_split(events)
    assert (len(train), len(valid)) == (802_016, 92_133)
    assert valid["user_id"].nunique() == 6_040
    _, valid10 = last_n_split(events)
    assert len(valid10) == 60_400


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
