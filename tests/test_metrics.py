"""Проверки метрики на вручную посчитанных случаях.

Запуск: uv run python tests/test_metrics.py
(файл совместим с pytest, если он появится позже)
"""

import tempfile
from pathlib import Path

import pandas as pd

from recsys.metrics import evaluate, load_submission, recall_at_k, save_submission


def _approx(a: float, b: float) -> bool:
    return abs(a - b) < 1e-12


def test_recall_variants():
    recs = pd.DataFrame(
        {
            "user_id": [1] * 10 + [2] * 10 + [4] * 10,
            "item_id": [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,      # user 1
                5, 5, 6, 7, 8, 9, 10, 11, 12, 13,    # user 2: дубль 5 в топ-k
                *range(100, 110),                    # user 4: без ground truth
            ],
        }
    )
    gt = pd.DataFrame(
        {
            "user_id": [1] * 12 + [2] * 3 + [3] * 2,
            "item_id": [
                *range(7, 19),  # user 1: gt {7..18}, 12 items, хиты {7,8,9,10} = 4
                5, 6, 99,       # user 2: gt {5,6,99}, хиты {5,6} = 2 (5 один раз)
                1, 2,           # user 3: gt есть, рекомендаций нет -> 0
            ],
        }
    )

    # macro / relevant: (4/12 + 2/3 + 0/2) / 3
    assert _approx(
        recall_at_k(recs, gt), (4 / 12 + 2 / 3) / 3
    )
    # macro / capped: (4/10 + 2/3 + 0/2) / 3
    assert _approx(
        recall_at_k(recs, gt, denominator="capped"), (4 / 10 + 2 / 3) / 3
    )
    # macro / k: (4/10 + 2/10 + 0) / 3
    assert _approx(
        recall_at_k(recs, gt, denominator="k"), (0.4 + 0.2) / 3
    )
    # micro: 6 хитов / 17 релевантных
    assert _approx(recall_at_k(recs, gt, average="micro"), 6 / 17)

    # k=5: user 1 берёт только [1..5] -> хит 0; user 2 [5,5,6,7,8] -> хиты {5,6}
    assert _approx(
        recall_at_k(recs, gt, k=5), (0 / 12 + 2 / 3) / 3
    )


def test_evaluate_panel_and_diagnostics():
    recs = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [7, 7, 5, 6],
            "rank": [1, 2, 1, 2],
        }
    )
    train = pd.DataFrame({"user_id": [1, 1], "item_id": [7, 999]})
    gt = pd.DataFrame({"user_id": [1, 2, 2], "item_id": [7, 5, 6]})

    panel = evaluate(recs, gt, k=2, train_events=train)
    # user 1: топ-2 = [7, 7] -> уникальный хит {7}, 1 из 1; дубль занимает слот
    # user 2: хиты {5, 6}, 2 из 2
    assert _approx(panel["recall@2_macro_relevant"], (1 / 1 + 1.0) / 2)
    assert _approx(panel["recall@2_micro"], 3 / 3)
    assert panel["n_eval_users"] == 2
    assert panel["users_without_recs"] == 0
    assert _approx(panel["dup_in_top_share"], 1 / 4)
    # item 7 у user 1 уже оценён в train и занимает оба слота [7, 7] -> 2 из 4
    assert _approx(panel["seen_in_top_share"], 2 / 4)
    assert panel["n_unique_rec_items"] == 3


def test_evaluate_min_rating():
    gt = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [5, 6],
            "rating": [5, 2],
        }
    )
    recs = pd.DataFrame({"user_id": [1, 1], "item_id": [5, 6]})
    panel = evaluate(recs, gt, k=2, min_rating=4)
    # после фильтра gt = {5}: рекомендован и 5, и 6 -> recall 1.0
    assert _approx(panel["recall@2_macro_relevant"], 1.0)


def test_users_without_recs_get_zero():
    gt = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 10]})
    recs = pd.DataFrame({"user_id": [1], "item_id": [10]})
    assert _approx(recall_at_k(recs, gt), 0.5)
    panel = evaluate(recs, gt)
    assert panel["users_without_recs"] == 1


def test_submission_roundtrip():
    recs = pd.DataFrame(
        {
            "user_id": [2, 2, 1, 1, 1],
            "item_id": [30, 20, 10, 10, 11],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "submission.csv")
        save_submission(recs, path, k=2)
        raw = Path(path).read_text().strip().splitlines()
        assert raw[0] == "user_id,item_id"
        assert raw[1] == "1,10 11"          # дубликат 10 выброшен, слот добран
        assert raw[2] == "2,30 20"          # порядок строк сохранён
        loaded = load_submission(path)
        assert loaded == {1: [10, 11], 2: [30, 20]}


def test_save_submission_rejects_short_lists():
    recs = pd.DataFrame({"user_id": [1, 1], "item_id": [1, 1]})
    try:
        save_submission(recs, "/dev/null", k=2)
    except ValueError:
        return
    raise AssertionError("ожидали ValueError при < k уникальных items")


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("все проверки пройдены")


if __name__ == "__main__":
    main()
