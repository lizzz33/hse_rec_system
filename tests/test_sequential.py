"""Проверки sequential-скоринга E09 (markov + last-K контекст) на синтетике.

Запуск: uv run python tests/test_sequential.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.sequential import (
    combine_rows,
    context_score_rows,
    last_context,
    markov_weights,
    popularity_vector,
    sorted_histories,
    topk_from_scores,
)


def _train_mini() -> pd.DataFrame:
    # user 1: 100 -> 101 -> 102;  user 2: 100 -> 101 -> 200
    # переходы: 100->101 ×2, 101->102 ×1, 101->200 ×1
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "item_id": [100, 101, 102, 100, 101, 200],
            "rating": [5, 4, 5, 3, 4, 2],
            "timestamp": [0, 1, 2, 0, 1, 2],
        }
    )


def test_sorted_histories_order_by_timestamp():
    shuffled = _train_mini().sample(frac=1.0, random_state=1)
    histories = sorted_histories(shuffled)
    assert histories[1].tolist() == [100, 101, 102]
    assert histories[2].tolist() == [100, 101, 200]


def test_last_context_recent_first_and_padding():
    items = np.array([100, 101, 102, 200])
    histories = {1: np.array([100, 101, 102]), 2: np.array([100])}
    users, context = last_context(histories, items, k=3)
    # самые свежие первыми, позиции в каталоге, короткая история — паддинг -1
    assert users.tolist() == [1, 2]
    assert context[0].tolist() == [2, 1, 0]  # 102, 101, 100
    assert context[1].tolist() == [0, -1, -1]  # 100


def test_markov_row_stochastic_and_counts():
    train = _train_mini()
    items = np.sort(train["item_id"].unique())
    weights = markov_weights(train, items, alpha=0.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    pos = {item: i for i, item in enumerate(items)}
    # 100->101 единственный переход из 100: P = 1
    assert weights[pos[100], pos[101]] == 1.0
    # из 101 два равных перехода
    assert np.isclose(weights[pos[101], pos[102]], 0.5)
    assert np.isclose(weights[pos[101], pos[200]], 0.5)
    # 102 и 200 не имеют исходящих переходов — backoff к равномерному
    assert np.allclose(weights[pos[102]], 0.25)
    assert np.allclose(weights[pos[200]], 0.25)
    # сглаживание делает строки строго положительными и сохраняет сумму 1
    smoothed = markov_weights(train, items, alpha=0.5)
    assert (smoothed > 0).all()
    assert np.allclose(smoothed.sum(axis=1), 1.0)


def test_context_score_decay_and_depth():
    items = np.array([100, 101, 102, 200])
    # W[100] = 1 везде, W[101] = 2 везде, прочие нули
    weights = np.zeros((4, 4), dtype=np.float32)
    weights[0, :] = 1.0
    weights[1, :] = 2.0
    context = np.array([[1, 0, -1]])  # свежее 101, затем 100
    rows = context_score_rows(context, weights)
    # глубина 1: только W[101]; глубина 2 c decay: W[101] + 0.5·W[100]
    assert np.allclose(combine_rows(rows, 1, 0.5), 2.0)
    assert np.allclose(combine_rows(rows, 2, 0.5), 2.0 + 0.5 * 1.0)
    # отсутствующие события (-1) не добавляют ничего
    assert np.allclose(combine_rows(rows, 3, 0.5), 2.0 + 0.5 * 1.0)


def test_popularity_vector_normalized():
    train = _train_mini()
    items = np.sort(train["item_id"].unique())
    pop = popularity_vector(train, items)
    assert np.isclose(pop.sum(), 1.0)
    assert pop.argmax() == 0  # item 100 встречается дважды


def test_topk_filters_seen():
    train = _train_mini()
    items = np.sort(train["item_id"].unique())
    histories = sorted_histories(train)
    users, context = last_context(histories, items, k=3)
    scores = np.full((len(users), len(items)), 1.0, dtype=np.float32)
    recs = topk_from_scores(scores, users, items, train, k=2)
    seen = set(zip(train["user_id"], train["item_id"]))
    assert not set(zip(recs["user_id"], recs["item_id"])) & seen
    # каталог 4 items, история по 3 — каждому остаётся ровно 1 unseen-кандидат
    assert recs.groupby("user_id").size().eq(1).all()


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("все проверки пройдены")


if __name__ == "__main__":
    main()
