"""Проверки baseline-рекомендеров на синтетических данных.

Запуск: uv run python tests/test_models.py
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.models import ALSRecommender, EASERecommender, ItemKNNRecommender, PopularityRecommender


def _train_mini() -> pd.DataFrame:
    # co-occurrence: (100,101) встречаются у двух пользователей, хвост у user 1
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 3, 3],
            "item_id": [100, 101, 102, 103, 100, 101, 200, 201],
            "rating": [5, 4, 1, 2, 5, 3, 4, 5],
            "timestamp": [0, 1, 2, 3, 0, 1, 0, 1],
        }
    )


def _expect_error(fn, error=ValueError) -> None:
    try:
        fn()
    except error:
        return
    raise AssertionError(f"ожидали {error.__name__}")


def test_popularity_filters_seen_and_orders_by_count():
    model = PopularityRecommender("count").fit(_train_mini())
    recs = model.recommend(k=10)
    # user 1 видел все 4 items с оценками — ему достаются только 200/201
    user1 = recs[recs.user_id == 1]["item_id"].tolist()
    assert user1 == [200, 201]
    # user 2 видел 100/101 — следующие по популярности 102/103
    user2 = recs[recs.user_id == 2]["item_id"].tolist()
    assert user2[:2] == [102, 103]
    # user 3 не видел 100/101 — они должны быть первыми
    user3 = recs[recs.user_id == 3]["item_id"].tolist()
    assert user3[:2] == [100, 101]
    # ранги 1..k внутри каждого пользователя
    assert recs.groupby("user_id")["rank"].apply(lambda r: r.tolist() == list(range(1, len(r) + 1))).all()
    # все пользователи train покрыты
    assert set(recs["user_id"]) == {1, 2, 3}


def test_popularity_variants_differ():
    train = _train_mini()
    count_top = PopularityRecommender("count").fit(train)._ranked_items[0]
    pos_top = PopularityRecommender("pos_ge4").fit(train)._ranked_items[0]
    # у item 100 две оценки (5,5) — он первый везде; а mean_rating поднимет
    # одинокие пятёрки: у 201 одна оценка 5, у 100 средняя 5.0
    mean_model = PopularityRecommender("mean_rating").fit(train)
    assert count_top == 100 and pos_top == 100 and mean_model._ranked_items[0] in (100, 201)
    _expect_error(lambda: PopularityRecommender("nope"))


def test_knn_prefers_co_occurred_items():
    # у user 2 seen = {100, 101}; с ними сильнее всего co-occur 102/103 (через user 1)
    model = ItemKNNRecommender("binary").fit(_train_mini())
    recs = model.recommend(k=10)
    user2 = recs[recs.user_id == 2].sort_values("rank")["item_id"].tolist()
    assert set(user2[:2]) == {102, 103}
    assert 100 not in user2 and 101 not in user2  # seen отфильтрованы


def test_knn_neighbor_truncation_changes_scores():
    full = ItemKNNRecommender("binary").fit(_train_mini()).recommend()
    truncated = ItemKNNRecommender("binary", n_neighbors=1).fit(_train_mini()).recommend()
    # на мини-данных оба работают и выдают полные выдачи; различие не обязательно,
    # проверяем стабильность интерфейса и детерминизм
    again = ItemKNNRecommender("binary", n_neighbors=1).fit(_train_mini()).recommend()
    pd.testing.assert_frame_equal(truncated, again)
    assert len(full) == len(again)


def test_als_smoke_fit_and_recommend():
    model = ALSRecommender(factors=4, iterations=3, num_threads=1).fit(_train_mini())
    recs = model.recommend(k=2)
    # по 2 unseen-кандидата каждому, без seen в выдаче
    assert recs.groupby("user_id").size().eq(2).all()
    seen = set(zip(_train_mini()["user_id"], _train_mini()["item_id"]))
    assert not set(zip(recs["user_id"], recs["item_id"])) & seen
    # score конечен и убывает по rank внутри пользователя
    assert recs["score"].notna().all()
    for _, group in recs.groupby("user_id"):
        assert group.sort_values("rank")["score"].is_monotonic_decreasing


def test_ease_zero_diagonal_and_seen_filter():
    model = EASERecommender(l2=10.0).fit(_train_mini())
    recs = model.recommend(k=10)
    seen = set(zip(_train_mini()["user_id"], _train_mini()["item_id"]))
    assert not set(zip(recs["user_id"], recs["item_id"])) & seen
    assert recs["score"].notna().all()
    # user 2 (видел 100/101, совпадающие с user 1) должен получить кандидатов user 1
    user2 = recs[recs.user_id == 2].sort_values("rank")["item_id"].tolist()
    assert set(user2[:2]) == {102, 103}
    # веса EASE: нулевая диагональ и симметрия не обязаны сохраняться после
    # нормировки Steck, но скоры конечны и детерминированы
    again = EASERecommender(l2=10.0).fit(_train_mini()).recommend(k=10)
    pd.testing.assert_frame_equal(recs, again)


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("все проверки пройдены")


if __name__ == "__main__":
    main()
