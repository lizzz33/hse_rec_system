"""Baseline-рекомендеры E01–E04: popularity, ItemKNN, ALS.

Общий протокол:

- ``fit(train)`` обучается только на train-событиях (никаких знаний о future);
- ``recommend(k)`` возвращает длинный DataFrame ``user_id, item_id, score, rank``
  по ``k`` рекомендациям на каждого пользователя train;
- уже оценённые пользователем items исключаются на этапе recommend: пары
  (user, item) не повторяются, такой item в принципе не может быть хитом,
  и фильтрация — ответственность модели, а не метрики (notes/plan.md).

SASRec (E05) живёт отдельно в ``scripts/e05_sasrec.py`` через RePlay —
его API принципиально другой (lightning + parquet sequences).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

POPULARITY_VARIANTS = ("count", "mean_rating", "bayes_m50", "pos_ge4", "pos_ge3")


def _user_item_indices(train: pd.DataFrame, users: np.ndarray, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Позиции строк train в dense сетке users × items (оба массива отсортированы)."""
    user_idx = np.searchsorted(users, train["user_id"].to_numpy())
    item_idx = np.searchsorted(items, train["item_id"].to_numpy())
    return user_idx, item_idx


def _seen_mask(user_idx: np.ndarray, item_idx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Булева матрица «пользователь уже оценил item» той же сетки."""
    return sp.csr_matrix((np.ones(len(user_idx), dtype=bool), (user_idx, item_idx)), shape=shape).toarray()


def _dense_top_n(
    scores: np.ndarray, seen: np.ndarray, users: np.ndarray, item_ids: np.ndarray, k: int
) -> pd.DataFrame:
    """Топ-k unseen items по плотной матрице score (n_users × n_items), без цикла по пользователям.

    Оценённые items получают -inf и не могут попасть в выдачу. Возвращает
    длинный DataFrame user_id, item_id, score, rank (rank с 1, по убыванию score).
    """
    scores = scores.astype(np.float32, copy=True)
    scores[seen] = -np.inf
    k = min(k, scores.shape[1])
    # argpartition по k-му сверху: последние k колонок каждой строки — кандидаты
    candidates = np.argpartition(scores, scores.shape[1] - k, axis=1)[:, -k:]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1, kind="stable")
    top_items = np.take_along_axis(candidates, order, axis=1)
    top_scores = np.take_along_axis(candidate_scores, order, axis=1)
    ranks = np.tile(np.arange(1, k + 1), (scores.shape[0], 1))
    frame = pd.DataFrame(
        {
            "user_id": np.repeat(users, k),
            "item_id": item_ids[top_items.ravel()],
            "score": top_scores.ravel(),
            "rank": ranks.ravel(),
        }
    )
    # при k >= n_items в кандидаты попадают и -inf (seen) слоты — выкидываем;
    # на реальных данных (3706 items, максимум 2056 оценок) это не срабатывает
    return frame[np.isfinite(frame["score"])].reset_index(drop=True)


class PopularityRecommender:
    """E01/E02: неперсонализированный топ-N по варианту популярности.

    Варианты:

    - ``count`` — число взаимодействий (E01);
    - ``mean_rating`` — средняя оценка (ожидаемо страдает от items с 1–2
      оценками — это и проверяем в E02);
    - ``bayes_m50`` — сглаживание средней оценки к глобальной средней с
      весом m=50 взаимодействий (IMDb-подобная взвешенная формула);
    - ``pos_ge4`` / ``pos_ge3`` — количество «позитивных» оценок.
    """

    def __init__(self, variant: str = "count"):
        if variant not in POPULARITY_VARIANTS:
            raise ValueError(f"неизвестный вариант popularity: {variant!r}; доступны {POPULARITY_VARIANTS}")
        self.variant = variant
        self._ranked_items: np.ndarray | None = None
        self._item_scores: pd.Series | None = None
        self._seen_by_user: dict | None = None

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        self._seen_by_user = train.groupby("user_id")["item_id"].agg(set).to_dict()
        if self.variant == "count":
            scores = train.groupby("item_id").size().astype(float)
        elif self.variant == "mean_rating":
            scores = train.groupby("item_id")["rating"].mean()
        elif self.variant == "bayes_m50":
            global_mean = train["rating"].mean()
            stats = train.groupby("item_id")["rating"].agg(["mean", "size"])
            m = 50.0
            scores = (stats["size"] * stats["mean"] + m * global_mean) / (stats["size"] + m)
        elif self.variant == "pos_ge4":
            scores = train.loc[train["rating"] >= 4].groupby("item_id").size().astype(float)
        else:  # pos_ge3
            scores = train.loc[train["rating"] >= 3].groupby("item_id").size().astype(float)
        self._item_scores = scores
        # сортировка: score по убыванию, при равенстве — item_id по возрастанию
        # (стабильные сабмиты и воспроизводимые сравнения вариантов)
        order = np.lexsort((scores.index.to_numpy(), -scores.to_numpy()))
        self._ranked_items = scores.index.to_numpy()[order]
        return self

    def recommend(self, k: int = 10) -> pd.DataFrame:
        if self._ranked_items is None or self._item_scores is None or self._seen_by_user is None:
            raise RuntimeError("сначала fit(train)")
        rows = []
        for user, seen in self._seen_by_user.items():
            top = [item for item in self._ranked_items if item not in seen][:k]
            rows.extend((user, item, self._item_scores[item], rank) for rank, item in enumerate(top, start=1))
        return pd.DataFrame(rows, columns=["user_id", "item_id", "score", "rank"])


class ItemKNNRecommender:
    """E03: item-to-item KNN по косинусной близости в пространстве пользователей.

    ``matrix`` задаёт веса user-item матрицы, из которой строится близость items:

    - ``binary`` — 1 за каждый interaction (классический implicit ItemKNN);
    - ``rating`` — вес = оценка 1..5 (rating-aware similarity);
    - ``centered`` — оценка минус средняя оценка пользователя (Pearson-стиль:
      сравниваются отклонения от личного центра, а не сырые шкалы).

    ``n_neighbors`` <= 0 — использовать всех соседей; иначе на каждый item
    остаются ``n_neighbors`` самых похожих (прореживание уменьшает «размытие»
    score через слабых соседей). Сам item из своих соседей исключён.
    """

    def __init__(self, matrix: Literal["binary", "rating", "centered"] = "binary", n_neighbors: int = 0):
        self.matrix = matrix
        self.n_neighbors = n_neighbors
        self._scores: np.ndarray | None = None
        self._users: np.ndarray | None = None
        self._items: np.ndarray | None = None
        self._seen: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "ItemKNNRecommender":
        users = np.sort(train["user_id"].unique())
        items = np.sort(train["item_id"].unique())
        user_idx, item_idx = _user_item_indices(train, users, items)
        weights = self._interaction_weights(train)

        shape = (len(users), len(items))
        mat = sp.csr_matrix((weights, (user_idx, item_idx)), shape=shape, dtype=np.float32)
        self._seen = _seen_mask(user_idx, item_idx, shape)

        # косинус: колонки-айтемы нормируются на L2-норму, затем Gram-матрица
        squared = mat.copy()
        squared.data = squared.data**2
        norm = np.sqrt(np.asarray(squared.sum(axis=0)).ravel())
        norm[norm == 0] = np.inf  # item без веса не должен портить нормировку
        normalized = mat @ sp.diags(1.0 / norm)
        similarity = (normalized.T @ normalized).toarray()
        np.fill_diagonal(similarity, 0.0)

        if self.n_neighbors > 0:
            neighbors = min(self.n_neighbors, len(items) - 1)
            threshold = np.partition(similarity, -neighbors, axis=1)[:, -neighbors]
            similarity[similarity < threshold[:, None]] = 0.0

        self._scores = np.asarray(mat @ similarity)
        self._users, self._items = users, items
        return self

    def _interaction_weights(self, train: pd.DataFrame) -> np.ndarray:
        ratings = train["rating"].to_numpy(dtype=np.float32)
        if self.matrix == "binary":
            return np.ones(len(train), dtype=np.float32)
        if self.matrix == "rating":
            return ratings
        user_mean = train.groupby("user_id")["rating"].transform("mean").to_numpy(dtype=np.float32)
        return ratings - user_mean

    def recommend(self, k: int = 10) -> pd.DataFrame:
        if self._scores is None:
            raise RuntimeError("сначала fit(train)")
        return _dense_top_n(self._scores, self._seen, self._users, self._items, k)


class ALSRecommender:
    """E04: implicit ALS (Hu-Koren-Volinsky) на confidence-матрице.

    ``weight`` задаёт значение ячейки матрицы до ``alpha``:

    - ``binary`` — 1 за interaction (iALS);
    - ``rating`` — вес = оценка 1..5 (rating-weighted feedback);
    - ``pos_ge4`` — только оценки 4–5 (низкие оценки удаляются целиком).

    ``alpha`` масштабирует confidence относительно регуляризации (классические
    значения для ML-1M-подобных данных — от 1 до 40).
    """

    def __init__(
        self,
        weight: Literal["binary", "rating", "pos_ge4"] = "binary",
        alpha: float = 1.0,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 20,
        num_threads: int = 4,
        random_state: int = 42,
    ):
        self.weight = weight
        self.alpha = alpha
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.num_threads = num_threads
        self.random_state = random_state
        self._scores: np.ndarray | None = None
        self._users: np.ndarray | None = None
        self._items: np.ndarray | None = None
        self._seen: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "ALSRecommender":
        from implicit.als import AlternatingLeastSquares

        users = np.sort(train["user_id"].unique())
        items = np.sort(train["item_id"].unique())
        user_idx, item_idx = _user_item_indices(train, users, items)

        cell = self._cell_values(train)
        keep = cell != 0  # pos_ge4 отбрасывает низкие оценки целиком
        shape = (len(users), len(items))
        mat = sp.csr_matrix((cell[keep], (user_idx[keep], item_idx[keep])), shape=shape, dtype=np.float32)
        self._seen = _seen_mask(user_idx, item_idx, shape)

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            num_threads=self.num_threads,
            random_state=self.random_state,
            calculate_training_loss=False,
        )
        model.fit(mat)
        scores = model.user_factors @ model.item_factors.T
        if scores.shape != shape:
            raise RuntimeError(f"неожиданная форма факторов {model.user_factors.shape}x{model.item_factors.shape} вместо {shape}")
        self._scores = scores.astype(np.float32)
        self._users, self._items = users, items
        return self

    def _cell_values(self, train: pd.DataFrame) -> np.ndarray:
        ratings = train["rating"].to_numpy(dtype=np.float32)
        if self.weight == "binary":
            return np.ones(len(train), dtype=np.float32)
        if self.weight == "rating":
            return ratings
        return ratings >= 4  # pos_ge4: bool, нули отбрасываются в fit

    def recommend(self, k: int = 10) -> pd.DataFrame:
        if self._scores is None:
            raise RuntimeError("сначала fit(train)")
        return _dense_top_n(self._scores, self._seen, self._users, self._items, k)
