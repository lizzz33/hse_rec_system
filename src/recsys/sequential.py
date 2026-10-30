"""Sequential-скоринг классических моделей: марковские переходы и контекст последних K событий (E09).

Реконструкция протокола скрытого теста (notes/data.md, раздел 7) показала, что
оцениваются последние 1–3 события каждого пользователя — задача next-item.
Рекомендеры E03/E04/E04b скорят пользователя по всей истории, что размывает
next-item сигнал. Здесь строятся те же item-item матрицы (косинусная близость,
веса EASE, переходы Маркова 1-го порядка), но скор пользователя считается по
последним K событиям его train-истории:

    score(u, ·) = Σ_{k=1..K} decay^(k-1) · W[last_k(u), ·]

где last_1(u) — самое свежее событие. Формула веса (decay^k против
decay^(k-1)) не меняет ранжирование — это общий множитель строки.

Обучение — только на train-событиях протокола; seen-фильтр — по всей
train-истории пользователя. Топ-k считается через ``models._dense_top_n``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.models import _dense_top_n, _seen_mask


def sorted_histories(train: pd.DataFrame) -> dict[int, np.ndarray]:
    """``{user_id: item_id по возрастанию timestamp}`` — контекст скоринга."""
    ordered = train.sort_values(["user_id", "timestamp"], kind="stable")
    return {
        user: items.to_numpy()
        for user, items in ordered.groupby("user_id", sort=False)["item_id"]
    }


def last_context(
    histories: dict[int, np.ndarray], items: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Последние ``k`` событий каждого пользователя, самые свежие — первыми.

    Возвращает ``(users, context)``: users — отсортированные user_id,
    context — (n_users × k) позиции в ``items``; короткие истории дополнены
    значением ``-1`` (события нет — строка W не прибавляется).
    """
    users = np.sort(np.fromiter(histories.keys(), dtype=np.int64))
    context = np.full((len(users), k), -1, dtype=np.int64)
    for row, user in enumerate(users):
        tail = histories[user][-k:][::-1]
        context[row, : len(tail)] = np.searchsorted(items, tail)
    return users, context


def markov_weights(
    train: pd.DataFrame, items: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """Матрица переходов T[i, j] = P(j сразу после i) по train-историям.

    Строки нормируются на сумму переходов из i; ``alpha`` — аддитивное
    сглаживание к равномерному распределению каталога (строки предметов
    с единственным переходом не вырождаются в one-hot).
    """
    ordered = train.sort_values(["user_id", "timestamp"], kind="stable")
    user = ordered["user_id"].to_numpy()
    item = ordered["item_id"].to_numpy()
    same_user = user[1:] == user[:-1]
    from_idx = np.searchsorted(items, item[:-1][same_user])
    to_idx = np.searchsorted(items, item[1:][same_user])
    counts = sp.coo_matrix(
        (np.ones(len(from_idx)), (from_idx, to_idx)), shape=(len(items), len(items))
    ).tocsr()
    smoothed = np.asarray(counts.todense(), dtype=np.float64)
    smoothed += alpha
    row_total = smoothed.sum(axis=1)
    normalized = np.zeros_like(smoothed)
    has_transitions = row_total > 0
    normalized[has_transitions] = (
        smoothed[has_transitions] / row_total[has_transitions, None]
    )
    # item, ни разу не встретившийся в левой части пары (например, только
    # последние события историй), при alpha=0 даёт 0/0 — backoff к равномерному
    normalized[~has_transitions] = 1.0 / len(items)
    return normalized.astype(np.float32)


def cosine_weights(train: pd.DataFrame, items: np.ndarray) -> np.ndarray:
    """Косинусная близость колонок binary user-item матрицы (item-item KNN).

    Та же близость, что в ``models.ItemKNNRecommender("binary")``; диагональ
    зануляется (item не похож сам на себя).
    """
    users = np.sort(train["user_id"].unique())
    user_idx = np.searchsorted(users, train["user_id"].to_numpy())
    item_idx = np.searchsorted(items, train["item_id"].to_numpy())
    mat = sp.csr_matrix(
        (np.ones(len(train), dtype=np.float32), (user_idx, item_idx)),
        shape=(len(users), len(items)),
    )
    squared = mat.copy()
    squared.data = squared.data**2
    norm = np.sqrt(np.asarray(squared.sum(axis=0)).ravel())
    norm[norm == 0] = np.inf
    normalized = mat @ sp.diags(1.0 / norm)
    similarity = (normalized.T @ normalized).toarray().astype(np.float32)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def ease_weights(train: pd.DataFrame, items: np.ndarray, l2: float) -> np.ndarray:
    """Веса EASE (Steck 2019): B = I − inv(G + λI) / diag — та же закрытая
    форма, что в ``models.EASERecommender``, но матрица, а не user-скоры."""
    users = np.sort(train["user_id"].unique())
    user_idx = np.searchsorted(users, train["user_id"].to_numpy())
    item_idx = np.searchsorted(items, train["item_id"].to_numpy())
    binary = sp.csr_matrix(
        (np.ones(len(train), dtype=np.float32), (user_idx, item_idx)),
        shape=(len(users), len(items)),
    )
    gram = (binary.T @ binary).toarray().astype(np.float64)
    diag = np.diag_indices_from(gram)
    gram[diag] += l2
    inverse = np.linalg.inv(gram)
    weights = -inverse / np.diag(inverse)[:, None]
    weights[diag] = 0.0
    return weights.astype(np.float32)


def popularity_vector(train: pd.DataFrame, items: np.ndarray) -> np.ndarray:
    """Нормированная частота items в train — unigram для markov-backoff."""
    counts = np.bincount(
        np.searchsorted(items, train["item_id"].to_numpy()), minlength=len(items)
    ).astype(np.float64)
    return (counts / counts.sum()).astype(np.float32)


def context_score_rows(
    context: np.ndarray, weights: np.ndarray
) -> list[np.ndarray]:
    """``S_k = W[last_k]`` для каждой позиции контекста k = 1..K.

    Отсутствующие события (позиция -1) дают нулевую строку. Скор при
    ``decay`` и глубине ``K`` — линейная комбинация ``Σ decay^(k-1) · S_k``,
    поэтому сетки (K, decay) не требуют пересчёта строк.
    """
    rows = []
    for k in range(context.shape[1]):
        score = np.zeros((context.shape[0], weights.shape[1]), dtype=np.float32)
        valid = context[:, k] >= 0
        score[valid] = weights[context[valid, k]]
        rows.append(score)
    return rows


def combine_rows(
    rows: list[np.ndarray], depth: int, decay: float
) -> np.ndarray:
    """Σ_{k=1..depth} decay^(k-1) · S_k — скор по последним ``depth`` событиям."""
    scores = np.zeros_like(rows[0])
    for k in range(depth):
        scores += (decay**k) * rows[k]
    return scores


def topk_from_scores(
    scores: np.ndarray,
    users: np.ndarray,
    items: np.ndarray,
    train: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Топ-k unseen items по матрице скоров (n_users × n_items)."""
    user_idx = np.searchsorted(users, train["user_id"].to_numpy())
    item_idx = np.searchsorted(items, train["item_id"].to_numpy())
    seen = _seen_mask(user_idx, item_idx, scores.shape)
    return _dense_top_n(scores, seen, users, items, k)
