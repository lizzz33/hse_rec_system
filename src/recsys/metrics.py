"""Локальная метрика Recall@10 и диагностика рекомендаций.

На страницах соревнования формула не опубликована (только «Метрика: recall@10»),
поэтому ``evaluate`` считает все стандартные варианты рядом. Основным до сверки
считаем макро-recall с делителем |ground truth| пользователя. Формулу Kaggle
выявляем сравнением панели вариантов со скором первых сабмитов: совпадение до
4 знаков после запятой на двух разных сабмитах однозначно определяет вариант.

Соглашения (notes/plan.md, раздел «Локальная метрика»):

- порядок строк внутри пользователя в ``recs`` = порядок ранжирования,
  рекомендациями считаются первые ``k`` строк (при наличии колонки ``rank``
  строки сортируются по ней);
- повторяющиеся items внутри топ-k учитываются один раз — дубликат «сжигает»
  слот и штрафуется метрикой автоматически;
- пользователи без ground truth исключаются из метрики (macro и micro);
- пользователи с ground truth, но без рекомендаций, получают recall = 0;
- уже оценённые пользователем items не могут быть хитами (в events.csv нет
  повторных пар user-item), поэтому метрика штрафует их автоматически;
  фильтровать «просмотренное» — решение модели, а не метрики;
- релевантным считается любое взаимодействие holdout независимо от rating
  (гипотеза: скрытый тест — замаскированные взаимодействия, notes/data.md,
  раздел 7); при необходимости фильтровать ground truth по оценке —
  параметр ``min_rating``.
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

Denominator = Literal["relevant", "capped", "k"]
_RECALL_DENOMINATORS: tuple[Denominator, ...] = ("relevant", "capped", "k")


def _recs_to_mapping(
    recs: pd.DataFrame | Mapping,
) -> dict:
    """Приводит рекомендации к виду ``{user_id: [item_id, ...]}``.

    Порядок items внутри пользователя = порядок рекомендаций.
    """
    if isinstance(recs, pd.DataFrame):
        if "rank" in recs.columns:
            recs = recs.sort_values(["user_id", "rank"], kind="stable")
        mapping: dict = {}
        for user_id, item_id in zip(
            recs["user_id"].to_numpy(), recs["item_id"].to_numpy()
        ):
            mapping.setdefault(user_id, []).append(item_id)
        return mapping
    if isinstance(recs, Mapping):
        return {user: list(items) for user, items in recs.items()}
    raise TypeError(f"recs: ожидается DataFrame или dict, получен {type(recs)}")


def _gt_to_mapping(ground_truth: pd.DataFrame | Mapping) -> dict:
    """Приводит ground truth к виду ``{user_id: {item_id, ...}}``."""
    if isinstance(ground_truth, pd.DataFrame):
        mapping: dict = {}
        for user_id, item_id in zip(
            ground_truth["user_id"].to_numpy(),
            ground_truth["item_id"].to_numpy(),
        ):
            mapping.setdefault(user_id, set()).add(item_id)
        return mapping
    if isinstance(ground_truth, Mapping):
        return {user: set(items) for user, items in ground_truth.items()}
    raise TypeError(
        f"ground_truth: ожидается DataFrame или dict, получен {type(ground_truth)}"
    )


def _denominator(n_relevant: int, k: int, denominator: Denominator) -> int:
    if denominator == "relevant":
        return n_relevant
    if denominator == "capped":
        return min(k, n_relevant)
    return k


def _user_hits(
    recs: Mapping, ground_truth: Mapping, k: int
) -> dict:
    """``{user_id: (hits, n_relevant)}`` для пользователей с ground truth."""
    hits = {}
    for user, relevant in ground_truth.items():
        top = recs.get(user, ())[:k]
        hits[user] = (len(set(top) & relevant), len(relevant))
    return hits


def recall_at_k(
    recs: pd.DataFrame | Mapping,
    ground_truth: pd.DataFrame | Mapping,
    k: int = 10,
    denominator: Denominator = "relevant",
    average: Literal["macro", "micro"] = "macro",
) -> float:
    """Recall@k.

    Варианты делителя:

    - ``relevant`` — |ground truth| пользователя (основной);
    - ``capped`` — min(k, |ground truth|);
    - ``k`` — каждый хит стоит 1/k.

    ``average="macro"`` — среднее по пользователям с ground truth,
    ``average="micro"`` — сумма хитов / сумма релевантных.
    """
    recs_m = _recs_to_mapping(recs)
    gt_m = _gt_to_mapping(ground_truth)
    hits = _user_hits(recs_m, gt_m, k)
    if not hits:
        return 0.0
    if average == "macro":
        per_user = [
            h / _denominator(n_rel, k, denominator) for h, n_rel in hits.values()
        ]
        return float(np.mean(per_user))
    if average == "micro":
        total_relevant = sum(n_rel for _, n_rel in hits.values())
        if total_relevant == 0:
            return 0.0
        return sum(h for h, _ in hits.values()) / total_relevant
    raise ValueError(f"average: ожидается macro или micro, получен {average!r}")


def evaluate(
    recs: pd.DataFrame | Mapping,
    ground_truth: pd.DataFrame | Mapping,
    k: int = 10,
    train_events: pd.DataFrame | None = None,
    min_rating: int | None = None,
) -> pd.Series:
    """Панель всех вариантов Recall@k + диагностика рекомендаций.

    ``train_events`` — train-события (user_id, item_id[, rating]); нужны только
    для диагностики доли «уже оценённых» items в топ-k. ``min_rating`` при
    не-None фильтрует ground truth по rating (ground_truth тогда обязан
    содержать колонку rating).
    """
    if min_rating is not None:
        if not isinstance(ground_truth, pd.DataFrame) or "rating" not in ground_truth.columns:
            raise ValueError("min_rating требует ground_truth DataFrame с колонкой rating")
        ground_truth = ground_truth[ground_truth["rating"] >= min_rating]
    if isinstance(ground_truth, pd.DataFrame):
        ground_truth = ground_truth[["user_id", "item_id"]]

    recs_m = _recs_to_mapping(recs)
    gt_m = _gt_to_mapping(ground_truth)
    hits = _user_hits(recs_m, gt_m, k)

    metrics: dict[str, float] = {}
    for denominator in _RECALL_DENOMINATORS:
        metrics[f"recall@{k}_macro_{denominator}"] = recall_at_k(
            recs_m, gt_m, k, denominator=denominator
        )
    metrics[f"recall@{k}_micro"] = recall_at_k(recs_m, gt_m, k, average="micro")

    gt_sizes = [n_rel for _, n_rel in hits.values()]
    metrics["n_eval_users"] = len(hits)
    metrics["users_without_recs"] = sum(
        1 for user in gt_m if not recs_m.get(user)
    )
    metrics["gt_per_user_mean"] = float(np.mean(gt_sizes))
    metrics["gt_per_user_median"] = float(np.median(gt_sizes))

    top_all = [recs_m.get(user, ())[:k] for user in gt_m]
    n_top_rows = sum(len(top) for top in top_all)
    n_dup = sum(len(top) - len(set(top)) for top in top_all)
    metrics["dup_in_top_share"] = n_dup / n_top_rows if n_top_rows else 0.0
    metrics["n_unique_rec_items"] = len({item for top in top_all for item in top})

    if train_events is not None:
        seen = set(
            zip(train_events["user_id"].to_numpy(), train_events["item_id"].to_numpy())
        )
        # доля слотов топ-k, занятых уже оценёнными в train items
        n_seen_slots = sum(
            sum((user, item) in seen for item in top)
            for user, top in zip(gt_m, top_all)
        )
        metrics["seen_in_top_share"] = (
            n_seen_slots / n_top_rows if n_top_rows else 0.0
        )
    return pd.Series(metrics, name="value")


def load_submission(path: str) -> dict:
    """Читает сабмит в формате submission_sample.csv: ``user_id,item_id``,
    где item_id — 10 айтемов через пробел. Возвращает ``{user_id: [item_id, ...]}``
    с int-айдишниками (item_id читается как int, как в events.csv).
    """
    df = pd.read_csv(path, dtype=str)
    return {
        int(row.user_id): [int(item) for item in row.item_id.split()]
        for row in df.itertuples(index=False)
    }


def save_submission(
    recs: pd.DataFrame | Mapping, path: str, k: int = 10
) -> None:
    """Пишет сабмит в формате submission_sample.csv.

    Для каждого пользователя берёт первые k уникальных items (дубликаты
    выбрасываются, слоты добираются из хвоста списка). Нарушение инвариантов
    E00 — меньше k уникальных items у пользователя — это ошибка.
    """
    mapping = _recs_to_mapping(recs)
    rows = []
    for user, items in mapping.items():
        top = list(dict.fromkeys(items))[:k]
        if len(top) < k:
            raise ValueError(
                f"user {user}: только {len(top)} уникальных items из {k} required"
            )
        rows.append((user, " ".join(str(item) for item in top)))
    pd.DataFrame(rows, columns=["user_id", "item_id"]).sort_values(
        "user_id"
    ).to_csv(path, index=False)
