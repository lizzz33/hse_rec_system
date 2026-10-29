"""E01: бейзлайн «глобальная популярность» + измерение локальной метрики.

Сквозная проверка связки split -> рекомендации -> evaluate:
топ-10 популярных по train фильмов каждому пользователю (без уже оценённых),
панель всех вариантов Recall@10 на chronological 90/10.

Запуск:
    uv run python scripts/e01_popularity.py
    uv run python scripts/e01_popularity.py --split last10
    uv run python scripts/e01_popularity.py --submission data/submission_e01.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from recsys.data import load_events
from recsys.metrics import evaluate, save_submission
from recsys.split import chronological_split, last_n_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def popularity_recommendations(train: pd.DataFrame, k: int = 10) -> dict:
    """Топ-k популярных items по train каждому пользователю, исключая оценённые.

    Исключение «просмотренного» — решение модели, а не метрики: такой item
    в принципе не может быть хитом, слот тратится впустую.
    """
    popular_items = train["item_id"].value_counts().index.to_numpy()
    seen = train.groupby("user_id")["item_id"].agg(set).to_dict()
    return {
        user: [item for item in popular_items if item not in seen_set][:k]
        for user, seen_set in seen.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["9010", "last10"], default="9010")
    parser.add_argument(
        "--submission",
        type=str,
        default=None,
        help="дополнительно записать сабмит по полным events (топ по всем данным)",
    )
    args = parser.parse_args()

    events = load_events(DATA_DIR)
    if args.split == "last10":
        train, valid = last_n_split(events, n=10)
    else:
        train, valid = chronological_split(events, train_fraction=0.9)
    print(f"split={args.split}: train {len(train):,} / valid {len(valid):,} "
          f"/ пользователей в valid {valid['user_id'].nunique():,}")

    recs = popularity_recommendations(train)
    panel = evaluate(recs, valid, k=10, train_events=train)
    print()
    print(panel.to_string())

    if args.submission:
        save_submission(popularity_recommendations(events), args.submission)
        print(f"\nсабмит по полным events записан: {args.submission}")


if __name__ == "__main__":
    main()
