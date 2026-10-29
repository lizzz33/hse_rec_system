"""E03: ItemKNN — косинусная item-item близость, варианты весов и соседей.

Гипотезы внутри эксперимента:

- персонализация через co-occurrence сильно бьёт popularity (H3, notes/data.md);
- учёт оценок (rating / centered) меняет близость и может улучшить recall,
  если низкие оценки действительно выражают антипатии;
- прореживание соседей до top-200 убирает шум слабых связей.

Лучший на 9010 вариант собирается в сабмит на полных events.

Запуск:
    uv run python scripts/e03_knn.py
    uv run python scripts/e03_knn.py --skip-submission
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recsys.data import load_events
from recsys.experiments import eval_variants, save_results
from recsys.metrics import save_submission
from recsys.models import ItemKNNRecommender

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-submission", action="store_true", help="только локальная валидация")
    args = parser.parse_args()

    variants = {
        "binary_cosine": lambda train: ItemKNNRecommender("binary").fit(train).recommend(),
        "binary_cosine_top200": lambda train: ItemKNNRecommender("binary", n_neighbors=200).fit(train).recommend(),
        "rating_cosine": lambda train: ItemKNNRecommender("rating").fit(train).recommend(),
        "centered_cosine_top200": lambda train: ItemKNNRecommender("centered", n_neighbors=200).fit(train).recommend(),
    }
    results = eval_variants(variants)
    path = save_results(results, "e03_knn")
    print(f"\nрезультаты: {path}")

    if args.skip_submission:
        return
    split_9010 = results["results"]["9010"]
    best = max(split_9010, key=lambda name: split_9010[name]["recall@10_macro_relevant"])
    config = {
        "binary_cosine": dict(matrix="binary"),
        "binary_cosine_top200": dict(matrix="binary", n_neighbors=200),
        "rating_cosine": dict(matrix="rating"),
        "centered_cosine_top200": dict(matrix="centered", n_neighbors=200),
    }[best]
    events = load_events()
    recs = ItemKNNRecommender(**config).fit(events).recommend()
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out = SUBMISSIONS_DIR / f"E03_knn_{best}.csv"
    save_submission(recs, out)
    print(f"сабмит E03 (лучший на 9010 — {best}): {out}")


if __name__ == "__main__":
    main()
