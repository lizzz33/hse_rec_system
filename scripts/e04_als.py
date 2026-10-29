"""E04: implicit ALS — варианты бинаризации и confidence-весов.

Гипотезы внутри эксперимента:

- латентные факторы находят предпочтения, которых не видит item co-occurrence;
- confidence-веса из оценок (rating 1..5) лучше бинарных, если шкала несёт сигнал;
- positives-only (rating >= 4) лучше, если низкие оценки — антипатии;
- alpha > 1 усиливает вклад наблюдений против регуляризации (классика iALS).

Лучший на 9010 вариант собирается в сабмит на полных events.

Запуск:
    uv run python scripts/e04_als.py
    uv run python scripts/e04_als.py --skip-submission
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recsys.data import load_events
from recsys.experiments import eval_variants, save_results
from recsys.metrics import save_submission
from recsys.models import ALSRecommender

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-submission", action="store_true", help="только локальная валидация")
    args = parser.parse_args()

    variants = {
        "als_binary": lambda train: ALSRecommender("binary", alpha=1).fit(train).recommend(),
        "als_binary_a10": lambda train: ALSRecommender("binary", alpha=10).fit(train).recommend(),
        "als_rating": lambda train: ALSRecommender("rating", alpha=1).fit(train).recommend(),
        "als_rating_a10": lambda train: ALSRecommender("rating", alpha=10).fit(train).recommend(),
        "als_pos_ge4": lambda train: ALSRecommender("pos_ge4", alpha=10).fit(train).recommend(),
    }
    results = eval_variants(variants)
    path = save_results(results, "e04_als")
    print(f"\nрезультаты: {path}")

    if args.skip_submission:
        return
    split_9010 = results["results"]["9010"]
    best = max(split_9010, key=lambda name: split_9010[name]["recall@10_macro_relevant"])
    config = {
        "als_binary": dict(weight="binary", alpha=1.0),
        "als_binary_a10": dict(weight="binary", alpha=10.0),
        "als_rating": dict(weight="rating", alpha=1.0),
        "als_rating_a10": dict(weight="rating", alpha=10.0),
        "als_pos_ge4": dict(weight="pos_ge4", alpha=10.0),
    }[best]
    events = load_events()
    recs = ALSRecommender(**config).fit(events).recommend()
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out = SUBMISSIONS_DIR / f"E04_{best}.csv"
    save_submission(recs, out)
    print(f"сабмит E04 (лучший на 9010 — {best}): {out}")


if __name__ == "__main__":
    main()
