"""E04b: усиление non-transformer ветки — тюнинг ALS + EASE.

Мотивация: лучший non-transformer сабмит (E04 ALS) дал 0.0783 — ниже
baseline_5 (0.0821). Преподаватель допускал отдельное требование решить
задачу без трансформера, поэтому ветка обязана пробить baseline_5 сама.

Гипотезы:
- большие факторы (128/256) и другая регуляризация ALS поднимут 9010
  с 0.0658 ближе к 0.08+ (Kaggle ×1.2 по калибровке);
- EASE (закрытая форма item-item с регуляризацией λ) обгонит ItemKNN
  и, возможно, ALS на 9010 — у него обычно высокий recall на ML-1M.

Выбор — по 9010 recall@10_macro_relevant (калибровка 29.10); masked10
считаем рядом как диагностику. Лучший вариант собирается в сабмит
на полных events.

Запуск:
    uv run python scripts/e04b_als_ease.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recsys.data import load_events
from recsys.experiments import eval_variants, save_results
from recsys.metrics import save_submission
from recsys.models import ALSRecommender, EASERecommender

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-submission", action="store_true")
    args = parser.parse_args()

    variants = {
        "als_f128": lambda train: ALSRecommender("rating", alpha=1, factors=128, regularization=0.05).fit(train).recommend(),
        "als_f256": lambda train: ALSRecommender("rating", alpha=1, factors=256, regularization=0.05).fit(train).recommend(),
        "als_f128_r01": lambda train: ALSRecommender("rating", alpha=1, factors=128, regularization=0.01).fit(train).recommend(),
        "als_f256_r01": lambda train: ALSRecommender("rating", alpha=1, factors=256, regularization=0.01).fit(train).recommend(),
        "ease_l10": lambda train: EASERecommender(l2=10.0).fit(train).recommend(),
        "ease_l50": lambda train: EASERecommender(l2=50.0).fit(train).recommend(),
        "ease_l200": lambda train: EASERecommender(l2=200.0).fit(train).recommend(),
    }
    results = eval_variants(variants, splits=("9010", "masked10"))
    path = save_results(results, "e04b_als_ease")
    print(f"\nрезультаты: {path}")

    if args.skip_submission:
        return
    split_9010 = results["results"]["9010"]
    best = max(split_9010, key=lambda name: split_9010[name]["recall@10_macro_relevant"])
    config = {
        "als_f128": lambda: ALSRecommender("rating", alpha=1, factors=128, regularization=0.05),
        "als_f256": lambda: ALSRecommender("rating", alpha=1, factors=256, regularization=0.05),
        "als_f128_r01": lambda: ALSRecommender("rating", alpha=1, factors=128, regularization=0.01),
        "als_f256_r01": lambda: ALSRecommender("rating", alpha=1, factors=256, regularization=0.01),
        "ease_l10": lambda: EASERecommender(l2=10.0),
        "ease_l50": lambda: EASERecommender(l2=50.0),
        "ease_l200": lambda: EASERecommender(l2=200.0),
    }[best]
    events = load_events()
    recs = config().fit(events).recommend()
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out = SUBMISSIONS_DIR / f"E04b_{best}.csv"
    save_submission(recs, out)
    print(f"сабмит E04b (лучший на 9010 — {best}): {out}")


if __name__ == "__main__":
    main()
