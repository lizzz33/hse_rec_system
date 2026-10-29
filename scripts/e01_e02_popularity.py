"""E01+E02: глобальная и rating-aware популярность на всех локальных сплитах.

E01 — топ-10 популярных фильмов по числу взаимодействий (нижняя содержательная
ступень лестницы). E02 проверяет, улучшает ли popularity учёт оценок:
средняя, байесовское сглаживание, позитивные пороги.

По лучшему на основном сплите (9010) варианту E02 собирается сабмит
на полных events (для скрытого теста всё, что нам выдано, — прошлое).

Запуск:
    uv run python scripts/e01_e02_popularity.py
    uv run python scripts/e01_e02_popularity.py --skip-submission
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recsys.data import load_events
from recsys.experiments import eval_variants, save_results
from recsys.metrics import save_submission
from recsys.models import PopularityRecommender

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-submission", action="store_true", help="только локальная валидация")
    args = parser.parse_args()

    variants = {
        "E01_count": lambda train: PopularityRecommender("count").fit(train).recommend(),
        "E02_mean_rating": lambda train: PopularityRecommender("mean_rating").fit(train).recommend(),
        "E02_bayes_m50": lambda train: PopularityRecommender("bayes_m50").fit(train).recommend(),
        "E02_pos_ge4": lambda train: PopularityRecommender("pos_ge4").fit(train).recommend(),
        "E02_pos_ge3": lambda train: PopularityRecommender("pos_ge3").fit(train).recommend(),
    }
    results = eval_variants(variants)
    path = save_results(results, "e01_e02_popularity")
    print(f"\nрезультаты: {path}")

    if args.skip_submission:
        return
    events = load_events()
    # E01 сабмитим всегда (ступень лестницы и калибровка метрики),
    # E02 — лучший из rating-aware вариантов на основном сплите 9010
    split_9010 = results["results"]["9010"]
    best_e02 = max(
        (name for name in split_9010 if name.startswith("E02_")),
        key=lambda name: split_9010[name]["recall@10_macro_relevant"],
    )
    e01_recs = PopularityRecommender("count").fit(events).recommend()
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    e01_path = SUBMISSIONS_DIR / "E01_popularity_count.csv"
    save_submission(e01_recs, e01_path)
    print(f"сабмит E01: {e01_path}")
    variant = best_e02.split("E02_", 1)[1]
    e02_recs = PopularityRecommender(variant).fit(events).recommend()
    e02_path = SUBMISSIONS_DIR / f"E02_popularity_{variant}.csv"
    save_submission(e02_recs, e02_path)
    print(f"сабмит E02 (лучший на 9010 — {best_e02}): {e02_path}")


if __name__ == "__main__":
    main()
