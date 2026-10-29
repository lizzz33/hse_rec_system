"""E06: candidate union и score/rank blending лучших моделей E03–E05.

Гипотезы:

- модели находят пересекающиеся, но не идентичные релевантные items:
  Candidate Recall@N у union выше, чем у любой модели по отдельности;
- блендинг скоров/рангов поднимает итоговый Recall@10 над лучшей одиночной
  моделью, если сигналы комплементарны.

Кандидатные генераторы (конфиги — лучшие на 9010 из E03–E05):

- ALS, rating-веса (alpha=1);
- ItemKNN, binary cosine;
- SASRec (RePlay, конфиг E05; обучается заново на train сплита).

Метрики: Candidate Recall@N (N=10,25,50,100,200) по моделям и union'ам;
Recall@10 панель для rank- и score-блендинга.

Запуск:
    uv run python scripts/e06_candidates.py --models als knn sasrec
    uv run python scripts/e06_candidates.py --models als knn            # без SASRec
    uv run python scripts/e06_candidates.py --models als knn sasrec --submission
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recsys.data import load_events
from recsys.experiments import RESULTS_DIR, SPLITTERS
from recsys.metrics import evaluate, recall_at_k, save_submission
from recsys.models import ALSRecommender, ItemKNNRecommender

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
N_CANDIDATES = 200

# конфиги лучших вариантов E03/E04 (зафиксированы в notes/hypotheses.md)
CLASSIC_FACTORIES = {
    "als": lambda: ALSRecommender(weight="rating", alpha=1.0),
    "knn": lambda: ItemKNNRecommender(matrix="binary"),
}


def get_sasrec_module():
    """Скрипт E05 переиспользуется как модуль: fit_predict_sasrec + фильтрация seen."""
    spec = importlib.util.spec_from_file_location("e05_sasrec", PROJECT_ROOT / "scripts" / "e05_sasrec.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidates_from_classic(name: str, train) -> dict:
    """{user_id: [item_id, ...]} — топ-N_CANDIDATES unseen items модели."""
    recs = CLASSIC_FACTORIES[name]().fit(train).recommend(k=N_CANDIDATES)
    return recs.groupby("user_id")["item_id"].apply(list).to_dict()


def candidates_from_sasrec(sasrec, train, valid, catalog_size, work_dir) -> dict:
    """Скоры SASRec по всему каталогу -> топ-N_CANDIDATES unseen items."""
    scores = sasrec.fit_predict_sasrec(train, valid, catalog_size, work_dir)
    recs = sasrec.filter_seen_topk(scores, train, k=N_CANDIDATES)
    return recs.groupby("user_id")["item_id"].apply(list).to_dict()


def round_robin_union(lists: list[dict]) -> dict:
    """Union кандидатов: round-robin по ранкам моделей (без учёта величин скоров).

    Порядок «модель1-1я, модель2-1я, ...» сохраняет сильные позиции каждой
    модели и не требует сопоставимости скоров между моделями.
    """
    users = set().union(*[set(lst) for lst in lists])
    union = {}
    for user in users:
        seen: set = set()
        merged: list = []
        for i in range(max((len(lst.get(user, [])) for lst in lists), default=0)):
            for lst in lists:
                if i < len(lst[user]):
                    item = lst[user][i]
                    if item not in seen:
                        seen.add(item)
                        merged.append(item)
        union[user] = merged
    return union


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["als", "knn", "sasrec"],
                        choices=["als", "knn", "sasrec"])
    parser.add_argument("--splits", nargs="+", default=["9010", "masked10"],
                        choices=sorted(SPLITTERS))
    parser.add_argument("--submission", action="store_true",
                        help="собрать сабмит лучшего бленда на полных events")
    args = parser.parse_args()

    events = load_events()
    sasrec = get_sasrec_module() if "sasrec" in args.models else None
    catalog_size = sasrec.item_catalog(events) if sasrec else None

    payload: dict = {"splits": {}, "results": {}}
    for split_name in args.splits:
        train, valid = SPLITTERS[split_name](events)
        payload["splits"][split_name] = {"train": len(train), "valid": len(valid)}
        print(f"\n=== {split_name} ===", flush=True)

        candidates: dict[str, dict] = {}
        for name in args.models:
            if name == "sasrec":
                candidates["sasrec"] = candidates_from_sasrec(
                    sasrec, train, valid, catalog_size, RESULTS_DIR / "e06_sasrec_work" / split_name
                )
            else:
                candidates[name] = candidates_from_classic(name, train)
            cr = {f"cand_recall@{n}": recall_at_k(candidates[name], valid, k=n)
                  for n in (10, 25, 50, 100, 200)}
            print(f"[{split_name}] {name}: " + " ".join(f"{k}={v:.4f}" for k, v in cr.items()), flush=True)
            payload["results"].setdefault(split_name, {})[f"cand_{name}"] = {k: round(v, 6) for k, v in cr.items()}

        # union'ы всех вложенных комбинаций из >=2 моделей
        names = list(candidates)
        from itertools import combinations
        for r in range(2, len(names) + 1):
            for combo in combinations(names, r):
                union = round_robin_union([candidates[n] for n in combo])
                cr = {f"cand_recall@{n}": recall_at_k(union, valid, k=n) for n in (10, 25, 50, 100, 200)}
                label = "cand_union_" + "_".join(combo)
                print(f"[{split_name}] {label}: " + " ".join(f"{k}={v:.4f}" for k, v in cr.items()), flush=True)
                payload["results"][split_name][label] = {k: round(v, 6) for k, v in cr.items()}

        # rank-блендинг: среднее нормированных рангов (1 - (rank-1)/len) на union
        union_all = round_robin_union(list(candidates.values()))
        rank_scores = {}
        for user, items_list in union_all.items():
            scores = {item: 0.0 for item in items_list}
            for name in names:
                for pos, item in enumerate(candidates[name].get(user, [])):
                    if item in scores:
                        scores[item] += 1.0 - pos / N_CANDIDATES
            rank_scores[user] = sorted(scores, key=scores.get, reverse=True)
        panel = evaluate({u: items[:10] for u, items in rank_scores.items()}, valid, k=10, train_events=train)
        payload["results"][split_name]["blend_rank_top10"] = {str(k): round(float(v), 6) for k, v in panel.items()}
        print(f"[{split_name}] blend_rank_top10: recall@10_macro_relevant="
              f"{panel['recall@10_macro_relevant']:.4f}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "e06_candidates.json"
    if out.is_file():
        existing = json.loads(out.read_text())
        existing["splits"].update(payload["splits"])
        existing["results"].update(payload["results"])
        payload = existing
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nрезультаты: {out}")

    if args.submission:
        print("\n=== сабмит на полных events: rank-blend " + "+".join(args.models) + " ===", flush=True)
        candidates = {}
        for name in args.models:
            if name == "sasrec":
                candidates["sasrec"] = candidates_from_sasrec(
                    sasrec, events, None, catalog_size, RESULTS_DIR / "e06_sasrec_work" / "full"
                )
            else:
                candidates[name] = candidates_from_classic(name, events)
        union_all = round_robin_union(list(candidates.values()))
        recs = {}
        for user, items_list in union_all.items():
            scores = {item: 0.0 for item in items_list}
            for name in args.models:
                for pos, item in enumerate(candidates[name].get(user, [])):
                    if item in scores:
                        scores[item] += 1.0 - pos / N_CANDIDATES
            recs[user] = sorted(scores, key=scores.get, reverse=True)[:10]
        SUBMISSIONS_DIR.mkdir(exist_ok=True)
        out_csv = SUBMISSIONS_DIR / f"E06_rank_blend_{'_'.join(args.models)}.csv"
        save_submission(recs, out_csv)
        print(f"сабмит E06: {out_csv}")


if __name__ == "__main__":
    main()
