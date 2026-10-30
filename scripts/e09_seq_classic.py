"""E09: non-transformer sequential stack — markov + last-K скоринг (проверка).

Черновой прогон 30.10 (notes/hypotheses.md, E09) дал blend 0.2425 против
monitor SASRec e128/len150/40эп = 0.2334, но ad-hoc скрипт не сохранился.
Этот скрипт — чистовая реимплементация того же стека с сохранением панелей
в results/e09_seq_classic.json и явным контролем утечек.

Протоколы (оба leakage-safe, все модели обучаются только на train-части):
- next1@9010 — chronological 90/10, gt = первое событие valid-части каждого
  пользователя. Это протокол monitor SASRec (HR@10 по первому holdout-событию,
  0.233444 у e128/len150/40эп в logs/e05c_sasrec_sweep_v2.log) — база
  сравнения из журнала E09;
- last1 — gt = последнее событие полной истории (last_n_split(1)). Ближайший
  локальный аналог скрытого теста: реконструкция data.md разд. 7 — оцениваются
  события после последнего видимого timestamp.

Варианты: knn_full (контроль, E03-скоринг всей историей), knn_last10,
ease_last5 (λ=1500), markov (сетка last×decay, backoff к unigram), бленд
рангов 4×markov + knn10 + ease5.

Честная оговорка: гиперпараметры markov/бленда выбираются на том же протоколе,
по которому считается итоговое число (у monitor SASRec симметричный эффект
best-epoch selection), — обе стороны сравнения слегка оптимистичны.

Запуск:
    uv run python scripts/e09_seq_classic.py --skip-submission   # только валидация
    uv run python scripts/e09_seq_classic.py                     # + сабмит на полных events
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from recsys.data import load_events
from recsys.experiments import DATA_DIR, save_results
from recsys.metrics import evaluate, save_submission
from recsys.models import ItemKNNRecommender
from recsys.sequential import (
    combine_rows,
    context_score_rows,
    cosine_weights,
    ease_weights,
    last_context,
    markov_weights,
    popularity_vector,
    sorted_histories,
    topk_from_scores,
)
from recsys.split import chronological_split, last_n_split

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# зафиксированные конфиги чернового прогона (журнал E09)
EASE_L2 = 1500.0
KNN_LAST_K = 10
EASE_LAST_K = 5
MARKOV_ALPHA = 0.05
GRID_DEPTH = range(1, 7)
GRID_DECAY = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID_BACKOFF_W = (0.0, 1.0, 2.0, 3.0, 5.0)
BLEND_POOL = 200
BLEND_K = 10

# monitor SASRec e128/len150/40эп на next1@9010 — база сравнения E09
MONITOR_BASELINE = 0.233444


def panel_to_dict(panel: pd.Series) -> dict:
    return {str(key): float(value) for key, value in panel.items()}


def first_valid_event(valid: pd.DataFrame) -> pd.DataFrame:
    """gt = первое событие valid-части каждого пользователя (по timestamp)."""
    idx = valid.groupby("user_id")["timestamp"].idxmin()
    return valid.loc[idx, ["user_id", "item_id"]]


def rank_blend(
    recs_by_model: dict[str, pd.DataFrame],
    weights: dict[str, float],
    pool: int = BLEND_POOL,
    k: int = BLEND_K,
) -> pd.DataFrame:
    """Взвешенное среднее нормированных рангов на общем пуле топ-pool кандидатов.

    Нормированный ранг 1 − (rank − 1)/pool; комбинированный скор — сумма
    взвешенных рангов по всем моделям, где item попал в пул. Тай-брейк по
    item_id — детерминированность сабмитов (как в E01).
    """
    frames = []
    for name, recs in recs_by_model.items():
        top = recs[recs["rank"] <= pool].copy()
        top["weighted"] = weights[name] * (1.0 - (top["rank"] - 1) / pool)
        frames.append(top[["user_id", "item_id", "weighted"]])
    merged = pd.concat(frames, ignore_index=True)
    combined = merged.groupby(["user_id", "item_id"], as_index=False)["weighted"].sum()
    combined = combined.sort_values(
        ["user_id", "weighted", "item_id"], ascending=[True, False, True], kind="stable"
    )
    combined["rank"] = combined.groupby("user_id").cumcount() + 1
    return combined.loc[
        combined["rank"] <= k, ["user_id", "item_id", "weighted", "rank"]
    ].rename(columns={"weighted": "score"})


def run_protocol(name: str, train: pd.DataFrame, gt: pd.DataFrame) -> dict:
    print(f"\n=== протокол {name}: train={len(train)}, gt={len(gt)} пользователей", flush=True)
    items = np.sort(train["item_id"].unique())
    histories = sorted_histories(train)
    users, context = last_context(histories, items, k=KNN_LAST_K)

    print("строим item-item матрицы: cosine, EASE(λ=1500), markov...", flush=True)
    w_cosine = cosine_weights(train, items)
    w_ease = ease_weights(train, items, l2=EASE_L2)
    w_markov = markov_weights(train, items, alpha=MARKOV_ALPHA)
    pop = popularity_vector(train, items)

    variants: dict[str, pd.Series] = {}

    # контроль: классический полный-history скоринг (E03)
    recs_full = ItemKNNRecommender("binary").fit(train).recommend(BLEND_K)
    variants["knn_full"] = evaluate(recs_full, gt, train_events=train)

    def rows_for(weights: np.ndarray, depth: int) -> list[np.ndarray]:
        return context_score_rows(context[:, :depth], weights)

    # knn / ease с last-K скорингом (конфиги черновика)
    knn_scores = combine_rows(rows_for(w_cosine, KNN_LAST_K), KNN_LAST_K, 1.0)
    variants["knn_last10"] = evaluate(
        topk_from_scores(knn_scores, users, items, train, BLEND_K), gt, train_events=train
    )
    ease_scores = combine_rows(rows_for(w_ease, EASE_LAST_K), EASE_LAST_K, 1.0)
    variants["ease_last5"] = evaluate(
        topk_from_scores(ease_scores, users, items, train, BLEND_K), gt, train_events=train
    )
    for variant in ("knn_full", "knn_last10", "ease_last5"):
        print(f"[{name}] {variant}: {variants[variant]['recall@10_macro_relevant']:.4f}", flush=True)

    # сетка markov: строки контекста считаются один раз, decay/depth — комбинации
    print("сетка markov (last × decay)...", flush=True)
    markov_rows = rows_for(w_markov, max(GRID_DEPTH))
    grid = []
    for depth in GRID_DEPTH:
        for decay in GRID_DECAY:
            scores = combine_rows(markov_rows, depth, decay)
            recall = evaluate(
                topk_from_scores(scores, users, items, train, BLEND_K), gt
            )["recall@10_macro_relevant"]
            grid.append(
                {"depth": depth, "decay": decay, "recall@10_macro_relevant": float(recall)}
            )
    best = max(grid, key=lambda r: r["recall@10_macro_relevant"])
    print(f"[{name}] markov best: last={best['depth']}, decay={best['decay']} -> {best['recall@10_macro_relevant']:.4f}", flush=True)

    # backoff к unigram-популярности на лучшей точке сетки
    best_scores = combine_rows(markov_rows, best["depth"], best["decay"])
    backoff_grid = []
    for w in GRID_BACKOFF_W:
        recall = evaluate(
            topk_from_scores(best_scores + w * pop[None, :], users, items, train, BLEND_K),
            gt,
        )["recall@10_macro_relevant"]
        backoff_grid.append({"w": w, "recall@10_macro_relevant": float(recall)})
    best_backoff = max(backoff_grid, key=lambda r: r["recall@10_macro_relevant"])
    markov_final_scores = best_scores + best_backoff["w"] * pop[None, :]
    variants["markov_backoff"] = evaluate(
        topk_from_scores(markov_final_scores, users, items, train, BLEND_K), gt, train_events=train
    )
    print(f"[{name}] markov+backoff w={best_backoff['w']}: {variants['markov_backoff']['recall@10_macro_relevant']:.4f}", flush=True)

    # бленды на общем пуле топ-200: markov-компонент чистый и с backoff,
    # с knn10 и без (на last1 knn10 заметно слабее и может разбавлять — урок E06)
    pool = {
        "knn10": topk_from_scores(knn_scores, users, items, train, BLEND_POOL),
        "ease5": topk_from_scores(ease_scores, users, items, train, BLEND_POOL),
        "markov": topk_from_scores(best_scores, users, items, train, BLEND_POOL),
        "markov_bo": topk_from_scores(markov_final_scores, users, items, train, BLEND_POOL),
    }
    blend_configs = {
        "blend_draft": {"markov_bo": 4.0, "knn10": 1.0, "ease5": 1.0},
        "blend_pure": {"markov": 4.0, "knn10": 1.0, "ease5": 1.0},
        "blend_mk_ease": {"markov": 4.0, "ease5": 1.0},
    }
    for blend_name, weights in blend_configs.items():
        blend = rank_blend({key: pool[key] for key in weights}, weights)
        variants[blend_name] = evaluate(blend, gt, train_events=train)
        print(
            f"[{name}] {blend_name} {weights}: "
            f"{variants[blend_name]['recall@10_macro_relevant']:.4f}",
            flush=True,
        )

    # чувствительность к alpha сглаживания markov на лучшем конфиге
    sensitivity = {}
    for alpha in (0.01, 0.2):
        scores = combine_rows(
            context_score_rows(
                context[:, : best["depth"]], markov_weights(train, items, alpha=alpha)
            ),
            best["depth"],
            best["decay"],
        ) + best_backoff["w"] * popularity_vector(train, items)[None, :]
        sensitivity[str(alpha)] = float(
            evaluate(topk_from_scores(scores, users, items, train, BLEND_K), gt)[
                "recall@10_macro_relevant"
            ]
        )
    print(f"[{name}] чувствительность alpha (0.05 в основе): {sensitivity}", flush=True)

    return {
        "split": {"train": len(train), "gt_users": int(len(gt))},
        "variants": {k: panel_to_dict(v) for k, v in variants.items()},
        "markov_best": {"depth": best["depth"], "decay": best["decay"]},
        "markov_grid": grid,
        "backoff_grid": backoff_grid,
        "backoff_best_w": best_backoff["w"],
        "alpha_sensitivity": sensitivity,
        "blend_configs": blend_configs,
    }


def build_submission(config: dict) -> dict:
    """Бленд на полных events конфигом, выигравшим на next1@9010."""
    print(f"\n=== сабмит: обучение на полных events, бленд {config['blend']}", flush=True)
    events = load_events(DATA_DIR)
    items = np.sort(events["item_id"].unique())
    histories = sorted_histories(events)
    users, context = last_context(histories, items, k=KNN_LAST_K)

    markov_pure = combine_rows(
        context_score_rows(
            context[:, : config["depth"]],
            markov_weights(events, items, alpha=MARKOV_ALPHA),
        ),
        config["depth"],
        config["decay"],
    )
    markov_scores = markov_pure + config["backoff_w"] * popularity_vector(events, items)[None, :]
    knn_scores = combine_rows(
        context_score_rows(context[:, :KNN_LAST_K], cosine_weights(events, items)),
        KNN_LAST_K,
        1.0,
    )
    ease_scores = combine_rows(
        context_score_rows(context[:, :EASE_LAST_K], ease_weights(events, items, l2=EASE_L2)),
        EASE_LAST_K,
        1.0,
    )
    pool = {
        "knn10": topk_from_scores(knn_scores, users, items, events, BLEND_POOL),
        "ease5": topk_from_scores(ease_scores, users, items, events, BLEND_POOL),
        "markov": topk_from_scores(markov_pure, users, items, events, BLEND_POOL),
        "markov_bo": topk_from_scores(markov_scores, users, items, events, BLEND_POOL),
    }
    blend = rank_blend({key: pool[key] for key in config["weights"]}, config["weights"])
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out = SUBMISSIONS_DIR / f"E09_seq_classic_{config['blend']}.csv"
    save_submission(blend, out)
    print(f"сабмит собран: {out}", flush=True)
    return {"path": str(out), "n_users": int(blend["user_id"].nunique())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-submission", action="store_true", help="только локальная валидация")
    parser.add_argument(
        "--select-on",
        choices=("last1", "next1@9010"),
        default="last1",
        help="протокол выбора конфига бленда для сабмита (last1 — локальный "
        "аналог условий сабмита: обучение почти на всех visible-событиях, "
        "предсказание после последнего видимого)",
    )
    args = parser.parse_args()

    events = load_events(DATA_DIR)
    train_9010, valid_9010 = chronological_split(events, train_fraction=0.9)
    train_last1, valid_last1 = last_n_split(events, n=1)

    payload = {
        "monitor_baseline_sasrec_e128_len150_ep40": MONITOR_BASELINE,
        "protocols": {
            "next1@9010": run_protocol("next1@9010", train_9010, first_valid_event(valid_9010)),
            "last1": run_protocol("last1", train_last1, valid_last1[["user_id", "item_id"]]),
        },
    }

    selection = payload["protocols"][args.select_on]
    blends_on_selection = {
        name: panel["recall@10_macro_relevant"]
        for name, panel in selection["variants"].items()
        if name.startswith("blend")
    }
    best_blend = max(blends_on_selection, key=blends_on_selection.get)
    payload["blend_config_for_submission"] = {
        "selected_on": args.select_on,
        "blend": best_blend,
        "weights": selection["blend_configs"][best_blend],
        **selection["markov_best"],
        "backoff_w": selection["backoff_best_w"] if "markov_bo" in selection["blend_configs"][best_blend] else 0.0,
    }
    if not args.skip_submission:
        payload["submission"] = build_submission(payload["blend_config_for_submission"])

    path = save_results(payload, "e09_seq_classic")
    print(f"\nрезультаты: {path}")

    def best_blend_on(protocol_block: dict) -> tuple[str, float]:
        name = max(
            (n for n in protocol_block["variants"] if n.startswith("blend")),
            key=lambda n: protocol_block["variants"][n]["recall@10_macro_relevant"],
        )
        return name, protocol_block["variants"][name]["recall@10_macro_relevant"]

    print(f"лучший бленд на next1@9010: {best_blend_on(payload['protocols']['next1@9010'])} "
          f"против monitor SASRec 0.2334")
    print(f"лучший бленд на last1: {best_blend_on(payload['protocols']['last1'])} "
          f"(протокол-аналог сабмита, множитель ≈ ×0.93–1.0 по popularity-калибровке)")
    if not args.skip_submission:
        cfg = payload["blend_config_for_submission"]
        print(f"сабмит собран конфигом с {args.select_on}: {cfg['blend']} {cfg['weights']}, "
              f"markov last{cfg['depth']}/decay {cfg['decay']}, backoff w={cfg['backoff_w']}")


if __name__ == "__main__":
    main()
