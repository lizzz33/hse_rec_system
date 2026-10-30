"""E07: reranking кандидатов SASRec бустингом (обучаемые веса вместо бленда E06).

Гипотеза: равномерный blend E06 проиграл, потому что веса фиксированы и модель
не знает, кому доверять в конкретной точке. CatBoost-ранкер на кандидатах
SASRec top-50 с признаками генераторов и контекста должен выучить, когда
доверять ALS/KNN/popularity против SASRec-скора, и поднять Recall@10 над
чистым SASRec на том же пуле кандидатов.

Протокол валидации — двухуровневый, без утечки меток в оценку:

    events = train_90% | valid_10%          (внешний уровень, как в E01–E06)
    train  = inner_90% | inner_valid_10%    (внутренний уровень)

- ранкер обучается на кандидатах/признаках inner-уровня с метками inner_valid;
- оценка — кандидаты/признаки внешнего train, применение того же ранкера,
  метрика на внешнем valid (метки которого ранкер никогда не видел);
- сабмит — SASRec и признаки на полных events + ранкер, дообученный на всех
  метках внешнего уровня (для скрытого теста все выданные данные — легальны).

Признаки (все — только от train-части соответствующего уровня):
- sasrec score_norm / rank (нормированный скор и позиция в каталоге);
- als score / rank / found (als_rating, топ-300);
- knn score / rank / found (binary cosine, топ-300);
- item: popularity, positive count (rating>=4), avg rating;
- user: активность, средняя оценка;
- user-item: жанровое сходство (доля жанров item в истории user).

Контроль: те же кандидаты, ранжированные чистым SASRec-скором.

Запуск:
    OMP_NUM_THREADS=1 uv run python scripts/e07_reranker.py --sasrec-config e64_len100_ep40 --submission
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("e05_sasrec", PROJECT_ROOT / "scripts" / "e05_sasrec.py")
e05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e05)

from catboost import CatBoostRanker, Pool
from recsys.data import GENRE_COLUMNS, load_events, load_item_features
from recsys.experiments import RESULTS_DIR, SPLITTERS
from recsys.metrics import evaluate, save_submission
from recsys.models import ALSRecommender, EASERecommender, ItemKNNRecommender
from recsys.split import chronological_split

N_CANDIDATES = 50
GEN_POOL = 300  # глубина пулов ALS/KNN/EASE для признаков
SEED = 42


def sasrec_candidates(train, valid, catalog_size, work_dir, epochs) -> pd.DataFrame:
    """Кандидаты: топ-N_CANDIDATES unseen items по скорам SASRec (+ полный скор-лист)."""
    scores = e05.fit_predict_sasrec(train, valid, catalog_size, work_dir, epochs=epochs)
    return e05.filter_seen_topk(scores, train, k=N_CANDIDATES), scores


def build_features(candidates: pd.DataFrame, sasrec_full: pd.DataFrame, train: pd.DataFrame,
                   item_features: pd.DataFrame, with_ease: bool = False) -> pd.DataFrame:
    """Признаки для каждой пары (user, item) из пула кандидатов SASRec."""
    full = sasrec_full.copy()
    full["sasrec_rank"] = full.groupby("user_id")["score"].rank(method="first")
    max_score = full.groupby("user_id")["score"].transform("max")
    min_score = full.groupby("user_id")["score"].transform("min")
    full["sasrec_score_norm"] = (full["score"] - min_score) / (max_score - min_score + 1e-9)
    features = candidates[["user_id", "item_id"]].merge(
        full[["user_id", "item_id", "sasrec_rank", "sasrec_score_norm"]], on=["user_id", "item_id"], how="left"
    )

    # ALS, KNN и (опционально) EASE: скоры/ранги из топ-GEN_POOL;
    # вне пула — ранг GEN_POOL+1, скор 0. EASE почти не пересекается с SASRec
    # (0.9/10 в топ-10) — самый комплементарный из классических генераторов
    generators = [
        ("als", ALSRecommender("rating", alpha=1.0)),
        ("knn", ItemKNNRecommender("binary")),
    ]
    if with_ease:
        generators.append(("ease", EASERecommender(l2=1500.0)))
    for name, model in generators:
        pool = model.fit(train).recommend(k=GEN_POOL)
        features = features.merge(
            pool[["user_id", "item_id", "score", "rank"]].rename(
                columns={"score": f"{name}_score", "rank": f"{name}_rank"}
            ),
            on=["user_id", "item_id"], how="left",
        )
        features[f"{name}_found"] = features[f"{name}_rank"].notna().astype(int)
        features[f"{name}_rank"] = features[f"{name}_rank"].fillna(GEN_POOL + 1).astype(int)
        features[f"{name}_score"] = features[f"{name}_score"].fillna(0.0)

    item_stats = train.groupby("item_id").agg(
        item_pop=("rating", "size"),
        item_pos=("rating", lambda r: int((r >= 4).sum())),
        item_avg=("rating", "mean"),
    ).reset_index()
    user_stats = train.groupby("user_id").agg(
        user_activity=("rating", "size"), user_avg=("rating", "mean"),
    ).reset_index()
    features = features.merge(item_stats, on="item_id", how="left").merge(user_stats, on="user_id", how="left")

    # жанровое сходство: доля жанров item, покрытая жанровым профилем user
    genres = item_features.set_index("item_id")[list(GENRE_COLUMNS)]
    user_genre = (
        train[["user_id", "item_id"]]
        .merge(genres.reset_index(), on="item_id", how="left")
        .groupby("user_id")[list(GENRE_COLUMNS)]
        .mean()
    )
    item_genre = genres.reindex(features["item_id"]).to_numpy(dtype=np.float32)
    user_genre_vec = user_genre.reindex(features["user_id"]).to_numpy(dtype=np.float32)
    features["genre_affinity"] = (user_genre_vec * item_genre).sum(axis=1) / (item_genre.sum(axis=1) + 1e-9)
    return features


def next1_of(events: pd.DataFrame) -> pd.DataFrame:
    """Одно «следующее» событие пользователя: первое holdout-событие или
    последнее событие самого events (для inner-уровня: оно исключается из
    inner-обучения SASRec выше по скрипту через chronological_split)."""
    return (events.sort_values(["user_id", "timestamp"])
                  .groupby("user_id").first().reset_index()[["user_id", "item_id"]])


def add_labels(features: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    gt = holdout.groupby("user_id")["item_id"].agg(set).to_dict()
    features = features.copy()
    features["label"] = [int(item in gt.get(user, ())) for user, item in zip(features["user_id"], features["item_id"])]
    return features


def fit_ranker(features: pd.DataFrame, feature_cols: list[str], seed: int = SEED) -> CatBoostRanker:
    """YetiRank-ранкер; 20% пользователей — early stopping."""
    users = features["user_id"].unique()
    rng = np.random.default_rng(seed)
    eval_users = set(rng.choice(users, size=len(users) // 5, replace=False))
    is_eval = features["user_id"].isin(eval_users)
    model = CatBoostRanker(
        iterations=1500, learning_rate=0.08, depth=8, loss_function="YetiRank",
        random_seed=seed, verbose=False, thread_count=4, early_stopping_rounds=200,
    )
    train_pool = Pool(
        features.loc[~is_eval, feature_cols], features.loc[~is_eval, "label"],
        group_id=features.loc[~is_eval, "user_id"].to_numpy(),
    )
    eval_pool = Pool(
        features.loc[is_eval, feature_cols], features.loc[is_eval, "label"],
        group_id=features.loc[is_eval, "user_id"].to_numpy(),
    )
    model.fit(train_pool, eval_set=eval_pool)
    return model


def apply_ranker(model: CatBoostRanker, features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Топ-10 внутри пользователя по скорам ранкера."""
    scored = features[["user_id", "item_id"]].copy()
    scored["score"] = model.predict(features[feature_cols])
    scored = scored.sort_values(["user_id", "score"], ascending=[True, False], kind="stable")
    top = scored.groupby("user_id", sort=False).head(10)
    top["rank"] = top.groupby("user_id", sort=False).cumcount() + 1
    return top


def main() -> None:
    global N_CANDIDATES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sasrec-config", default="e64_len100_ep40",
                        help="имя конфига из свипа e05c (для протокола журнала)")
    parser.add_argument("--epochs", type=int, default=None, help="эпохи SASRec (по умолчанию из имени конфига)")
    parser.add_argument("--submission", action="store_true", help="собрать сабмит на полных events")
    parser.add_argument("--with-ease", action="store_true", help="добавить признаки EASE (lambda=1500)")
    parser.add_argument("--pool", type=int, default=N_CANDIDATES, help="размер пула кандидатов SASRec")
    parser.add_argument("--protocol", choices=["tail", "next1"], default="tail",
                        help="tail: метки/оценка по 10%% хвосту; next1: метки = последнее событие "
                             "inner, оценка = первое событие внешнего holdout (реконструкция "
                             "скрытого теста = последние 1-3 события, data.md раздел 7)")
    args = parser.parse_args()

    N_CANDIDATES = args.pool
    events = load_events()
    item_features = load_item_features()
    catalog_size = e05.item_catalog(events)
    # конфиг вида e{dim}_len{len}_ep{epochs}: применяем к модулю e05 целиком,
    # иначе fit_predict_sasrec останется на дефолтных e64/len100
    parts = args.sasrec_config.split("_")
    e05.EMBEDDING_DIM = int(parts[0][1:])
    e05.MAX_SEQ_LEN = int(parts[1][3:])
    e05.NUM_HEADS = 4 if e05.EMBEDDING_DIM >= 128 else 2
    epochs = args.epochs or int(parts[2][2:])
    # рабочий каталог — с именем конфига: в нём лежит чекпойнт лучшей эпохи,
    # и чекпойнт другой размерности (например, e64 после e128) не подгрузится
    work = RESULTS_DIR / "e05_sasrec_work" / "e07" / args.sasrec_config
    print(f"SASRec: {args.sasrec_config} -> dim={e05.EMBEDDING_DIM}, len={e05.MAX_SEQ_LEN}, "
          f"heads={e05.NUM_HEADS}, epochs={epochs}", flush=True)

    # --- уровень 1: ранкер обучается внутри train ---
    train, valid = SPLITTERS["9010"](events)
    inner_train, inner_valid = chronological_split(train, train_fraction=0.9)
    # next1 (реконструкция скрытого теста, data.md раздел 7): метки ранкера =
    # первое событие inner_valid, оценка = первое событие внешнего valid
    # (в точности monitor-протокол SASRec); tail — прежние полные хвосты
    if args.protocol == "next1":
        inner_label_gt = next1_of(inner_valid)
        outer_eval_gt = next1_of(valid)
    else:
        inner_label_gt = inner_valid
        outer_eval_gt = valid
    print(f"[inner] SASRec на {len(inner_train):,} событий...", flush=True)
    inner_candidates, inner_full = sasrec_candidates(
        inner_train, inner_valid, catalog_size, work / "inner", epochs
    )
    inner_features = add_labels(
        build_features(inner_candidates, inner_full, inner_train, item_features, args.with_ease), inner_label_gt
    )
    feature_cols = [c for c in inner_features.columns if c not in ("user_id", "item_id", "label")]
    print(f"признаков: {len(feature_cols)}; inner кандидатов: {len(inner_features):,}; "
          f"hit-rate: {inner_features['label'].mean():.4f}", flush=True)
    ranker = fit_ranker(inner_features, feature_cols)

    # --- уровень 2: честная оценка на внешнем valid ---
    print(f"[outer] SASRec на {len(train):,} событий...", flush=True)
    candidates, sasrec_full = sasrec_candidates(train, valid, catalog_size, work / "outer", epochs)
    features = build_features(candidates, sasrec_full, train, item_features, args.with_ease)
    reranked = apply_ranker(ranker, features, feature_cols)
    panel = evaluate(reranked, outer_eval_gt, k=10, train_events=train)
    control = candidates.sort_values(["user_id", "rank"]).groupby("user_id", sort=False).head(10)
    control_panel = evaluate(control, outer_eval_gt, k=10, train_events=train)
    print(f"[{args.protocol}] контроль SASRec-only ({epochs} эп): recall@10={control_panel['recall@10_macro_relevant']:.4f}")
    print(f"[{args.protocol}] catboost reranking (честно):        recall@10={panel['recall@10_macro_relevant']:.4f}", flush=True)

    payload = {
        "sasrec_config": f"{args.sasrec_config}{'_ease' if args.with_ease else ''}_pool{args.pool}",
        "protocol": f"two-level ({args.protocol}): ranker on inner, evaluated on outer",
        "results": {args.protocol: {
            "sasrec_only": {str(k): float(v) for k, v in control_panel.items()},
            "catboost_rerank": {str(k): float(v) for k, v in panel.items()},
        }},
    }
    out = RESULTS_DIR / f"e07_reranker{'_ease' if args.with_ease else ''}_pool{args.pool}_{args.protocol}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"результаты: {out}")

    if args.submission:
        # --- финал: SASRec и признаки на полных events, ранкер на всех внешних метках ---
        print("\n=== сабмит на полных events ===", flush=True)
        outer_features = add_labels(build_features(candidates, sasrec_full, train, item_features, args.with_ease), outer_eval_gt)
        final_ranker = fit_ranker(outer_features, feature_cols)
        full_scores = e05.fit_predict_sasrec(events, None, catalog_size, work / "full", epochs=epochs)
        full_candidates = e05.filter_seen_topk(full_scores, events, k=N_CANDIDATES)
        full_features = build_features(full_candidates, full_scores, events, item_features, args.with_ease)
        recs = apply_ranker(final_ranker, full_features, feature_cols)
        out_csv = PROJECT_ROOT / "submissions" / f"E07_catboost_rerank_{args.sasrec_config}{'_ease' if args.with_ease else ''}_pool{args.pool}_{args.protocol}.csv"
        save_submission(recs, out_csv)
        print(f"сабмит E07: {out_csv}")


if __name__ == "__main__":
    main()
