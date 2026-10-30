"""E11: гибридный ранкер — пул SASRec ∪ markov ∪ ease-last + sequential-признаки.

E07 упирается в потолок пула SASRec top-50 (Candidate Recall@50 = 0.406);
markov/ease-last (E09) расширяют покрытие дешевле, чем углубление SASRec-пула.
Проверяем (Н11): CatBoost-ранкер на union-пуле с признаками скоров/рангов всех
генераторов, recency-жанровой аффинностью (последние 3/5 событий), демографией
и позиционно-взвешенной популярностью превысит E07-e128 (0.1949 на 9010)
на >= 4% (>= 0.2027); сабмит — при >= +2% (>= 0.1988).

Пул: SASRec top-100 (unseen) ∪ markov top-30 (last6, decay 0.6, alpha 0.05 —
лучший next1-конфиг E09) ∪ ease-last5 top-30 (λ=1500); source-флаги
in_sasrec / in_markov / in_ease. Генераторы считаются от train-части
соответствующего уровня, seen-фильтр — по всей train-истории.

SASRec — inference-only готовых чекпойнтов e128_len150_ep40 из
results/e05_sasrec_work/e07/ (inner/outer уровни E07; сплиты детерминированы,
переобучение не нужно; в e05_sasrec_work ничего не пишем). Контроль pipeline:
SASRec-only top-10 на внешнем valid = 0.1875 (tail, допуск ±0.003) и
0.2334 (next1, допуск ±0.005); без чистого контроля эксперимент не продолжается.

Протокол двухуровневый, как в E07: ранкер учится на inner-кандидатах с метками
inner_valid (tail) либо первого события inner_valid (next1-диагностика), оценка —
применение того же ранкера к кандидатам внешнего train, метрика на внешнем
valid. Ablation: тот же пул и протокол, но только 14 признаков E07.

Сабмит (при rerank >= 0.1988 на 9010 tail): SASRec на полных events (40 эпох,
свой work-каталог e11_ranker_work/sasrec_full), markov/ease/als/knn/признаки
на полных events, финальный ранкер на всех метках внешнего уровня
(как --submission в E07).

Запуск:
    uv run python scripts/e11_hybrid_ranker.py --control-only   # только контроль чекпойнта
    uv run python scripts/e11_hybrid_ranker.py                  # полный прогон (+сабмит при пороге)
    uv run python scripts/e11_hybrid_ranker.py --no-submission  # полный прогон без сабмита
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# до импорта torch/lightning: ограничиваем BLAS-потоки (совет из примера RePlay)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import lightning as L
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)

from replay.data.nn import ParquetModule, TensorFeatureInfo, TensorSchema
from replay.data.schema import FeatureHint, FeatureType
from replay.nn.lightning import LightningModule
from replay.nn.lightning.callback import PandasTopItemsCallback
from replay.nn.sequential import SasRec
from replay.nn.transform.template import make_default_sasrec_transforms

spec = importlib.util.spec_from_file_location("e05_sasrec", PROJECT_ROOT / "scripts" / "e05_sasrec.py")
e05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e05)

from catboost import CatBoostRanker, Pool
from recsys.data import GENRE_COLUMNS, load_events, load_item_features, load_user_features
from recsys.experiments import DATA_DIR, RESULTS_DIR, save_results
from recsys.metrics import evaluate, save_submission
from recsys.models import ALSRecommender, ItemKNNRecommender
from recsys.sequential import (
    combine_rows,
    context_score_rows,
    ease_weights,
    last_context,
    markov_weights,
    sorted_histories,
    topk_from_scores,
)
from recsys.split import chronological_split

# --- конфигурация ---
SASREC_CONFIG = "e128_len150_ep40"
SASREC_WORK = RESULTS_DIR / "e05_sasrec_work" / "e07" / SASREC_CONFIG
WORK_DIR = RESULTS_DIR / "e11_ranker_work"

EMBEDDING_DIM = 128
MAX_SEQ_LEN = 150
NUM_HEADS = 4
NUM_BLOCKS = 2
DROPOUT = 0.3
EPOCHS = 40

SASREC_TOP = 100       # пул кандидатов SASRec (unseen)
MARKOV_TOP = 30        # пул кандидатов markov
EASE_TOP = 30          # пул кандидатов ease-last
GEN_POOL = 300         # глубина пулов генераторов-признаков (как E07)
EASE_L2 = 1500.0
MARKOV_DEPTH = 6
MARKOV_DECAY = 0.6
MARKOV_ALPHA = 0.05
EASE_LAST_K = 5
SEED = 42

# зафиксированные baselines протокола (журнал E07/E05c) и допуски контроля
TAIL_CONTROL, TAIL_TOL = 0.1875, 0.003
NEXT1_CONTROL, NEXT1_TOL = 0.233444, 0.005
E07_BASELINE = 0.1949          # CatBoost rerank E07 (e128, пул top-50)
SUBMIT_THRESHOLD = 0.1988      # >= +2% к E07
TARGET_THRESHOLD = 0.2027      # цель гипотезы H11: >= +4% к E07

E07_FEATURES = [
    "sasrec_rank", "sasrec_score_norm",
    "als_score", "als_rank", "als_found",
    "knn_score", "knn_rank", "knn_found",
    "item_pop", "item_pos", "item_avg",
    "user_activity", "user_avg", "genre_affinity",
]
SEQ_FEATURES = [
    "in_sasrec", "in_markov", "in_ease",
    "markov_score", "markov_rank", "markov_found",
    "ease_score", "ease_rank", "ease_found",
    "genre_affinity_last3", "genre_affinity_last5",
    "gender_m", "age", "item_wpop",
]
FULL_FEATURES = E07_FEATURES + SEQ_FEATURES


def sasrec_scores_from_checkpoint(
    checkpoint: Path, train: pd.DataFrame, catalog_size: int, out_dir: Path
) -> pd.DataFrame:
    """Inference-only скоринг SASRec готовым чекпойнтом по всему каталогу.

    Predict-стадия один-в-один как в e05_sasrec.py (ParquetModule по train-
    последовательностям, metadata {"predict": ...}, PandasTopItemsCallback
    top_k = catalog_size, Trainer(inference_mode=True)); отличие только в том,
    что веса берутся из чекпойнта вместо обучения. Свои predict-файлы пишем
    в out_dir — в e05_sasrec_work только читаем (чекпойнт).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    padding = catalog_size
    schema = TensorSchema(
        TensorFeatureInfo(
            "item_id",
            is_seq=True,
            padding_value=padding,
            cardinality=catalog_size,
            embedding_dim=EMBEDDING_DIM,
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.ITEM_ID,
        )
    )
    sequences = e05.build_sequences(train)
    # сплиты детерминированы: пересчитанные последовательности обязаны совпасть
    # с теми, на которых обучался чекпойнт (иначе чекпойнт не от этих данных)
    stored = pd.read_parquet(checkpoint.parents[1] / "predict.parquet")
    if len(stored) != len(sequences) or not (
        stored["user_id"].to_numpy() == sequences["user_id"].to_numpy()
    ).all():
        raise SystemExit(f"{checkpoint}: predict-последовательности не совпали с пересчитанными — сплит drift?")
    predict_path = out_dir / "predict.parquet"
    sequences.to_parquet(predict_path, index=False)

    transforms = make_default_sasrec_transforms(schema)
    model = SasRec.from_params(
        schema,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_sequence_length=MAX_SEQ_LEN,
        dropout=DROPOUT,
    )
    module = LightningModule.load_from_checkpoint(str(checkpoint), model=model)
    callback = PandasTopItemsCallback(
        top_k=catalog_size, query_column="user_id", item_column="item_id", rating_column="score"
    )
    datamodule = ParquetModule(
        predict_path=str(predict_path),
        batch_size=64,
        metadata={"predict": {"user_id": {}, "item_id": {"shape": MAX_SEQ_LEN, "padding": padding}}},
        transforms=transforms,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
        inference_mode=True,
        callbacks=[callback],
    )
    trainer.predict(module, datamodule=datamodule, return_predictions=False)
    scores = callback.get_result()
    return scores.sort_values(["user_id", "score"], ascending=[True, False], kind="stable")


def first_valid_event(valid: pd.DataFrame) -> pd.DataFrame:
    """gt = первое по timestamp событие valid-части каждого пользователя (next1)."""
    idx = valid.groupby("user_id")["timestamp"].idxmin()
    return valid.loc[idx, ["user_id", "item_id"]]


def sequential_pools(train: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """markov и ease-last: кандидаты в пул (top-K) и признаки (top-GEN_POOL).

    Конфиги — лучшие next1-конфиги E09: markov last6 / decay 0.6 / alpha 0.05,
    ease last5 / λ=1500. Скор — по последним событиям train-истории,
    seen-фильтр — по всей train-истории (topk_from_scores).
    """
    items = np.sort(train["item_id"].unique())
    histories = sorted_histories(train)
    users, context = last_context(histories, items, k=MARKOV_DEPTH)
    top_k = {"markov": MARKOV_TOP, "ease": EASE_TOP}
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for name, weights, depth, decay in (
        ("markov", markov_weights(train, items, alpha=MARKOV_ALPHA), MARKOV_DEPTH, MARKOV_DECAY),
        ("ease", ease_weights(train, items, l2=EASE_L2), EASE_LAST_K, 1.0),
    ):
        rows = context_score_rows(context[:, :depth], weights)
        scores = combine_rows(rows, depth, decay)
        out[name] = {
            "cands": topk_from_scores(scores, users, items, train, k=top_k[name]),
            "feats": topk_from_scores(scores, users, items, train, k=GEN_POOL),
        }
    return out


def build_pool(
    sasrec_cands: pd.DataFrame, markov_cands: pd.DataFrame, ease_cands: pd.DataFrame
) -> pd.DataFrame:
    """Union кандидатов трёх генераторов с source-флагами.

    Порядок внутри пользователя: sasrec по рангу, затем markov, затем ease
    (повторяющиеся items остаются на первом месте появления) — pool_rank
    задаёт позицию для усечения @100/@130 в диагностике пула.
    """
    frames = []
    for name, cands in (("sasrec", sasrec_cands), ("markov", markov_cands), ("ease", ease_cands)):
        part = cands[["user_id", "item_id"]].copy()
        part["source"] = name
        frames.append(part)
    union = pd.concat(frames, ignore_index=True).drop_duplicates(["user_id", "item_id"], keep="first")
    flags = pd.concat(
        [
            sasrec_cands[["user_id", "item_id"]].assign(in_sasrec=1),
            markov_cands[["user_id", "item_id"]].assign(in_markov=1),
            ease_cands[["user_id", "item_id"]].assign(in_ease=1),
        ],
        ignore_index=True,
    ).groupby(["user_id", "item_id"], as_index=False).max()
    pool = union.merge(flags, on=["user_id", "item_id"], how="left")
    for col in ("in_sasrec", "in_markov", "in_ease"):
        pool[col] = pool[col].fillna(0).astype(np.int8)
    # concat даёт блоки «все sasrec, затем все markov, затем все ease» — строки
    # одного пользователя оказываются разбросаны по фрейму, а group_id CatBoost
    # требует континуальных групп. Stable-сортировка по user_id собирает
    # пользователей подряд, сохраняя порядок источников внутри пользователя.
    pool = pool.sort_values("user_id", kind="stable").reset_index(drop=True)
    pool["pool_rank"] = pool.groupby("user_id", sort=False).cumcount() + 1
    return pool


def _merge_generator(features: pd.DataFrame, top: pd.DataFrame, name: str) -> pd.DataFrame:
    """Скор/ранг/found генератора из его топ-GEN_POOL (вне пула: ранг GEN_POOL+1, скор 0)."""
    features = features.merge(
        top[["user_id", "item_id", "score", "rank"]].rename(
            columns={"score": f"{name}_score", "rank": f"{name}_rank"}
        ),
        on=["user_id", "item_id"], how="left",
    )
    features[f"{name}_found"] = features[f"{name}_rank"].notna().astype(int)
    features[f"{name}_rank"] = features[f"{name}_rank"].fillna(GEN_POOL + 1).astype(int)
    features[f"{name}_score"] = features[f"{name}_score"].fillna(0.0)
    return features


def build_features(
    pool: pd.DataFrame,
    sasrec_full: pd.DataFrame,
    train: pd.DataFrame,
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    seq_feats: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Все признаки пар (user, item) из пула: 14 признаков E07 + 14 sequential-признаков E11.

    Новое против E07: markov/ease скор/ранг/found; source-флаги; жанровая
    аффинность к последним 3/5 событиям (а не всей истории); пол и возраст;
    позиционно-взвешенная популярность item (частота, взвешенная 1/(позиция
    от конца истории)).
    """
    full = sasrec_full.copy()
    full["sasrec_rank"] = full.groupby("user_id")["score"].rank(method="first")
    max_score = full.groupby("user_id")["score"].transform("max")
    min_score = full.groupby("user_id")["score"].transform("min")
    full["sasrec_score_norm"] = (full["score"] - min_score) / (max_score - min_score + 1e-9)
    features = pool.merge(
        full[["user_id", "item_id", "sasrec_rank", "sasrec_score_norm"]],
        on=["user_id", "item_id"], how="left",
    )
    del full

    # ALS и KNN — как в E07 (обучение на train-части уровня)
    for name, model in (("als", ALSRecommender("rating", alpha=1.0)),
                        ("knn", ItemKNNRecommender("binary"))):
        features = _merge_generator(features, model.fit(train).recommend(k=GEN_POOL), name)

    # markov / ease-last — скор/ранг/found из топ-GEN_POOL последовательностного скоринга
    for name, top in seq_feats.items():
        features = _merge_generator(features, top, name)

    item_stats = train.groupby("item_id").agg(
        item_pop=("rating", "size"),
        item_pos=("rating", lambda r: int((r >= 4).sum())),
        item_avg=("rating", "mean"),
    ).reset_index()
    user_stats = train.groupby("user_id").agg(
        user_activity=("rating", "size"), user_avg=("rating", "mean"),
    ).reset_index()
    features = features.merge(item_stats, on="item_id", how="left").merge(user_stats, on="user_id", how="left")

    # жанровая аффинность: доля жанров item в жанровом профиле user —
    # по всей истории (E07) и отдельно по последним 3/5 событиям (E11)
    genres = item_features.set_index("item_id")[list(GENRE_COLUMNS)]
    item_genre = genres.reindex(features["item_id"]).to_numpy(dtype=np.float32)
    ordered = train.sort_values(["user_id", "timestamp"], kind="stable")
    histories = {"genre_affinity": ordered}
    for k in (3, 5):
        histories[f"genre_affinity_last{k}"] = ordered.groupby("user_id", sort=False).tail(k)
    for col_name, source in histories.items():
        profile = (
            source[["user_id", "item_id"]]
            .merge(genres.reset_index(), on="item_id", how="left")
            .groupby("user_id")[list(GENRE_COLUMNS)]
            .mean()
        )
        user_vec = profile.reindex(features["user_id"]).to_numpy(dtype=np.float32)
        features[col_name] = (user_vec * item_genre).sum(axis=1) / (item_genre.sum(axis=1) + 1e-9)

    # демография: пол (M=1/F=0) и возраст как число
    uf = user_features.set_index("user_id")
    features["gender_m"] = features["user_id"].map((uf["gender"] == "M").astype(int)).to_numpy()
    features["age"] = features["user_id"].map(uf["age"]).to_numpy()

    # позиционно-взвешенная популярность: Σ по train 1/(pos_from_end+1),
    # последнему событию истории вес 1; нормировка на сумму весов
    size = ordered.groupby("user_id", sort=False)["user_id"].transform("size").to_numpy()
    cum = ordered.groupby("user_id", sort=False).cumcount().to_numpy()
    pos_weight = 1.0 / (size - cum)
    wpop = pd.Series(pos_weight).groupby(ordered["item_id"].to_numpy()).sum()
    features["item_wpop"] = features["item_id"].map(wpop / pos_weight.sum()).to_numpy()
    return features


def add_labels(features: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    gt = holdout.groupby("user_id")["item_id"].agg(set).to_dict()
    features = features.copy()
    features["label"] = [int(item in gt.get(user, ())) for user, item in zip(features["user_id"], features["item_id"])]
    return features


def fit_ranker(
    features: pd.DataFrame, feature_cols: list[str], seed: int = SEED, tag: str = "ranker"
) -> CatBoostRanker:
    """YetiRank-ранкер; 20% пользователей — early stopping (настройки E07).

    Модель и важности признаков сохраняются в catboost/{tag}.* — важности
    нужны как вход E17 (состав финального бленда) и для журнала E11.
    """
    users = features["user_id"].unique()
    rng = np.random.default_rng(seed)
    eval_users = set(rng.choice(users, size=len(users) // 5, replace=False))
    is_eval = features["user_id"].isin(eval_users)
    model = CatBoostRanker(
        iterations=1500, learning_rate=0.08, depth=8, loss_function="YetiRank",
        random_seed=seed, verbose=False, thread_count=4, early_stopping_rounds=200,
        train_dir=str(WORK_DIR / "catboost"),
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
    print(f"  ранкер обучен: {model.get_best_iteration() + 1} итераций (early stopping)", flush=True)
    importance = dict(sorted(
        zip(feature_cols, (float(value) for value in model.get_feature_importance(train_pool))),
        key=lambda kv: -kv[1],
    ))
    catboost_dir = WORK_DIR / "catboost"
    catboost_dir.mkdir(exist_ok=True)
    (catboost_dir / f"feature_importance_{tag}.json").write_text(
        json.dumps(importance, indent=2, ensure_ascii=False)
    )
    model.save_model(str(catboost_dir / f"{tag}.cbm"))
    return model


def apply_ranker(model: CatBoostRanker, features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Топ-10 внутри пользователя по скорам ранкера."""
    scored = features[["user_id", "item_id"]].copy()
    scored["score"] = model.predict(features[feature_cols])
    scored = scored.sort_values(["user_id", "score"], ascending=[True, False], kind="stable")
    top = scored.groupby("user_id", sort=False).head(10).copy()
    top["rank"] = top.groupby("user_id", sort=False).cumcount() + 1
    return top


def candidate_recall(candidates: pd.DataFrame, gt: pd.DataFrame) -> float:
    """Доля gt-айтемов пользователя, покрытая множеством кандидатов."""
    cand = candidates.groupby("user_id")["item_id"].agg(set).to_dict()
    gt_sets = gt.groupby("user_id")["item_id"].agg(set).to_dict()
    return float(np.mean([len(cand.get(u, set()) & items) / len(items) for u, items in gt_sets.items()]))


def level_pipeline(
    name: str, train_part: pd.DataFrame, sasrec_scores: pd.DataFrame,
    item_features: pd.DataFrame, user_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Кандидаты/пул/признаки одного уровня (inner / outer / full events)."""
    t0 = time.time()
    sasrec_cands = e05.filter_seen_topk(sasrec_scores, train_part, k=SASREC_TOP)
    seq = sequential_pools(train_part)
    pool = build_pool(sasrec_cands, seq["markov"]["cands"], seq["ease"]["cands"])
    feats = build_features(
        pool, sasrec_scores, train_part, item_features, user_features,
        {key: value["feats"] for key, value in seq.items()},
    )
    mean_pool = pool.groupby("user_id").size().mean()
    print(f"[{name}] пул: {len(pool):,} строк, средний размер {mean_pool:.1f} на пользователя; "
          f"признаков: {len(feats.columns)} ({time.time() - t0:.0f} с)", flush=True)
    return sasrec_cands, pool, feats


def panel_to_dict(panel: pd.Series) -> dict:
    return {str(key): float(value) for key, value in panel.items()}


def check_invariants(name: str, panel: pd.Series) -> dict:
    """Инварианты evaluate: без seen/dup в топ-10, все 6040 пользователей."""
    checks = {
        "seen_in_top_share": float(panel["seen_in_top_share"]),
        "dup_in_top_share": float(panel["dup_in_top_share"]),
        "n_eval_users": int(panel["n_eval_users"]),
    }
    ok = checks["seen_in_top_share"] == 0.0 and checks["dup_in_top_share"] == 0.0 and checks["n_eval_users"] == 6040
    print(f"[{name}] инварианты: {'ок' if ok else 'НАРУШЕНЫ'} {checks}", flush=True)
    return checks


def build_submission(
    events: pd.DataFrame, item_features: pd.DataFrame, user_features: pd.DataFrame,
    outer_feats: pd.DataFrame, valid: pd.DataFrame, catalog_size: int,
) -> str:
    """Сабмит: SASRec на полных events (40 эпох) + финальный ранкер на всех внешних метках."""
    print("\n=== сабмит: SASRec на полных events (40 эпох) + финальный ранкер ===", flush=True)
    e05.EMBEDDING_DIM, e05.MAX_SEQ_LEN, e05.NUM_HEADS = EMBEDDING_DIM, MAX_SEQ_LEN, NUM_HEADS
    full_scores = e05.fit_predict_sasrec(events, None, catalog_size, WORK_DIR / "sasrec_full", epochs=EPOCHS)
    _, _, full_feats = level_pipeline("full", events, full_scores, item_features, user_features)
    final_ranker = fit_ranker(add_labels(outer_feats, valid), FULL_FEATURES, tag="final_outer_tail")
    recs = apply_ranker(final_ranker, full_feats, FULL_FEATURES)
    out = PROJECT_ROOT / "submissions" / "E11_hybrid_ranker.csv"
    save_submission(recs, out)
    print(f"сабмит E11: {out}", flush=True)
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-only", action="store_true", help="только контроль SASRec-only и выход")
    parser.add_argument("--no-submission", action="store_true", help="не собирать сабмит даже при достижении порога")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "catboost").mkdir(parents=True, exist_ok=True)

    events = load_events(DATA_DIR)
    item_features = load_item_features(DATA_DIR)
    user_features = load_user_features(DATA_DIR)
    catalog_size = e05.item_catalog(events)
    train, valid = chronological_split(events, train_fraction=0.9)
    inner_train, inner_valid = chronological_split(train, train_fraction=0.9)
    gt_tail, gt_next1 = valid[["user_id", "item_id"]], first_valid_event(valid)
    print(f"events {len(events):,}; train {len(train):,} / valid {len(valid):,}; "
          f"inner_train {len(inner_train):,} / inner_valid {len(inner_valid):,}", flush=True)

    # --- контроль: inference-only внешнего чекпойнта против зафиксированных чисел ---
    # Две независимые базы: tail (топ-10 unseen, весь valid) = 0.1875 — база
    # E07/E11; monitor-эквивалент (топ-10 БЕЗ seen-фильтра, gt = первое событие
    # valid) = 0.2334 — протокол ComputeMetricsCallback при обучении e05
    # (чекпойнт best отбирался именно по нему). Seen-фильтрованный next1 —
    # протокол E07-next1/E09 — заведомо выше monitor: без фильтра заметная доля
    # топ-10 занята уже оценёнными items (диагностика seen_in_top_share).
    ckpt_used, outer_scores, control = None, None, {}
    for ckpt_name in ("best.ckpt", "best-v1.ckpt"):
        ckpt = SASREC_WORK / "outer" / "checkpoints" / ckpt_name
        if not ckpt.is_file():
            continue
        print(f"\n[control] SASRec-only, внешний чекпойнт {ckpt_name}...", flush=True)
        scores = sasrec_scores_from_checkpoint(ckpt, train, catalog_size, WORK_DIR / "outer_infer")
        top10 = e05.filter_seen_topk(scores, train, k=10)
        tail_panel = evaluate(top10, gt_tail, k=10, train_events=train)
        next1_panel = evaluate(top10, gt_next1, k=10, train_events=train)
        monitor_panel = evaluate(
            scores.groupby("user_id", sort=False).head(10), gt_next1, k=10, train_events=train
        )
        print(f"[control] tail={tail_panel['recall@10_macro_relevant']:.4f} "
              f"(ожидание {TAIL_CONTROL}±{TAIL_TOL}); "
              f"next1 seen-фильтр={next1_panel['recall@10_macro_relevant']:.4f}; "
              f"monitor-эквивалент без фильтра={monitor_panel['recall@10_macro_relevant']:.4f} "
              f"(ожидание {NEXT1_CONTROL}±{NEXT1_TOL}, seen в топе "
              f"{monitor_panel['seen_in_top_share']:.2f})", flush=True)
        control = {
            "tail": panel_to_dict(tail_panel),
            "next1_seen_filtered": panel_to_dict(next1_panel),
            "next1_monitor_equivalent_unfiltered": panel_to_dict(monitor_panel),
        }
        if (abs(tail_panel["recall@10_macro_relevant"] - TAIL_CONTROL) <= TAIL_TOL
                and abs(monitor_panel["recall@10_macro_relevant"] - NEXT1_CONTROL) <= NEXT1_TOL):
            ckpt_used, outer_scores = ckpt_name, scores
            break
        print(f"[control] {ckpt_name} вне допуска", flush=True)
    if ckpt_used is None:
        raise SystemExit("контроль не прошёл: внешний чекпойнт не воспроизвёл 0.1875/0.2334 — переобучи outer")
    print(f"[control] контроль пройден на {ckpt_used}", flush=True)
    if args.control_only:
        return

    # --- уровень 1: inner — обучение ранкеров ---
    inner_ckpt = SASREC_WORK / "inner" / "checkpoints" / (
        ckpt_used if (SASREC_WORK / "inner" / "checkpoints" / ckpt_used).is_file() else "best.ckpt"
    )
    print(f"\n[inner] SASRec inference ({inner_ckpt.name})...", flush=True)
    inner_scores = sasrec_scores_from_checkpoint(inner_ckpt, inner_train, catalog_size, WORK_DIR / "inner_infer")
    _, inner_pool, inner_feats = level_pipeline("inner", inner_train, inner_scores, item_features, user_features)
    missing = [c for c in FULL_FEATURES if c not in inner_feats.columns]
    assert not missing, f"нет признаков: {missing}"

    labeled_tail = add_labels(inner_feats, inner_valid)                    # метки inner_valid полностью
    labeled_next1 = add_labels(inner_feats, first_valid_event(inner_valid))  # метки = первое событие inner_valid
    print(f"[inner] hit-rate меток: tail={labeled_tail['label'].mean():.4f}, "
          f"next1={labeled_next1['label'].mean():.4f}", flush=True)

    print("[inner] ранкер 1/3: full-признаки, метки tail (главный)...", flush=True)
    ranker_tail_full = fit_ranker(labeled_tail, FULL_FEATURES, tag="inner_tail_full")
    print("[inner] ранкер 2/3: только признаки E07, метки tail (ablation)...", flush=True)
    ranker_tail_e07 = fit_ranker(labeled_tail, E07_FEATURES, tag="inner_tail_e07")
    print("[inner] ранкер 3/3: full-признаки, метки next1 (диагностика)...", flush=True)
    ranker_next1_full = fit_ranker(labeled_next1, FULL_FEATURES, tag="inner_next1_full")

    # --- уровень 2: outer — честная оценка (outer_scores из контроля) ---
    outer_sasrec_cands, outer_pool, outer_feats = level_pipeline(
        "outer", train, outer_scores, item_features, user_features
    )

    results = {"tail": {"sasrec_only": control["tail"]},
               "next1": {"sasrec_only": control["next1_seen_filtered"]}}
    invariants = {}
    runs = [
        ("tail", "rerank_full", ranker_tail_full, FULL_FEATURES, gt_tail),
        ("tail", "rerank_e07_features", ranker_tail_e07, E07_FEATURES, gt_tail),
        ("next1", "rerank_full", ranker_next1_full, FULL_FEATURES, gt_next1),
    ]
    for protocol, key, ranker, cols, gt in runs:
        reranked = apply_ranker(ranker, outer_feats, cols)
        panel = evaluate(reranked, gt, k=10, train_events=train)
        results[protocol][key] = panel_to_dict(panel)
        invariants[f"{protocol}:{key}"] = check_invariants(f"{protocol}/{key}", panel)
        print(f"[{protocol}] {key}: recall@10={panel['recall@10_macro_relevant']:.4f}", flush=True)

    # --- диагностика пула: потолок кандидатов против SASRec top-50/100 ---
    pool_diag = {
        "mean_pool_size": float(outer_pool.groupby("user_id").size().mean()),
        "tail": {
            "pool_full": candidate_recall(outer_pool, gt_tail),
            "pool@100": candidate_recall(outer_pool[outer_pool["pool_rank"] <= 100], gt_tail),
            "pool@130": candidate_recall(outer_pool[outer_pool["pool_rank"] <= 130], gt_tail),
            "sasrec@50": candidate_recall(outer_sasrec_cands[outer_sasrec_cands["rank"] <= 50], gt_tail),
            "sasrec@100": candidate_recall(outer_sasrec_cands, gt_tail),
        },
        "next1": {
            "pool_full": candidate_recall(outer_pool, gt_next1),
            "pool@100": candidate_recall(outer_pool[outer_pool["pool_rank"] <= 100], gt_next1),
            "pool@130": candidate_recall(outer_pool[outer_pool["pool_rank"] <= 130], gt_next1),
            "sasrec@50": candidate_recall(outer_sasrec_cands[outer_sasrec_cands["rank"] <= 50], gt_next1),
            "sasrec@100": candidate_recall(outer_sasrec_cands, gt_next1),
        },
    }
    print(f"пул: средний размер {pool_diag['mean_pool_size']:.1f}; candidate recall tail: "
          f"pool={pool_diag['tail']['pool_full']:.4f} (sasrec@50={pool_diag['tail']['sasrec@50']:.4f}, "
          f"sasrec@100={pool_diag['tail']['sasrec@100']:.4f}); next1: pool={pool_diag['next1']['pool_full']:.4f}", flush=True)

    tail_metric = results["tail"]["rerank_full"]["recall@10_macro_relevant"]
    e07_gain = tail_metric / E07_BASELINE - 1.0
    print(f"\n[E11] tail recall@10 = {tail_metric:.4f} ({e07_gain:+.1%} к E07 {E07_BASELINE}); "
          f"порог сабмита {SUBMIT_THRESHOLD} {'пройден' if tail_metric >= SUBMIT_THRESHOLD else 'не пройден'}, "
          f"цель H11 {TARGET_THRESHOLD} {'достигнута' if tail_metric >= TARGET_THRESHOLD else 'не достигнута'}", flush=True)

    payload = {
        "config": {
            "sasrec_config": SASREC_CONFIG,
            "checkpoints": {"outer": ckpt_used, "inner": inner_ckpt.name},
            "pool": {"sasrec_top": SASREC_TOP, "markov_top": MARKOV_TOP, "ease_top": EASE_TOP,
                     "markov": f"last{MARKOV_DEPTH} decay {MARKOV_DECAY} alpha {MARKOV_ALPHA}",
                     "ease": f"last{EASE_LAST_K} l2 {EASE_L2}", "gen_pool": GEN_POOL},
            "features": {"e07": E07_FEATURES, "seq": SEQ_FEATURES},
            "protocol": "two-level: ranker on inner (labels: inner_valid tail / first event), evaluated on outer valid",
            "baselines": {"e07_e128": E07_BASELINE, "sasrec_only_e128": TAIL_CONTROL},
            "thresholds": {"submit": SUBMIT_THRESHOLD, "target_h11": TARGET_THRESHOLD},
        },
        "control_sasrec_only": control,
        "results": results,
        "pool_diagnostics": pool_diag,
        "invariants": invariants,
        "feature_importance": {
            tag: json.loads((WORK_DIR / "catboost" / f"feature_importance_{tag}.json").read_text())
            for tag in ("inner_tail_full", "inner_tail_e07", "inner_next1_full")
        },
        "submission": None,
    }
    if tail_metric >= SUBMIT_THRESHOLD and not args.no_submission:
        payload["submission"] = {"path": build_submission(
            events, item_features, user_features, outer_feats, valid, catalog_size
        )}
    path = save_results(payload, "e11_hybrid_ranker")
    print(f"\nрезультаты: {path}", flush=True)


if __name__ == "__main__":
    main()
