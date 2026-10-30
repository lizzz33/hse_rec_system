"""E10: rank-blend SASRec + markov + ease без обучения (фиксированные веса).

E09 показал комплементарность sequential-классики и SASRec: markov+ease
0.2515 (seen-фильтрованный) против monitor SASRec e128/len150/40эп 0.2334
на next1@9010. В ходе E10 установлено, что monitor RePlay не фильтрует
уже оценённые items: нефильтрованный monitor = 0.2334, фильтрованная сила
SASRec на том же протоколе = 0.319 — «бленд > monitor» из E09, вероятно,
артефакт сопоставления фильтрованного с нефильтрованным. E06 отверг
равномерный бленд слабых генераторов; здесь проверяем взвешенный rank-blend
двух сильных сигналов с честным протоколом весов: подбор на inner-уровне,
фиксация, оценка на внешнем next1@9010 против фильтрованного SASRec.

Протокол весов (без подглядывания во внешнюю оценку):
    inner:  SASRec = inference inner-чекпойнта (обучен на chronological_split
            (train_9010, 0.9).train), markov/ease на inner_train, gt = первое
            событие inner_valid;
    outer:  SASRec = inference outer-чекпойнта (train_9010), markov depth6/decay0.6
            (лучший next1-конфиг E09), веса фиксированы с inner;
    last1:  тот же бленд на last_n_split(events, 1) — протокол-аналог сабмита.

SASRec-скоры считаются inference-only по готовым чекпойнтам E07
(results/e05_sasrec_work/e07/e128_len150_ep40/{inner,outer}; туда ничего
не пишется) — переобучение только там, где чекпойнта нет (last1 и полные
events), и только в свой work-каталог results/e10_hybrid_blend_work/.

Стадии:
    check        — детерминизм сплитов: parquet чекпойнтов совпадает со свежим
                   chronological_split(events, 0.9) и его inner-уровнем;
    control      — инференс outer против gt = первое событие valid_9010, два числа:
                   нефильтрованный топ-10 = monitor 0.233444 (±0.005; RePlay
                   ComputeMetricsCallback не фильтрует seen) и фильтрованный
                   топ-10 ~0.319 — честная сила SASRec, база планки сабмита;
    inner        — инференс inner + markov/ease пулы + сетка весов -> фиксация;
    outer        — фиксированные веса на next1@9010 + потолок (подбор на outer,
                   диагностический) + overlap-диагностика;
    sasrec-last1 — обучение SASRec e128/40эп на train_last1 с monitor (долгая, фон);
    last1        — бленд с фиксированными весами на last1 + контроль SASRec-only;
    submit       — при blend >= monitor * 1.02 (~0.2381): стек на полных events
                   (SASRec обучается с нуля) и сабмит submissions/E10_hybrid_blend.csv.

Запуск:
    uv run python scripts/e10_hybrid_blend.py --stage check
    uv run python scripts/e10_hybrid_blend.py --stage control   # и далее по порядку
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# до импорта torch/lightning: ограничиваем BLAS-потоки (как в e05)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

spec = importlib.util.spec_from_file_location("e05_sasrec", PROJECT_ROOT / "scripts" / "e05_sasrec.py")
e05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e05)

# конфиг e128_len150_ep40 (свип e05c / чекпойнты E07): применяем к e05 целиком,
# иначе from_params и metadata предсказания разойдутся с чекпойнтом
e05.EMBEDDING_DIM = 128
e05.MAX_SEQ_LEN = 150
e05.NUM_HEADS = 4
SASREC_EPOCHS = 40

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

from recsys.data import load_events
from recsys.experiments import DATA_DIR, RESULTS_DIR
from recsys.metrics import evaluate, save_submission
from recsys.sequential import (
    combine_rows,
    context_score_rows,
    ease_weights,
    last_context,
    markov_weights,
    sorted_histories,
    topk_from_scores,
)
from recsys.split import chronological_split, last_n_split

WORK_DIR = RESULTS_DIR / "e10_hybrid_blend_work"
CACHE_DIR = WORK_DIR / "cache"
CKPT_DIR = RESULTS_DIR / "e05_sasrec_work" / "e07" / "e128_len150_ep40"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
STATE_PATH = RESULTS_DIR / "e10_hybrid_blend.json"

# зафиксированные конфиги (E09/E10 план)
EASE_L2 = 1500.0
EASE_LAST_K = 5
MARKOV_ALPHA = 0.05
MARKOV_DEPTH = 6
PRIMARY_DECAY = 0.6  # лучший next1-конфиг E09 — предписан для внешней оценки
GRID_DECAY = (0.5, 0.6, 0.7)
GRID_A = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
GRID_M = (0.5, 1.0, 1.5, 2.0, 3.0)
GRID_E = (0.25, 0.5, 1.0, 1.5)
POOL = 200
TOP_K = 10

MONITOR_BASELINE = 0.233444  # monitor e05c: recall@10 БЕЗ seen-фильтра (RePlay)
CONTROL_TOLERANCE = 0.005
# планка сабмита: честная сила SASRec на next1@9010 — фильтрованный контроль
# (пересчитывается в stage_outer из state["control"]["filtered_recall"] как
# filtered * 1.02 ~= 0.3254); monitor*1.02 = 0.2381 оставлена как legacy-планка
CRITERION_LEGACY = MONITOR_BASELINE * 1.02  # ~0.2381, для непрерывности журнала

# опорные числа E09 на тех же протоколах — для перекрёстных проверок
# (все — seen-фильтрованные: пулы E09 строятся через topk_from_scores)
E09_REF = {
    "next1@9010": {"markov_d6_0.6": 0.2359, "ease_last5": 0.2139, "mk_ease_4_1": 0.2515},
    "last1": {"markov_d6_0.7": 0.2263, "ease_last5": 0.1960, "mk_ease_4_1_d0.7": 0.2374},
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def panel_to_dict(panel: pd.Series) -> dict:
    return {str(key): float(value) for key, value in panel.items()}


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    """Сохранить состояние с merge по верхнему уровню ключей.

    Стадии могут выполняться параллельными процессами (обучение last1 идёт
    в фоне, пока считаются inner/outer) — перезапись файла целиком стирала
    бы чужие ключи; merge сохраняет всё, что появилось на диске после
    load_state этого процесса.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    if STATE_PATH.is_file():
        on_disk = json.loads(STATE_PATH.read_text())
        on_disk.update(state)
        state = on_disk
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    log(f"результаты: {STATE_PATH} (ключи: {sorted(state)})")


def check_invariants(panel: pd.Series, stage: str, n_users: int = 6040) -> None:
    """Инварианты панелей: без seen/duplicates в топ-k, все пользователи."""
    for key in ("seen_in_top_share", "dup_in_top_share"):
        if panel[key] != 0.0:
            raise RuntimeError(f"[{stage}] инвариант нарушен: {key}={panel[key]}")
    if panel["n_eval_users"] != n_users:
        raise RuntimeError(f"[{stage}] n_eval_users={panel['n_eval_users']} != {n_users}")


def first_valid_event(valid: pd.DataFrame) -> pd.DataFrame:
    """gt = первое событие valid-части каждого пользователя (по timestamp)."""
    idx = valid.groupby("user_id")["timestamp"].idxmin()
    return valid.loc[idx, ["user_id", "item_id"]]


# --------------------------------------------------------------------------
# SASRec: inference-only по готовому чекпойнту
# --------------------------------------------------------------------------

def sasrec_scores_from_checkpoint(
    ckpt_path: Path, predict_parquet: Path, catalog_size: int
) -> pd.DataFrame:
    """Скоры SASRec по всему каталогу весами готового чекпойнта.

    Predict-стадия один-в-один как в e05_sasrec.fit_predict_sasrec:
    ParquetModule по существующему predict.parquet, PandasTopItemsCallback
    с top_k = размер каталога, Trainer(inference_mode=True).
    """
    padding = catalog_size
    schema = TensorSchema(
        TensorFeatureInfo(
            "item_id",
            is_seq=True,
            padding_value=padding,
            cardinality=catalog_size,
            embedding_dim=e05.EMBEDDING_DIM,
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.ITEM_ID,
        )
    )
    transforms = make_default_sasrec_transforms(schema)
    model = SasRec.from_params(
        schema,
        embedding_dim=e05.EMBEDDING_DIM,
        num_heads=e05.NUM_HEADS,
        num_blocks=e05.NUM_BLOCKS,
        max_sequence_length=e05.MAX_SEQ_LEN,
        dropout=e05.DROPOUT,
    )
    module = LightningModule.load_from_checkpoint(str(ckpt_path), model=model)

    predict_callback = PandasTopItemsCallback(
        top_k=catalog_size, query_column="user_id", item_column="item_id", rating_column="score"
    )
    predict_datamodule = ParquetModule(
        predict_path=str(predict_parquet),
        batch_size=e05.BATCH_SIZE,
        metadata={"predict": {"user_id": {}, "item_id": {"shape": e05.MAX_SEQ_LEN, "padding": padding}}},
        transforms=transforms,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
        inference_mode=True,
        callbacks=[predict_callback],
    )
    trainer.predict(module, datamodule=predict_datamodule, return_predictions=False)
    scores = predict_callback.get_result()
    return scores.sort_values(["user_id", "score"], ascending=[True, False], kind="stable")


def unfiltered_top10(scores: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """Топ-k по сырому скору БЕЗ seen-фильтра — протокол monitor RePlay.

    ComputeMetricsCallback в e05 не фильтрует уже оценённые items (установлено
    E10/E11: monitor 0.233444 воспроизводится именно без фильтра; в топ-10
    ~37% слотов занято seen), поэтому воспроизведение monitor — отдельная
    панель, а честная сила SASRec на next1 — фильтрованный топ-10.
    """
    top = scores.groupby("user_id", sort=False).head(k).reset_index(drop=True)
    top["rank"] = top.groupby("user_id", sort=False).cumcount() + 1
    return top


def sasrec_pool(train: pd.DataFrame, catalog_size: int, ckpt_path: Path | None, cache_name: str) -> tuple[pd.DataFrame, str]:
    """Топ-POOL unseen items по скорам SASRec: из кэша или инференсом чекпойнта."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if cache_path.is_file():
        log(f"SASRec-пул из кэша: {cache_path}")
        return pd.read_parquet(cache_path), str(cache_path)
    if ckpt_path is None:
        raise RuntimeError(f"нет кэша {cache_path} и не указан чекпойнт")

    predict_parquet = ckpt_path.parent.parent / "predict.parquet"
    log(f"инференс чекпойнта {ckpt_path} по {predict_parquet}...")
    scores = sasrec_scores_from_checkpoint(ckpt_path, predict_parquet, catalog_size)
    pool = e05.filter_seen_topk(scores, train, k=POOL)
    pool.to_parquet(cache_path, index=False)
    log(f"SASRec-пул закэширован: {cache_path} ({len(pool):,} строк)")
    return pool, str(ckpt_path)


# --------------------------------------------------------------------------
# Классика: пулы markov/ease и бленд
# --------------------------------------------------------------------------

def markov_pools(train: pd.DataFrame, decays: tuple[float, ...]) -> dict[str, pd.DataFrame]:
    """Пулы топ-POOL unseen markov(depth6, decay) на каждый decay.

    Матрица переходов строится один раз, decay меняет только веса строк
    контекста (linear combination в combine_rows).
    """
    items = np.sort(train["item_id"].unique())
    histories = sorted_histories(train)
    users, context = last_context(histories, items, k=MARKOV_DEPTH)
    log("матрица markov (alpha=0.05)...")
    rows = context_score_rows(context[:, :MARKOV_DEPTH], markov_weights(train, items, alpha=MARKOV_ALPHA))
    pools = {}
    for decay in decays:
        pools[f"markov_{decay}"] = topk_from_scores(
            combine_rows(rows, MARKOV_DEPTH, decay), users, items, train, POOL
        )
        log(f"пул markov depth{MARKOV_DEPTH}/decay{decay}: готов")
    return pools


def ease_pool(train: pd.DataFrame) -> pd.DataFrame:
    """Пул топ-POOL unseen ease-last5 (λ=1500, скоринг по последним 5 событиям)."""
    items = np.sort(train["item_id"].unique())
    histories = sorted_histories(train)
    users, context = last_context(histories, items, k=MARKOV_DEPTH)
    log("матрица EASE (l2=1500)...")
    weights = ease_weights(train, items, l2=EASE_L2)
    scores = combine_rows(context_score_rows(context[:, :EASE_LAST_K], weights), EASE_LAST_K, 1.0)
    return topk_from_scores(scores, users, items, train, POOL)


def blend_frame(pools: dict[str, pd.DataFrame], pool_size: int = POOL) -> pd.DataFrame:
    """Outer-merge нормированных рангов 1 − (rank − 1)/pool всех моделей.

    Эквивалент groupby-sum взвешенных рангов rank_blend из E09: item, которого
    нет в пуле модели, даёт нулевой вклад (NaN -> 0).
    """
    frames = []
    for name, recs in pools.items():
        top = recs[recs["rank"] <= pool_size][["user_id", "item_id", "rank"]].copy()
        top[name] = 1.0 - (top["rank"] - 1) / pool_size
        frames.append(top[["user_id", "item_id", name]])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["user_id", "item_id"], how="outer")
    merged[list(pools)] = merged[list(pools)].fillna(0.0)
    return merged


def blend_top_k(merged: pd.DataFrame, weights: dict[str, float], k: int = TOP_K) -> pd.DataFrame:
    """Топ-k по сумме вес*нормранг; тай-брейк по item_id (детерминированность)."""
    score = None
    for name, weight in weights.items():
        score = merged[name] * weight if score is None else score + merged[name] * weight
    out = merged[["user_id", "item_id"]].copy()
    out["score"] = score
    out = out.sort_values(["user_id", "score", "item_id"], ascending=[True, False, True], kind="stable")
    top = out.groupby("user_id", sort=False).head(k).reset_index(drop=True)
    top["rank"] = top.groupby("user_id", sort=False).cumcount() + 1
    return top


def rank_blend_reference(
    recs_by_model: dict[str, pd.DataFrame], weights: dict[str, float],
    pool: int = POOL, k: int = TOP_K,
) -> pd.DataFrame:
    """Верbatim-порт rank_blend из e09 — эталон для проверки blend_top_k."""
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
    return combined.loc[combined["rank"] <= k, ["user_id", "item_id", "weighted", "rank"]].rename(
        columns={"weighted": "score"}
    )


def assert_blend_equivalence(
    pools: dict[str, pd.DataFrame], weights: dict[str, float], gt: pd.DataFrame
) -> None:
    """Диагностическая сверка путей бленда (один раз на пул-набор).

    Побитовое совпадение невозможно: pandas groupby-sum и последовательное
    сложение дают разницу в последних битах (1.7 vs 1.7000000000000002),
    которая переставляет exact-тайлы местами — на границе топ-10 это стоит
    ~1 хита из 6040. Канонической для E10 считается быстрая арифметика
    (blend_frame + blend_top_k, единая в сетке, панелях и сабмите);
    расхождение с эталоном E09 больше 0.001 — признак настоящего бага.
    """
    fast = blend_top_k(blend_frame(pools), weights)
    reference = rank_blend_reference(pools, weights)
    hr_fast = fast_hit_rate(fast, gt)
    hr_ref = fast_hit_rate(reference, gt)
    diff = abs(hr_fast - hr_ref)
    log(f"сверка путей бленда: fast={hr_fast:.6f}, e09-reference={hr_ref:.6f}, diff={diff:.6f}")
    if diff > 0.001:
        raise RuntimeError("blend_top_k разошёлся с rank_blend E09 существеннее tie-шума")


def fast_hit_rate(top: pd.DataFrame, gt: pd.DataFrame) -> float:
    """HR@10 по одиночному gt (|gt| = 1): доля пользователей с gt в топ-10.

    Только для перебора сетки весов; все сохраняемые панели — через evaluate.
    """
    gt_map = dict(zip(gt["user_id"].to_numpy(), gt["item_id"].to_numpy()))
    top10 = top[top["rank"] <= TOP_K]
    pairs = set(zip(top10["user_id"].to_numpy(), top10["item_id"].to_numpy()))
    hits = sum(1 for user, item in gt_map.items() if (user, item) in pairs)
    return hits / len(gt_map)


def run_weight_grid(sasrec: pd.DataFrame, classic: dict[str, pd.DataFrame], gt: pd.DataFrame) -> list[dict]:
    """Сетка (a, m, e) × markov-decay; возвращает все точки с HR@10."""
    rows = []
    markov_names = [name for name in classic if name.startswith("markov")]
    for name in markov_names:
        decay = float(name.split("_")[1])
        pools = {"sasrec": sasrec, "markov": classic[name], "ease": classic["ease"]}
        merged = blend_frame(pools)
        assert_blend_equivalence(
            pools, {"sasrec": GRID_A[0], "markov": GRID_M[0], "ease": GRID_E[0]}, gt
        )
        for a in GRID_A:
            for m in GRID_M:
                for e in GRID_E:
                    weights = {"sasrec": a, "markov": m, "ease": e}
                    hr = fast_hit_rate(blend_top_k(merged, weights), gt)
                    rows.append({"decay": decay, "a": a, "m": m, "e": e, "hr10": hr})
        log(f"сетка весов на markov decay={decay}: готова ({len(rows)} точек накоплено)")
    return rows


def solo_panel(pool: pd.DataFrame, gt: pd.DataFrame, train: pd.DataFrame) -> dict:
    panel = evaluate(pool[pool["rank"] <= TOP_K], gt, train_events=train)
    check_invariants(panel, "solo")
    return panel_to_dict(panel)


def blend_panel(pools: dict[str, pd.DataFrame], weights: dict[str, float],
                gt: pd.DataFrame, train: pd.DataFrame) -> dict:
    blend = blend_top_k(blend_frame(pools), weights)
    panel = evaluate(blend, gt, train_events=train)
    check_invariants(panel, "blend")
    return panel_to_dict(panel)


def _top_sets(frame: pd.DataFrame, k: int) -> dict[int, set[int]]:
    sub = frame[frame["rank"] <= k]
    sets: dict[int, set[int]] = {}
    for user, item in zip(sub["user_id"].to_numpy(), sub["item_id"].to_numpy()):
        sets.setdefault(int(user), set()).add(int(item))
    return sets


# --------------------------------------------------------------------------
# Стадии
# --------------------------------------------------------------------------

def stage_check() -> None:
    """Детерминизм сплитов: parquet чекпойнтов == свежие chronological_split."""
    events = load_events(DATA_DIR)
    train_9010, _ = chronological_split(events, train_fraction=0.9)
    inner_train, _ = chronological_split(train_9010, train_fraction=0.9)

    for level, train in (("outer", train_9010), ("inner", inner_train)):
        cached = pd.read_parquet(CKPT_DIR / level / "predict.parquet")
        fresh = e05.build_sequences(train)
        same_users = cached["user_id"].tolist() == fresh["user_id"].tolist()
        same_items = all(
            np.array_equal(np.asarray(a, dtype=int), np.asarray(b, dtype=int))
            for a, b in zip(cached["item_id"], fresh["item_id"])
        )
        same_ts = all(
            np.array_equal(np.asarray(a, dtype=int), np.asarray(b, dtype=int))
            for a, b in zip(cached["timestamp"], fresh["timestamp"])
        )
        log(f"{level}: users={same_users}, items={same_items}, timestamps={same_ts} "
            f"({len(cached)} последовательностей)")
        if not (same_users and same_items and same_ts):
            raise RuntimeError(f"{level}: сплит не совпал с parquet чекпойнта — инференс был бы по чужим данным")


def stage_control(state: dict) -> None:
    """Контроль: инференс outer воспроизводит monitor 0.233444 на next1@9010.

    Monitor e05c (ComputeMetricsCallback) считает recall@10 БЕЗ seen-фильтра,
    поэтому контрольное условие — нефильтрованный топ-10 (ожидание 0.2334).
    Фильтрованный топ-10 (ожидание ~0.319) — честная сила SASRec на next1 и
    база честной планки сабмита. Переобучение — только если monitor не
    воспроизводится без фильтра (mismatch чекпойнта/сплита).
    """
    events = load_events(DATA_DIR)
    catalog_size = e05.item_catalog(events)
    train_9010, valid_9010 = chronological_split(events, train_fraction=0.9)
    gt = first_valid_event(valid_9010)

    candidates = [CKPT_DIR / "outer" / "checkpoints" / "best.ckpt",
                  CKPT_DIR / "outer" / "checkpoints" / "best-v1.ckpt"]
    control: dict | None = None
    for ckpt in candidates:
        if not ckpt.is_file():
            continue
        predict_parquet = ckpt.parent.parent / "predict.parquet"
        log(f"контроль: инференс {ckpt}...")
        scores = sasrec_scores_from_checkpoint(ckpt, predict_parquet, catalog_size)

        raw_panel = evaluate(unfiltered_top10(scores), gt, train_events=train_9010)
        raw_value = raw_panel["recall@10_macro_relevant"]
        pool = e05.filter_seen_topk(scores, train_9010, k=POOL)
        panel = evaluate(pool[pool["rank"] <= TOP_K], gt, train_events=train_9010)
        check_invariants(panel, "control")
        value = panel["recall@10_macro_relevant"]
        log(f"контроль {ckpt.name}: без фильтра {raw_value:.6f} (monitor {MONITOR_BASELINE}, "
            f"diff {raw_value - MONITOR_BASELINE:+.6f}); с фильтром {value:.6f} "
            f"(seen в нефильтрованном топ-10: {raw_panel['seen_in_top_share']:.3f})")
        control = {
            "checkpoint": str(ckpt),
            "monitor_reproduction_unfiltered": float(raw_value),
            "diff_vs_monitor_unfiltered": float(raw_value - MONITOR_BASELINE),
            "filtered_recall": float(value),
            "within_tolerance": bool(abs(raw_value - MONITOR_BASELINE) <= CONTROL_TOLERANCE),
            "unfiltered_panel": panel_to_dict(raw_panel),
            "filtered_panel": panel_to_dict(panel),
            "note": "monitor RePlay не фильтрует seen: 0.2334 — unfiltered, "
                    "~0.319 — фильтрованная сила SASRec (установка E10/E11)",
        }
        if abs(raw_value - MONITOR_BASELINE) <= CONTROL_TOLERANCE:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            pool.to_parquet(CACHE_DIR / "sasrec_outer_top200.parquet", index=False)
            log("контроль пройден (monitor воспроизведён без фильтра), пул закэширован")
            break
    if control is None or not control["within_tolerance"]:
        raise RuntimeError(
            "monitor не воспроизведён без фильтра ни одним чекпойнтом — "
            "чекпойнт/сплит не совпадают с e05c, разбираться вручную"
        )
    state["meta"] = {
        "sasrec_config": "e128_len150_ep40 (dim=128, len=150, heads=4, blocks=2, dropout=0.3, ep=40, seed=42)",
        "monitor_baseline": MONITOR_BASELINE,
    }
    state["control"] = control
    save_state(state)


def stage_inner(state: dict) -> None:
    """Inner-уровень: инференс inner-чекпойнта + сетка весов (a, m, e) × decay."""
    events = load_events(DATA_DIR)
    catalog_size = e05.item_catalog(events)
    train_9010, _ = chronological_split(events, train_fraction=0.9)
    inner_train, inner_valid = chronological_split(train_9010, train_fraction=0.9)
    gt = first_valid_event(inner_valid)

    sasrec, source = sasrec_pool(
        inner_train, catalog_size, CKPT_DIR / "inner" / "checkpoints" / "best.ckpt",
        "sasrec_inner_top200.parquet",
    )
    classic: dict[str, pd.DataFrame] = markov_pools(inner_train, GRID_DECAY)
    classic["ease"] = ease_pool(inner_train)

    solo = {"sasrec": solo_panel(sasrec, gt, inner_train)}
    for name, pool in classic.items():
        solo[name] = solo_panel(pool, gt, inner_train)
        log(f"[inner] {name} соло: {solo[name]['recall@10_macro_relevant']:.4f}")

    grid = run_weight_grid(sasrec, classic, gt)
    best = max(grid, key=lambda row: row["hr10"])
    best_at_primary = max((r for r in grid if r["decay"] == PRIMARY_DECAY), key=lambda row: row["hr10"])
    log(f"[inner] лучший бленд: decay={best['decay']}, a={best['a']}, m={best['m']}, e={best['e']} "
        f"-> HR@10={best['hr10']:.4f}")
    log(f"[inner] лучший при decay={PRIMARY_DECAY}: a={best_at_primary['a']}, m={best_at_primary['m']}, "
        f"e={best_at_primary['e']} -> HR@10={best_at_primary['hr10']:.4f}")

    def weights_of(row: dict) -> dict[str, float]:
        return {"sasrec": row["a"], "markov": row["m"], "ease": row["e"]}

    best_panel = blend_panel(
        {"sasrec": sasrec, "markov": classic[f"markov_{best['decay']}"], "ease": classic["ease"]},
        weights_of(best), gt, inner_train,
    )
    primary_panel = blend_panel(
        {"sasrec": sasrec, "markov": classic[f"markov_{PRIMARY_DECAY}"], "ease": classic["ease"]},
        weights_of(best_at_primary), gt, inner_train,
    )

    state["inner"] = {
        "train_events": len(inner_train), "checkpoint": source,
        "solo": solo, "grid_best": best, "grid_best_at_primary_decay": best_at_primary,
        "grid": grid, "best_blend_panel": best_panel,
        "best_blend_panel_primary_decay": primary_panel,
    }
    save_state(state)


def stage_outer(state: dict) -> None:
    """Внешняя честная оценка next1@9010: фиксированные веса с inner."""
    if "inner" not in state:
        raise RuntimeError("сначала --stage inner")
    events = load_events(DATA_DIR)
    catalog_size = e05.item_catalog(events)
    train_9010, valid_9010 = chronological_split(events, train_fraction=0.9)
    gt = first_valid_event(valid_9010)

    sasrec, source = sasrec_pool(train_9010, catalog_size, None, "sasrec_outer_top200.parquet")
    classic: dict[str, pd.DataFrame] = markov_pools(train_9010, GRID_DECAY)
    classic["ease"] = ease_pool(train_9010)

    best_inner = state["inner"]["grid_best"]
    best_at_primary = state["inner"]["grid_best_at_primary_decay"]

    # (1) главный честный бленд: markov d6/decay0.6 (конфиг E09), веса с inner
    # (шаг inner на том же decay — согласованная пара конфиг+веса)
    primary_weights = {"sasrec": best_at_primary["a"], "markov": best_at_primary["m"], "ease": best_at_primary["e"]}
    primary = blend_panel(
        {"sasrec": sasrec, "markov": classic[f"markov_{PRIMARY_DECAY}"], "ease": classic["ease"]},
        primary_weights, gt, train_9010,
    )
    log(f"[outer] ЧЕСТНЫЙ бленд (markov d6/{PRIMARY_DECAY}, веса с inner): "
        f"{primary['recall@10_macro_relevant']:.4f}")

    # (2) полностью замороженный inner-конфиг (если inner выбрал другой decay)
    frozen_weights = {"sasrec": best_inner["a"], "markov": best_inner["m"], "ease": best_inner["e"]}
    frozen = blend_panel(
        {"sasrec": sasrec, "markov": classic[f'markov_{best_inner["decay"]}'], "ease": classic["ease"]},
        frozen_weights, gt, train_9010,
    )
    log(f"[outer] замороженный inner-конфиг (decay={best_inner['decay']}): "
        f"{frozen['recall@10_macro_relevant']:.4f}")

    # (3) диагностический потолок: лучший подбор весов на самом outer
    grid = run_weight_grid(sasrec, classic, gt)
    ceiling = max(grid, key=lambda row: row["hr10"])
    ceiling_weights = {"sasrec": ceiling["a"], "markov": ceiling["m"], "ease": ceiling["e"]}
    ceiling_panel = blend_panel(
        {"sasrec": sasrec, "markov": classic[f'markov_{ceiling["decay"]}'], "ease": classic["ease"]},
        ceiling_weights, gt, train_9010,
    )
    log(f"[outer] потолок (подбор на outer): {ceiling_panel['recall@10_macro_relevant']:.4f} "
        f"(decay={ceiling['decay']}, a={ceiling['a']}, m={ceiling['m']}, e={ceiling['e']})")

    # соло-панели + кросс-проверка против чисел E09 на том же протоколе
    solo: dict[str, dict] = {"sasrec": solo_panel(sasrec, gt, train_9010)}
    for name, pool in classic.items():
        solo[name] = solo_panel(pool, gt, train_9010)
    solo["markov_d6_0.6"] = solo[f"markov_{PRIMARY_DECAY}"]
    solo["ease_last5"] = solo["ease"]
    solo["mk_ease_4_1"] = blend_panel(
        {"markov": classic[f"markov_{PRIMARY_DECAY}"], "ease": classic["ease"]},
        {"markov": 4.0, "ease": 1.0}, gt, train_9010,
    )
    for name, ref in E09_REF["next1@9010"].items():
        got = solo[name]["recall@10_macro_relevant"]
        log(f"[outer] кросс-чек E09 {name}: {got:.4f} против {ref} (diff {got - ref:+.4f})")

    state["outer"] = {
        "train_events": len(train_9010), "sasrec_source": source, "gt": "first event of valid_9010",
        "primary_fixed_weights": {**primary_weights, "markov_decay": PRIMARY_DECAY, "markov_depth": MARKOV_DEPTH},
        "primary_panel": primary,
        "frozen_inner_weights": {**frozen_weights, "markov_decay": best_inner["decay"]},
        "frozen_inner_panel": frozen,
        "ceiling_weights": {**ceiling_weights, "markov_decay": ceiling["decay"]},
        "ceiling_panel": ceiling_panel, "grid_ceiling": grid, "solo": solo,
    }

    # overlap-диагностика комплементарности
    state["overlap"] = overlap_diagnostics(sasrec, classic, gt)
    log(f"[outer] overlap: {state['overlap']}")

    honest = max(primary["recall@10_macro_relevant"], frozen["recall@10_macro_relevant"])
    filtered_sasrec = state["control"]["filtered_recall"]
    honest_threshold = filtered_sasrec * 1.02
    state["criterion"] = {
        "best_honest_blend": float(honest),
        "filtered_sasrec_baseline": float(filtered_sasrec),
        "honest_threshold": float(honest_threshold),
        "honest_beaten": bool(honest >= honest_threshold),
        "legacy_threshold": CRITERION_LEGACY,
        "legacy_beaten": bool(honest >= CRITERION_LEGACY),
        "submit_recommended": bool(honest >= honest_threshold),
        "note": "monitor*1.02 (0.2381) — legacy-планка от нефильтрованного monitor; "
                "честная планка = фильтрованный SASRec 0.319 * 1.02 ~= 0.3254: "
                "сравнение фильтрованного бленда с нефильтрованным monitor некорректно "
                "(переинтерпретация E10 после установки, что RePlay-monitor не фильтрует seen)",
    }
    log(f"[outer] критерий: честный бленд {honest:.4f}; legacy-планка {CRITERION_LEGACY:.4f} "
        f"({'+пройдена' if honest >= CRITERION_LEGACY else 'не пройдена'}); честная планка "
        f"{honest_threshold:.4f} = фильтрованный SASRec {filtered_sasrec:.4f} x1.02 "
        f"({'+пройдена — СОБИРАТЬ САБМИТ' if honest >= honest_threshold else 'не пройдена — сабмит не собираем'})")
    save_state(state)


def overlap_diagnostics(sasrec: pd.DataFrame, classic: dict[str, pd.DataFrame], gt: pd.DataFrame) -> dict:
    """Комплементарность SASRec и markov на next1@9010.

    (а) доля пользователей, у которых gt НЕ в топ-10 SASRec, но ЕСТЬ в топ-10
        бленда markov+ease (4:1, без SASRec); (б) пересечение топ-30 кандидатов
        SASRec и markov: средний Jaccard и средняя доля совпадения |∩|/30.
    """
    gt_map = dict(zip(gt["user_id"].to_numpy(), gt["item_id"].to_numpy()))
    me_blend = blend_top_k(
        blend_frame({"markov": classic[f"markov_{PRIMARY_DECAY}"], "ease": classic["ease"]}),
        {"markov": 4.0, "ease": 1.0},
    )
    s10 = _top_sets(sasrec, TOP_K)
    me10 = _top_sets(me_blend, TOP_K)
    missed = [u for u, item in gt_map.items() if item not in s10.get(u, set())]
    recovered = [u for u in missed if gt_map[u] in me10.get(u, set())]

    s30 = _top_sets(sasrec, 30)
    m30 = _top_sets(classic[f"markov_{PRIMARY_DECAY}"], 30)
    jaccards, overlaps = [], []
    for user, s_items in s30.items():
        m_items = m30.get(user, set())
        union = s_items | m_items
        inter = len(s_items & m_items)
        jaccards.append(inter / len(union) if union else 1.0)
        overlaps.append(inter / 30.0)

    return {
        "users_missed_by_sasrec": len(missed),
        "missed_recovered_by_markov_ease": len(recovered),
        "missed_recovered_share": float(len(recovered) / len(missed)) if missed else 0.0,
        "top30_jaccard_mean": float(np.mean(jaccards)),
        "top30_overlap_share_mean": float(np.mean(overlaps)),
        "note": "markov = depth6/decay0.6, mk+ease бленд 4:1 (конфиг E09)",
    }


def stage_sasrec_last1(state: dict) -> None:
    """Обучение SASRec e128/40эп на train_last1 с monitor (~10-30 мин CPU)."""
    events = load_events(DATA_DIR)
    catalog_size = e05.item_catalog(events)
    train_last1, valid_last1 = last_n_split(events, n=1)
    gt = valid_last1[["user_id", "item_id"]]

    work = WORK_DIR / "sasrec_last1"
    log(f"обучение SASRec last1 на {len(train_last1):,} событий, monitor = последнее событие...")
    scores = e05.fit_predict_sasrec(train_last1, valid_last1, catalog_size, work, epochs=SASREC_EPOCHS)
    pool = e05.filter_seen_topk(scores, train_last1, k=POOL)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(CACHE_DIR / "sasrec_last1_top200.parquet", index=False)
    panel = evaluate(pool[pool["rank"] <= TOP_K], gt, train_events=train_last1)
    check_invariants(panel, "sasrec-last1")
    log(f"[last1] контроль SASRec-only: {panel['recall@10_macro_relevant']:.4f}")
    state["last1_sasrec"] = {
        "checkpoint": str(work / "checkpoints" / "best.ckpt"),
        "sasrec_only_panel": panel_to_dict(panel),
    }
    save_state(state)


def stage_last1(state: dict) -> None:
    """last1: бленд с фиксированными весами + контроль SASRec-only."""
    cache = CACHE_DIR / "sasrec_last1_top200.parquet"
    if not cache.is_file():
        raise RuntimeError("сначала --stage sasrec-last1 (обучение ~10-30 мин)")
    if "inner" not in state:
        raise RuntimeError("сначала --stage inner")
    events = load_events(DATA_DIR)
    train_last1, valid_last1 = last_n_split(events, n=1)
    gt = valid_last1[["user_id", "item_id"]]

    sasrec = pd.read_parquet(cache)
    classic: dict[str, pd.DataFrame] = markov_pools(train_last1, GRID_DECAY)
    classic["ease"] = ease_pool(train_last1)

    best_at_primary = state["inner"]["grid_best_at_primary_decay"]
    weights = {"sasrec": best_at_primary["a"], "markov": best_at_primary["m"], "ease": best_at_primary["e"]}
    primary = blend_panel(
        {"sasrec": sasrec, "markov": classic[f"markov_{PRIMARY_DECAY}"], "ease": classic["ease"]},
        weights, gt, train_last1,
    )
    log(f"[last1] бленд (markov d6/{PRIMARY_DECAY}, веса с inner): {primary['recall@10_macro_relevant']:.4f}")

    best_inner = state["inner"]["grid_best"]
    frozen = blend_panel(
        {"sasrec": sasrec, "markov": classic[f'markov_{best_inner["decay"]}'], "ease": classic["ease"]},
        {"sasrec": best_inner["a"], "markov": best_inner["m"], "ease": best_inner["e"]},
        gt, train_last1,
    )
    log(f"[last1] замороженный inner-конфиг (decay={best_inner['decay']}): "
        f"{frozen['recall@10_macro_relevant']:.4f}")

    solo: dict[str, dict] = {"sasrec": solo_panel(sasrec, gt, train_last1)}
    for name, pool in classic.items():
        solo[name] = solo_panel(pool, gt, train_last1)
    solo["markov_d6_0.7"] = solo["markov_0.7"]
    solo["ease_last5"] = solo["ease"]
    solo["mk_ease_4_1_d0.7"] = blend_panel(
        {"markov": classic["markov_0.7"], "ease": classic["ease"]},
        {"markov": 4.0, "ease": 1.0}, gt, train_last1,
    )
    for name, ref in E09_REF["last1"].items():
        got = solo[name]["recall@10_macro_relevant"]
        log(f"[last1] кросс-чек E09 {name}: {got:.4f} против {ref} (diff {got - ref:+.4f})")

    state["last1"] = {
        "train_events": len(train_last1), "weights": weights, "markov_decay": PRIMARY_DECAY,
        "primary_panel": primary, "frozen_inner_panel": frozen, "solo": solo,
        "kaggle_multiplier_note": "last1 -> Kaggle ~ x0.93-1.0 (калибровка popularity/E09)",
    }
    save_state(state)


def stage_submit(state: dict) -> None:
    """Сабмит: markov/ease на полных events + SASRec e128/40эп с нуля + фикс-веса."""
    if "criterion" not in state or not state["criterion"]["submit_recommended"]:
        log("критерий не достигнут — сабмит не собираем")
        return
    events = load_events(DATA_DIR)
    catalog_size = e05.item_catalog(events)
    best_at_primary = state["inner"]["grid_best_at_primary_decay"]
    weights = {"sasrec": best_at_primary["a"], "markov": best_at_primary["m"], "ease": best_at_primary["e"]}

    classic: dict[str, pd.DataFrame] = markov_pools(events, (PRIMARY_DECAY,))
    classic["ease"] = ease_pool(events)

    work = WORK_DIR / "sasrec_full"
    cache = CACHE_DIR / "sasrec_full_top200.parquet"
    if cache.is_file():
        sasrec = pd.read_parquet(cache)
    else:
        log("обучение SASRec на полных events (40 эпох, без валидации, ~10-30 мин)...")
        scores = e05.fit_predict_sasrec(events, None, catalog_size, work, epochs=SASREC_EPOCHS)
        sasrec = e05.filter_seen_topk(scores, events, k=POOL)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sasrec.to_parquet(cache, index=False)

    blend = blend_top_k(
        blend_frame({"sasrec": sasrec, "markov": classic[f"markov_{PRIMARY_DECAY}"], "ease": classic["ease"]}),
        weights,
    )
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out = SUBMISSIONS_DIR / "E10_hybrid_blend.csv"
    save_submission(blend, out)
    log(f"сабмит собран: {out} ({blend['user_id'].nunique()} пользователей)")
    state["submission"] = {
        "path": str(out), "n_users": int(blend["user_id"].nunique()),
        "weights": weights, "markov_decay": PRIMARY_DECAY, "markov_depth": MARKOV_DEPTH,
        "sasrec_work_dir": str(work),
    }
    save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=(
        "check", "control", "inner", "outer", "sasrec-last1", "last1", "submit"))
    args = parser.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    stages = {
        "check": stage_check,
        "control": lambda: stage_control(state),
        "inner": lambda: stage_inner(state),
        "outer": lambda: stage_outer(state),
        "sasrec-last1": lambda: stage_sasrec_last1(state),
        "last1": lambda: stage_last1(state),
        "submit": lambda: stage_submit(state),
    }
    stages[args.stage]()


if __name__ == "__main__":
    main()
