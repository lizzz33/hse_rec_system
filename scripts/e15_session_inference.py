"""E15: сессионный инференс SASRec без обучения (короткое окно + бленд с полным).

H15a: урок E09 (сессионный скоринг поднимает классику в 2-3.3 раза) переносится
на трансформер на этапе ИНФЕРЕНСА: rank-average скоров одного чекпойнта при
полном окне (150) и коротком (последние 30/50 событий) подчеркнёт локальный
next-item сигнал (жанровые стрики, P=0.66) и даст +1-2% к фильтрованному
SASRec на next1@9010 (база 0.3190, e10 control; monitor 0.2334 — unfiltered,
RePlay ComputeMetricsCallback не фильтрует seen).

H15b (опция): второй инференс-проход с контекстом, дополненным top-1
предсказанием первого прохода (lookahead) — покрытие gt-позиций 2-3.

Механика короткого окна: тот же чекпойнт, но sequences в predict-parquet
обрезаются до последних N событий (пересборка через build_sequences + tail-
slice). RePlay-ридер левый-паддинг: короткая последовательность занимает те же
хвостовые позиции окна 150, что и в полном инференсе — модель видит ровно
«последние N событий», как при обучении у коротких пользователей.

Протокол весов (как в E10): подбор на inner-уровне (inner-чекпойнт, gt =
первое событие inner_valid, seen-фильтр по inner_train), фиксация лучшего,
оценка на outer next1@9010 (gt = первое событие valid_9010) и на last1
(last1-чекпойнт из e10_hybrid_blend_work, gt = последнее событие истории).
Бленд: rank-norm на пользователя (1 - (rank-1)/200) + среднее с весом w
короткого окна, сетка short_len {30, 50} x w {0.3, 0.5, 0.7}.

Ничего не обучаем; чекпойнты читаются из e05_sasrec_work/e07 и
e10_hybrid_blend_work (только чтение), свой ворк-каталог results/e15_work/.

Стадии:
    check        — детерминизм: predict.parquet чекпойнтов совпадает со свежими
                   сплитами; целостность переиспользуемого e10-пула outer;
    inner        — 3 инференса inner-чекпойнта (150/30/50) + соло-панели +
                   сетка (short_len x w) -> фиксация лучшего конфига;
    outer        — пулы outer-чекпойнта (полный окно — из кэша e10 control,
                   короткие — инференсом), соло-панели, фиксированный бленд,
                   диагностический потолок (подбор на outer), артефакт top-200;
    last1        — 3 инференса last1-чекпойнта, соло-панели (база = соло 150),
                   фиксированный бленд, артефакт top-200;
    lookahead    — H15b (опция): второй проход с top-1 первого в контексте.

Запуск:
    uv run python scripts/e15_session_inference.py --stage check
    uv run python scripts/e15_session_inference.py --stage inner   # и далее
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

# до импорта torch/lightning: ограничиваем BLAS-потоки (паттерн e05/e10;
# на машине параллельно идёт другой инференс — torch держим на 4 потоках)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# e10 импортирует и настраивает e05 под конфиг чекпойнтов e128/len150/ep40
# (EMBEDDING_DIM=128, MAX_SEQ_LEN=150, NUM_HEADS=4) и содержит все нужные
# функции инференса/бленда — переиспользуем, не копируем
_spec_e10 = importlib.util.spec_from_file_location("e10_hybrid_blend", PROJECT_ROOT / "scripts" / "e10_hybrid_blend.py")
e10 = importlib.util.module_from_spec(_spec_e10)
_spec_e10.loader.exec_module(e10)
e05 = e10.e05

import numpy as np
import pandas as pd

from recsys.experiments import DATA_DIR, RESULTS_DIR
from recsys.split import chronological_split, last_n_split

WORK_DIR = RESULTS_DIR / "e15_work"
STATE_PATH = RESULTS_DIR / "e15_session_inference.json"
CKPT_DIR = RESULTS_DIR / "e05_sasrec_work" / "e07" / "e128_len150_ep40"
LAST1_DIR = RESULTS_DIR / "e10_hybrid_blend_work" / "sasrec_last1"
E10_STATE_PATH = RESULTS_DIR / "e10_hybrid_blend.json"
E10_OUTER_POOL = RESULTS_DIR / "e10_hybrid_blend_work" / "cache" / "sasrec_outer_top200.parquet"

SHORT_LENS = (30, 50)
GRID_W = (0.3, 0.5, 0.7)  # вес короткого окна в бленде
POOL = 200
TOP_K = 10

# базы e10 (control stage): фильтрованный SASRec next1@9010 и monitor-unfiltered
BASE_FILTERED_OUTER = 0.31903973509933775
BASE_UNFILTERED_OUTER = 0.23344370860927152
CRITERION_GAIN = 0.01  # H15a: +1% к фильтрованному SASRec -> компонента E17


def log(message: str) -> None:
    e10.log(message)


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    log(f"результаты: {STATE_PATH}")


# --------------------------------------------------------------------------
# Уровни протокола: train / gt / чекпойнт / parquet полного окна
# --------------------------------------------------------------------------

def level_setup(level: str, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """(train, gt, ckpt, predict_parquet_полного_окна) для уровня протокола."""
    if level == "inner":
        train_9010, _ = chronological_split(events, train_fraction=0.9)
        inner_train, inner_valid = chronological_split(train_9010, train_fraction=0.9)
        ckpt_dir = CKPT_DIR / "inner"
        return inner_train, e10.first_valid_event(inner_valid), ckpt_dir / "checkpoints" / "best.ckpt", ckpt_dir / "predict.parquet"
    if level == "outer":
        train_9010, valid_9010 = chronological_split(events, train_fraction=0.9)
        ckpt_dir = CKPT_DIR / "outer"
        return train_9010, e10.first_valid_event(valid_9010), ckpt_dir / "checkpoints" / "best.ckpt", ckpt_dir / "predict.parquet"
    if level == "last1":
        train_last1, valid_last1 = last_n_split(events, n=1)
        return train_last1, valid_last1[["user_id", "item_id"]], LAST1_DIR / "checkpoints" / "best.ckpt", LAST1_DIR / "predict.parquet"
    raise ValueError(f"неизвестный уровень: {level}")


def build_short_predict(train: pd.DataFrame, short_len: int, out_path: Path) -> Path:
    """Predict-parquet с последовательностями, обрезанными до последних short_len событий."""
    if out_path.is_file():
        return out_path
    sequences = e05.build_sequences(train)
    short = pd.DataFrame({
        "user_id": sequences["user_id"],
        "item_id": [np.asarray(row, dtype=int)[-short_len:].tolist() for row in sequences["item_id"]],
        "timestamp": [np.asarray(row, dtype=int)[-short_len:].tolist() for row in sequences["timestamp"]],
    })
    short.to_parquet(out_path, index=False)
    log(f"короткий predict-parquet (последние {short_len}): {out_path} ({len(short)} последовательностей)")
    return out_path


def window_pool(
    level: str, short_len: int, train: pd.DataFrame, gt: pd.DataFrame,
    ckpt: Path, full_predict: Path, catalog_size: int,
) -> tuple[pd.DataFrame, dict | None, str]:
    """Пул топ-POOL unseen SASRec для окна short_len (150 = полное).

    Возвращает (пул, unfiltered-панель | None, источник). Пул и панель
    кэшируются в results/e15_work/pool_<level>_len<N>.parquet; при повторном
    запуске пул берётся из кэша, unfiltered-панель — из state (если есть).
    """
    cache = WORK_DIR / f"pool_{level}_len{short_len}.parquet"
    state = load_state()
    cached_panel = (state.get(level, {}).get("solo", {}).get(f"len{short_len}", {}) or {}).get("unfiltered_panel")
    if cache.is_file():
        return pd.read_parquet(cache), cached_panel, str(cache)

    if short_len == e05.MAX_SEQ_LEN:
        predict_parquet = full_predict
    else:
        predict_parquet = build_short_predict(train, short_len, WORK_DIR / f"predict_{level}_len{short_len}.parquet")
    log(f"[{level}] инференс окно {short_len}: {ckpt} по {predict_parquet}...")
    scores = e10.sasrec_scores_from_checkpoint(ckpt, predict_parquet, catalog_size)
    unfiltered = e10.panel_to_dict(e10.evaluate(e10.unfiltered_top10(scores), gt, train_events=train))
    pool = e05.filter_seen_topk(scores, train, k=POOL)
    pool.to_parquet(cache, index=False)
    log(f"[{level}] окно {short_len}: пул {len(pool):,} строк -> {cache}")
    return pool, unfiltered, str(ckpt)


# --------------------------------------------------------------------------
# Сетки и бленды (пулы именуются full/short — функции e10 агностичны к именам)
# --------------------------------------------------------------------------

def run_session_grid(full: pd.DataFrame, shorts: dict[int, pd.DataFrame], gt: pd.DataFrame) -> list[dict]:
    """Сетка short_len x w: HR@10 бленда rank-norm full + short с весом w короткого."""
    rows = []
    for short_len, short in shorts.items():
        merged = e10.blend_frame({"full": full, "short": short})
        for w in GRID_W:
            weights = {"full": 1.0 - w, "short": w}
            hr = e10.fast_hit_rate(e10.blend_top_k(merged, weights), gt)
            rows.append({"short_len": int(short_len), "w_short": float(w), "hr10": float(hr)})
        log(f"сетка окно {short_len}: " + ", ".join(f"w={r['w_short']}: {r['hr10']:.4f}" for r in rows if r["short_len"] == short_len))
    return rows


def fixed_blend_panel(full: pd.DataFrame, short: pd.DataFrame, w: float,
                      gt: pd.DataFrame, train: pd.DataFrame) -> dict:
    return e10.blend_panel({"full": full, "short": short}, {"full": 1.0 - w, "short": w}, gt, train)


def save_blend_artifact(pools: dict[str, pd.DataFrame], weights: dict[str, float], path: Path) -> Path:
    """Топ-200 лучшего бленда (user_id, item_id, score, rank) — вход для E17."""
    blend = e10.rank_blend_reference(pools, weights, pool=POOL, k=POOL)
    blend.to_parquet(path, index=False)
    log(f"артефакт бленда top-{POOL}: {path} ({len(blend):,} строк)")
    return path


def overlap_diagnostics(full: pd.DataFrame, short: pd.DataFrame, gt: pd.DataFrame, short_len: int) -> dict:
    """Комплементарность окон: recovery промахов полного окна коротким + jaccard top-30."""
    gt_map = dict(zip(gt["user_id"].to_numpy(), gt["item_id"].to_numpy()))
    f10 = e10._top_sets(full, TOP_K)
    s10 = e10._top_sets(short, TOP_K)
    missed = [u for u, item in gt_map.items() if item not in f10.get(u, set())]
    recovered = [u for u in missed if gt_map[u] in s10.get(u, set())]
    f30 = e10._top_sets(full, 30)
    s30 = e10._top_sets(short, 30)
    jaccards = [
        len(s_items & s30.get(user, set())) / len(s_items | s30.get(user, set())) if s_items | s30.get(user, set()) else 1.0
        for user, s_items in f30.items()
    ]
    return {
        "short_len": int(short_len),
        "users_missed_by_full": len(missed),
        "missed_recovered_by_short": len(recovered),
        "missed_recovered_share": float(len(recovered) / len(missed)) if missed else 0.0,
        "top30_jaccard_mean": float(np.mean(jaccards)),
    }


# --------------------------------------------------------------------------
# Стадии
# --------------------------------------------------------------------------

def stage_check() -> None:
    """Детерминизм сплитов: parquet'ы чекпойнтов == свежие сплиты; кэш e10 цел."""
    events = e05.load_events(DATA_DIR)
    train_9010, _ = chronological_split(events, train_fraction=0.9)
    inner_train, _ = chronological_split(train_9010, train_fraction=0.9)
    train_last1, _ = last_n_split(events, n=1)

    for level, train, parquet in (
        ("inner", inner_train, CKPT_DIR / "inner" / "predict.parquet"),
        ("outer", train_9010, CKPT_DIR / "outer" / "predict.parquet"),
        ("last1", train_last1, LAST1_DIR / "predict.parquet"),
    ):
        cached = pd.read_parquet(parquet)
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
        log(f"{level}: users={same_users}, items={same_items}, timestamps={same_ts} ({len(cached)} последовательностей)")
        if not (same_users and same_items and same_ts):
            raise RuntimeError(f"{level}: parquet не совпал со свежим сплитом — инференс был бы по чужим данным")

    # переиспользуемый пул e10 (outer, полное окно): целостность + кросс-чек базы
    if E10_OUTER_POOL.is_file():
        pool = pd.read_parquet(E10_OUTER_POOL)
        _, gt_outer, _, _ = level_setup("outer", events)
        hr = e10.fast_hit_rate(pool[pool["rank"] <= TOP_K], gt_outer)
        n_users = pool["user_id"].nunique()
        ranks_ok = pool.groupby("user_id")["rank"].apply(lambda r: sorted(r) == list(range(1, POOL + 1))).all()
        seen = set(zip(train_9010["user_id"].to_numpy(), train_9010["item_id"].to_numpy()))
        seen_share = sum((u, i) in seen for u, i in zip(pool[pool["rank"] <= TOP_K]["user_id"].to_numpy(), pool[pool["rank"] <= TOP_K]["item_id"].to_numpy())) / (n_users * TOP_K)
        log(f"e10-пул outer: {n_users} пользователей, ranks_ok={ranks_ok}, seen_in_top={seen_share:.4f}, "
            f"HR@10={hr:.6f} (база e10 {BASE_FILTERED_OUTER:.6f}, diff {hr - BASE_FILTERED_OUTER:+.6f})")
        if n_users != 6040 or not ranks_ok or seen_share != 0.0 or abs(hr - BASE_FILTERED_OUTER) > 1e-6:
            raise RuntimeError("кэш e10 outer не прошёл целостность/кросс-чек — считать пул инференсом самому")
    else:
        log("кэш e10 outer не найден — полное окно outer будет посчитано инференсом")


def collect_level(level: str, events: pd.DataFrame, catalog_size: int,
                  reuse_e10_pool: bool = False) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[int, dict]]:
    """Пулы всех окон уровня + train + gt + unfiltered-панели."""
    train, gt, ckpt, full_predict = level_setup(level, events)
    pools: dict[int, pd.DataFrame] = {}
    unfiltered: dict[int, dict] = {}
    for short_len in (e05.MAX_SEQ_LEN, *SHORT_LENS):
        if reuse_e10_pool and short_len == e05.MAX_SEQ_LEN and E10_OUTER_POOL.is_file():
            pool = pd.read_parquet(E10_OUTER_POOL)
            e10_state = json.loads(E10_STATE_PATH.read_text()) if E10_STATE_PATH.is_file() else {}
            panel = (e10_state.get("control", {}).get("unfiltered_panel"))
            log(f"[{level}] полное окно: пул из кэша e10 control {E10_OUTER_POOL}")
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            pool.to_parquet(WORK_DIR / f"pool_{level}_len{short_len}.parquet", index=False)
            pools[short_len] = pool
            unfiltered[short_len] = panel
            continue
        pool, panel, source = window_pool(level, short_len, train, gt, ckpt, full_predict, catalog_size)
        pools[short_len] = pool
        if panel is not None:
            unfiltered[short_len] = panel
    return pools, train, gt, unfiltered


def stage_inner(state: dict) -> None:
    """Inner-уровень: инференсы 150/30/50 + сетка (short_len x w) -> фиксация."""
    events = e05.load_events(DATA_DIR)
    catalog_size = e05.item_catalog(events)
    pools, train, gt, unfiltered = collect_level("inner", events, catalog_size)

    solo = {}
    for short_len, pool in pools.items():
        solo[f"len{short_len}"] = {
            "filtered_panel": e10.solo_panel(pool, gt, train),
            **({"unfiltered_panel": unfiltered[short_len]} if short_len in unfiltered else {}),
        }
        log(f"[inner] соло окно {short_len}: filtered {solo[f'len{short_len}']['filtered_panel']['recall@10_macro_relevant']:.4f}"
            + (f", unfiltered {unfiltered[short_len]['recall@10_macro_relevant']:.4f}" if short_len in unfiltered else ""))

    shorts = {n: p for n, p in pools.items() if n != e05.MAX_SEQ_LEN}
    grid = run_session_grid(pools[e05.MAX_SEQ_LEN], shorts, gt)
    best = max(grid, key=lambda row: row["hr10"])
    best_panel = fixed_blend_panel(pools[e05.MAX_SEQ_LEN], pools[best["short_len"]], best["w_short"], gt, train)
    log(f"[inner] лучший конфиг: short_len={best['short_len']}, w_short={best['w_short']} "
        f"-> HR@10={best['hr10']:.4f}, panel={best_panel['recall@10_macro_relevant']:.4f}")

    state["inner"] = {
        "train_events": int(len(train)), "gt": "first event of inner_valid",
        "solo": solo, "grid": grid, "best": best, "best_panel": best_panel,
        "overlap": {str(n): overlap_diagnostics(pools[e05.MAX_SEQ_LEN], p, gt, n) for n, p in shorts.items()},
    }
    save_state(state)


def stage_outer(state: dict) -> None:
    """Outer next1@9010: фиксированный inner-конфиг + соло + потолок + артефакт."""
    if "inner" not in state:
        raise RuntimeError("сначала --stage inner")
    events = e05.load_events(DATA_DIR)
    catalog_size = e05.item_catalog(events)
    pools, train, gt, unfiltered = collect_level("outer", events, catalog_size, reuse_e10_pool=True)

    best = state["inner"]["best"]
    solo = {}
    for short_len, pool in pools.items():
        solo[f"len{short_len}"] = {
            "filtered_panel": e10.solo_panel(pool, gt, train),
            **({"unfiltered_panel": unfiltered[short_len]} if short_len in unfiltered else {}),
        }
        log(f"[outer] соло окно {short_len}: filtered {solo[f'len{short_len}']['filtered_panel']['recall@10_macro_relevant']:.4f}"
            + (f", unfiltered {unfiltered[short_len]['recall@10_macro_relevant']:.4f}" if short_len in unfiltered else ""))
    full_filtered = solo[f"len{e05.MAX_SEQ_LEN}"]["filtered_panel"]["recall@10_macro_relevant"]
    if abs(full_filtered - BASE_FILTERED_OUTER) > 0.002:
        raise RuntimeError(f"[outer] соло полного окна {full_filtered:.6f} разошлось с базой e10 {BASE_FILTERED_OUTER:.6f}")

    # главный честный результат: замороженный inner-конфиг на outer
    fixed = fixed_blend_panel(pools[e05.MAX_SEQ_LEN], pools[best["short_len"]], best["w_short"], gt, train)
    log(f"[outer] ФИКСИРОВАННЫЙ бленд (short_len={best['short_len']}, w={best['w_short']}): "
        f"{fixed['recall@10_macro_relevant']:.4f} против фильтрованного SASRec {full_filtered:.4f} "
        f"({(fixed['recall@10_macro_relevant'] / full_filtered - 1) * 100:+.2f}%)")

    # диагностический потолок: лучший подбор на самом outer
    shorts = {n: p for n, p in pools.items() if n != e05.MAX_SEQ_LEN}
    grid = run_session_grid(pools[e05.MAX_SEQ_LEN], shorts, gt)
    ceiling = max(grid, key=lambda row: row["hr10"])
    ceiling_panel = fixed_blend_panel(pools[e05.MAX_SEQ_LEN], pools[ceiling["short_len"]], ceiling["w_short"], gt, train)
    log(f"[outer] потолок (подбор на outer): {ceiling_panel['recall@10_macro_relevant']:.4f} "
        f"(short_len={ceiling['short_len']}, w={ceiling['w_short']})")

    artifact = save_blend_artifact(
        {"full": pools[e05.MAX_SEQ_LEN], "short": pools[best["short_len"]]},
        {"full": 1.0 - best["w_short"], "short": best["w_short"]},
        WORK_DIR / "e15_outer_best_blend_top200.parquet",
    )

    gain = fixed["recall@10_macro_relevant"] / full_filtered - 1.0
    state["outer"] = {
        "train_events": int(len(train)), "gt": "first event of valid_9010",
        "solo": solo, "fixed_config": best, "fixed_panel": fixed,
        "ceiling": {**ceiling, "panel": ceiling_panel}, "grid_ceiling": grid,
        "overlap": {str(n): overlap_diagnostics(pools[e05.MAX_SEQ_LEN], p, gt, n) for n, p in shorts.items()},
        "artifact_top200": str(artifact),
        "bases": {"filtered": BASE_FILTERED_OUTER, "unfiltered_monitor": BASE_UNFILTERED_OUTER},
    }
    state["criterion"] = {
        "h15a_gain_vs_filtered_sasrec": float(gain),
        "h15a_threshold": CRITERION_GAIN,
        "h15a_accepted": bool(gain >= CRITERION_GAIN),
        "note": "+1% к фильтрованному SASRec (0.3190) на next1@9010 -> компонента финального бленда E17",
    }
    log(f"[outer] критерий H15a: прирост {gain * 100:+.2f}% "
        f"({'ДОСТИГНУТ -> компонента E17' if gain >= CRITERION_GAIN else 'не достигнут'})")
    save_state(state)


def stage_last1(state: dict) -> None:
    """last1: фиксированный inner-конфиг на last1-чекпойнте; база = соло 150."""
    if "inner" not in state:
        raise RuntimeError("сначала --stage inner")
    events = e05.load_events(DATA_DIR)
    catalog_size = e05.item_catalog(events)
    pools, train, gt, unfiltered = collect_level("last1", events, catalog_size)

    best = state["inner"]["best"]
    solo = {}
    for short_len, pool in pools.items():
        solo[f"len{short_len}"] = {
            "filtered_panel": e10.solo_panel(pool, gt, train),
            **({"unfiltered_panel": unfiltered[short_len]} if short_len in unfiltered else {}),
        }
        log(f"[last1] соло окно {short_len}: filtered {solo[f'len{short_len}']['filtered_panel']['recall@10_macro_relevant']:.4f}"
            + (f", unfiltered {unfiltered[short_len]['recall@10_macro_relevant']:.4f}" if short_len in unfiltered else ""))
    base = solo[f"len{e05.MAX_SEQ_LEN}"]["filtered_panel"]["recall@10_macro_relevant"]

    fixed = fixed_blend_panel(pools[e05.MAX_SEQ_LEN], pools[best["short_len"]], best["w_short"], gt, train)
    log(f"[last1] ФИКСИРОВАННЫЙ бленд (short_len={best['short_len']}, w={best['w_short']}): "
        f"{fixed['recall@10_macro_relevant']:.4f} против соло 150 {base:.4f} "
        f"({(fixed['recall@10_macro_relevant'] / base - 1) * 100:+.2f}%)")

    # диагностический потолок на самом last1 (выбор конфига здесь смотрел бы в gt)
    shorts = {n: p for n, p in pools.items() if n != e05.MAX_SEQ_LEN}
    grid = run_session_grid(pools[e05.MAX_SEQ_LEN], shorts, gt)
    ceiling = max(grid, key=lambda row: row["hr10"])
    ceiling_panel = fixed_blend_panel(pools[e05.MAX_SEQ_LEN], pools[ceiling["short_len"]], ceiling["w_short"], gt, train)

    artifact = save_blend_artifact(
        {"full": pools[e05.MAX_SEQ_LEN], "short": pools[best["short_len"]]},
        {"full": 1.0 - best["w_short"], "short": best["w_short"]},
        WORK_DIR / "e15_last1_best_blend_top200.parquet",
    )
    state["last1"] = {
        "train_events": int(len(train)), "gt": "последнее событие истории (last_n_split n=1)",
        "solo": solo, "fixed_config": best, "fixed_panel": fixed, "base_solo_len150": float(base),
        "ceiling": {**ceiling, "panel": ceiling_panel}, "grid_ceiling": grid,
        "overlap": {str(n): overlap_diagnostics(pools[e05.MAX_SEQ_LEN], p, gt, n) for n, p in shorts.items()},
        "artifact_top200": str(artifact),
        "kaggle_multiplier_note": "last1 -> Kaggle ~ x0.93-1.0 (калибровка popularity/E09)",
    }
    save_state(state)


def second_valid_event(valid: pd.DataFrame) -> pd.DataFrame:
    """gt = второе по timestamp событие valid-части каждого пользователя."""
    ordered = valid.sort_values(["user_id", "timestamp"], kind="stable")
    return ordered[ordered.groupby("user_id").cumcount() == 1][["user_id", "item_id"]]


def _panel(top: pd.DataFrame, gt: pd.DataFrame, train: pd.DataFrame, stage: str) -> dict:
    panel = e10.evaluate(top[top["rank"] <= TOP_K], gt, train_events=train)
    e10.check_invariants(panel, stage, n_users=int(gt["user_id"].nunique()))
    return e10.panel_to_dict(panel)


def stage_lookahead(state: dict) -> None:
    """H15b: второй инференс-проход с top-1 первого прохода в контексте.

    Контекст = последние 149 реальных событий + top-1 unseen предсказание
    первого прохода в последней позиции (ридер берёт хвост окна 150). Оценки:
    next1@9010 (gt = первое событие valid) и next2@9010 (gt = второе событие)
    — прямая проверка «покрытия gt-позиций 2-3». Панели: lookahead-соло и
    rank-average 0.5/0.5 полного и lookahead-проходов против соло прохода 1.
    """
    if "outer" not in state:
        raise RuntimeError("сначала --stage outer")
    events = e05.load_events(DATA_DIR)
    catalog_size = e05.item_catalog(events)
    train, gt1, ckpt, full_predict = level_setup("outer", events)
    _, valid_9010 = chronological_split(events, train_fraction=0.9)
    gt2 = second_valid_event(valid_9010)
    log(f"[lookahead] gt1: {len(gt1)} пользователей, gt2: {len(gt2)}")

    full = pd.read_parquet(WORK_DIR / "pool_outer_len150.parquet")
    top1 = full[full["rank"] == 1].set_index("user_id")["item_id"].to_dict()

    lookahead_parquet = WORK_DIR / "predict_outer_lookahead.parquet"
    if not lookahead_parquet.is_file():
        sequences = e05.build_sequences(train)
        items = [list(map(int, np.asarray(row, dtype=int))) for row in sequences["item_id"]]
        stamps = [list(map(int, np.asarray(row, dtype=int))) for row in sequences["timestamp"]]
        for user, item_list, stamp_list in zip(sequences["user_id"].to_numpy(), items, stamps):
            item_list.append(int(top1[int(user)]))
            stamp_list.append(stamp_list[-1] + 1)
        pd.DataFrame({"user_id": sequences["user_id"], "item_id": items, "timestamp": stamps}).to_parquet(
            lookahead_parquet, index=False
        )
        log(f"lookahead-parquet (полное окно + top-1): {lookahead_parquet}")

    pool_path = WORK_DIR / "pool_outer_lookahead.parquet"
    if pool_path.is_file():
        lookahead = pd.read_parquet(pool_path)
    else:
        log(f"[lookahead] инференс второго прохода: {ckpt}...")
        scores = e10.sasrec_scores_from_checkpoint(ckpt, lookahead_parquet, catalog_size)
        lookahead = e05.filter_seen_topk(scores, train, k=POOL)
        lookahead.to_parquet(pool_path, index=False)

    blended = e10.blend_top_k(e10.blend_frame({"full": full, "lookahead": lookahead}), {"full": 0.5, "lookahead": 0.5})
    state["lookahead"] = {
        "mechanic": "контекст окна 150 = последние 149 событий + top-1 unseen прохода 1; тот же outer-чекпойнт",
        "next1": {
            "pass1_solo": state["outer"]["solo"]["len150"]["filtered_panel"],
            "lookahead_solo": _panel(lookahead, gt1, train, "lookahead-next1"),
            "blend_05": _panel(blended, gt1, train, "lookahead-next1-blend"),
        },
        "next2": {
            "pass1_solo": _panel(full, gt2, train, "lookahead-next2"),
            "lookahead_solo": _panel(lookahead, gt2, train, "lookahead-next2"),
            "blend_05": _panel(blended, gt2, train, "lookahead-next2-blend"),
        },
    }
    for target in ("next1", "next2"):
        part = state["lookahead"][target]
        log(f"[lookahead] {target}: pass1 {part['pass1_solo']['recall@10_macro_relevant']:.4f} | "
            f"lookahead {part['lookahead_solo']['recall@10_macro_relevant']:.4f} | "
            f"blend0.5 {part['blend_05']['recall@10_macro_relevant']:.4f}")
    save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=(
        "check", "inner", "outer", "last1", "lookahead"))
    args = parser.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    stages = {
        "check": stage_check,
        "inner": lambda: stage_inner(state),
        "outer": lambda: stage_outer(state),
        "last1": lambda: stage_last1(state),
        "lookahead": lambda: stage_lookahead(state),
    }
    stages[args.stage]()


if __name__ == "__main__":
    main()
