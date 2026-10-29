"""Прогон вариантов гипотез на фиксированном наборе сплитов + сохранение результатов.

Все эксперименты E01–E05 используют один реестр сплитов и одну панель метрик
(``metrics.evaluate``), чтобы результаты были сравнимы между собой (правило
честности из notes/hypotheses.md). Результат пишется в ``results/<name>.json``:

    {"splits": {"9010": {"train": 802016, "valid": 92133, ...}},
     "results": {"9010": {"<variant>": {"recall@10_macro_relevant": ...}}}}

Сплиты:

- ``9010`` — chronological 90/10, основной temporal-протокол;
- ``last10`` — последние 10 событий, robustness-проверка;
- ``masked10`` — случайная маска 10% историй, competition-mimic диагностика
  (гипотеза о замаскированных взаимодействиях, notes/data.md, раздел 7).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd

from recsys.data import load_events
from recsys.metrics import evaluate
from recsys.split import chronological_split, last_n_split, masked_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# (train, valid) по events; все три детерминированы
SPLITTERS: dict[str, Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]] = {
    "9010": lambda events: chronological_split(events, train_fraction=0.9),
    "last10": lambda events: last_n_split(events, n=10),
    "masked10": lambda events: masked_split(events, valid_fraction=0.1, seed=20261029),
}


def eval_variants(
    variants: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
    splits: tuple[str, ...] = ("9010", "last10", "masked10"),
    events: pd.DataFrame | None = None,
    k: int = 10,
) -> dict:
    """Прогнать каждый вариант (train -> recs) на каждом сплите.

    Возвращает структуру для ``save_results``: размеры сплитов и полная панель
    метрик каждого варианта на каждом сплите.
    """
    if events is None:
        events = load_events(DATA_DIR)
    payload: dict = {"splits": {}, "results": {}}
    for split_name in splits:
        train, valid = SPLITTERS[split_name](events)
        payload["splits"][split_name] = {
            "train": len(train),
            "valid": len(valid),
            "valid_users": int(valid["user_id"].nunique()),
        }
        split_results: dict = {}
        for variant_name, make_recs in variants.items():
            recs = make_recs(train)
            panel = evaluate(recs, valid, k=k, train_events=train)
            split_results[variant_name] = {str(key): float(value) for key, value in panel.items()}
            print(
                f"[{split_name}] {variant_name}: "
                f"recall@10_macro_relevant={panel['recall@10_macro_relevant']:.4f} "
                f"micro={panel['recall@10_micro']:.4f}",
                flush=True,
            )
        payload["results"][split_name] = split_results
    return payload


def best_variant(results: dict, split: str = "9010", metric: str = "recall@10_macro_relevant") -> str:
    """Имя варианта с лучшей метрикой на заданном сплите."""
    split_results = results["results"][split]
    return max(split_results, key=lambda name: split_results[name][metric])


def save_results(results: dict, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return path


def load_results(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())
