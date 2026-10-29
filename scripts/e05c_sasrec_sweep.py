"""E05c: свип ёмкости SASRec на CPU (без GPU-ядра).

Мотивация: E05 (e64/len100/ep15) на 9010 недообучен — лучшей была последняя
эпоха (14/15), monitor recall@10 продолжал расти. Проверяем на основном
сплите 9010 (решение по калибровке 2026-10-29: выбор моделей — по
recall@10_macro_relevant на 9010):

- ep40: тот же конфиг, 40 эпох — есть ли плато после 15-й;
- e128_len150: embedding 128, окно 150, 25 эпох;
- e128_len200: embedding 128, окно 200, 25 эпох (если время позволяет).

Лучший конфиг обучается на полных events (без валидации, те же эпохи) и
собирается в сабмит. Скрипт e05_sasrec.py импортируется как модуль;
гиперпараметры переопределяются через его глобалы (функция читает их
в момент вызова).

Запуск:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python scripts/e05c_sasrec_sweep.py
"""

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

spec = importlib.util.spec_from_file_location("e05_sasrec", PROJECT_ROOT / "scripts" / "e05_sasrec.py")
e05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e05)

from recsys.data import load_events
from recsys.experiments import RESULTS_DIR, SPLITTERS
from recsys.metrics import evaluate, save_submission

SWEEP = [
    # история: первая версия свипа из-за default-аргумента epochs обучила все
    # конфиги по 15 эпох независимо от имени — записи переименованы в *_ep15
    {"name": "e64_len100_ep15", "EMBEDDING_DIM": 64, "MAX_SEQ_LEN": 100, "NUM_HEADS": 2, "EPOCHS": 15, "BATCH_SIZE": 64},
    {"name": "e128_len150_ep15", "EMBEDDING_DIM": 128, "MAX_SEQ_LEN": 150, "NUM_HEADS": 4, "EPOCHS": 15, "BATCH_SIZE": 64},
    {"name": "e128_len200_ep15", "EMBEDDING_DIM": 128, "MAX_SEQ_LEN": 200, "NUM_HEADS": 4, "EPOCHS": 15, "BATCH_SIZE": 64},
    # настоящий 40-эпоховый прогон лучшего конфига (эпохи дают +8.5% на e64)
    {"name": "e128_len150_ep40", "EMBEDDING_DIM": 128, "MAX_SEQ_LEN": 150, "NUM_HEADS": 4, "EPOCHS": 40, "BATCH_SIZE": 64},
]


def apply_config(config: dict) -> None:
    for attr, value in config.items():
        if attr == "name":
            continue
        setattr(e05, attr, value)


def main() -> None:
    events = load_events()
    catalog_size = e05.item_catalog(events)
    train, valid = SPLITTERS["9010"](events)

    results_path = RESULTS_DIR / "e05c_sasrec_sweep.json"
    payload: dict = {"results": {}}
    if results_path.is_file():
        payload = json.loads(results_path.read_text())

    best_name, best_recall = None, -1.0
    for config in SWEEP:
        name = config["name"]
        if name in payload["results"]:
            print(f"[skip] {name}: уже посчитан", flush=True)
        else:
            apply_config(config)
            print(f"\n=== {name}: {config} ===", flush=True)
            # epochs передаётся явно: default-аргумент fit_predict_sasrec
            # заморожен на моменте импорта и не видит переопределения EPOCHS
            scores = e05.fit_predict_sasrec(
                train, valid, catalog_size, e05.WORK_DIR / "e05c" / name, epochs=config["EPOCHS"]
            )
            recs = e05.filter_seen_topk(scores, train)
            panel = evaluate(recs, valid, k=10, train_events=train)
            payload["results"][name] = {str(k): float(v) for k, v in panel.items()}
            payload["results"][name]["config"] = {k: v for k, v in config.items() if k != "name"}
            results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            print(f"[9010] {name}: recall@10_macro_relevant={panel['recall@10_macro_relevant']:.4f}", flush=True)
        recall = payload["results"][name]["recall@10_macro_relevant"]
        if recall > best_recall:
            best_name, best_recall = name, recall

    print(f"\nлучший конфиг: {best_name} ({best_recall:.4f}) — обучение на полных events", flush=True)
    best_config = next(c for c in SWEEP if c["name"] == best_name)
    apply_config(best_config)
    scores = e05.fit_predict_sasrec(events, None, catalog_size, e05.WORK_DIR / "e05c" / "full", epochs=best_config["EPOCHS"])
    recs = e05.filter_seen_topk(scores, events)
    out = PROJECT_ROOT / "submissions" / f"E05c_sasrec_{best_name}.csv"
    save_submission(recs, out)
    print(f"сабмит E05c: {out}")


if __name__ == "__main__":
    main()
