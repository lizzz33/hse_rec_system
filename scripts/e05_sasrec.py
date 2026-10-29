"""E05: SASRec через RePlay (transformer baseline, рекомендация преподавателя).

Проверяем утверждение преподавателя, что sequential transformer — сильный
baseline почти в стиле fit -> predict: порядок событий внутри истории
(наш timestamp) — явный сигнал для next-item предсказания.

Протокол (единый с E01–E04): обучение только на train-части сплита,
рекомендации — топ-10 unseen items; валидация метрикой recsys.metrics.

Особенности RePlay 0.22 (notes/replay.md и официальный пример 09):

- sequences строятся ``groupby_sequences`` и пишутся в parquet;
- обучение — lightning ``Trainer`` + ``ParquetModule``;
- monitor на валидации — recall@10 по одному gt-item на пользователя
  (первое событие holdout); итоговая метрика считается нашим ``evaluate``
  на полном holdout после фильтрации уже оценённых items;
- финальные скоры берём по всему каталогу (top_k = item_count), чтобы
  после фильтрации seen у каждого пользователя осталось >= 10 кандидатов.

Запуск:
    uv run python scripts/e05_sasrec.py                     # 9010 + masked10
    uv run python scripts/e05_sasrec.py --splits 9010
    uv run python scripts/e05_sasrec.py --submission        # + обучение на полных events
"""

import argparse
import json
import os
import sys
from pathlib import Path

# до импорта torch/lightning: ограничиваем BLAS-потоки (совет из примера RePlay),
# иначе dataloader и обучающий поток дерутся за ядра
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import lightning as L
import pandas as pd
import torch

torch.set_num_threads(4)

from replay.data.nn import ParquetModule, TensorFeatureInfo, TensorSchema
from replay.data.nn.utils import groupby_sequences
from replay.data.schema import FeatureHint, FeatureType
from replay.nn.lightning import LightningModule
from replay.nn.lightning.callback import ComputeMetricsCallback, PandasTopItemsCallback
from replay.nn.sequential import SasRec
from replay.nn.transform.template import make_default_sasrec_transforms

from recsys.data import load_events
from recsys.experiments import RESULTS_DIR, SPLITTERS
from recsys.metrics import evaluate, save_submission

WORK_DIR = RESULTS_DIR / "e05_sasrec_work"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# гиперпараметры: близко к примеру RePlay 09, уменьшены под CPU и наш размер данных
MAX_SEQ_LEN = 100
EMBEDDING_DIM = 64
NUM_HEADS = 2
NUM_BLOCKS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 15
SEED = 42


def item_catalog(events: pd.DataFrame) -> int:
    """Размер каталога: все item_id из item_features (0..3705) + 1 на padding."""
    from recsys.data import load_item_features

    catalog_size = len(load_item_features(PROJECT_ROOT / "data"))
    assert events["item_id"].max() < catalog_size
    return catalog_size


def build_sequences(train: pd.DataFrame) -> pd.DataFrame:
    """user_id -> упорядоченная по timestamp последовательность item_id."""
    return groupby_sequences(
        train[["user_id", "item_id", "timestamp"]], groupby_col="user_id", sort_col="timestamp"
    )


def fit_predict_sasrec(
    train: pd.DataFrame,
    valid: pd.DataFrame | None,
    catalog_size: int,
    work_dir: Path,
    epochs: int = EPOCHS,
) -> pd.DataFrame:
    """Обучить SASRec на train-последовательностях и скорить все items.

    ``valid`` (train-time monitor) — по одному gt-item на пользователя
    (первое событие holdout); None — обучение без валидации (полные данные).
    Возвращает user_id, item_id, score по всему каталогу, отсортировано
    внутри пользователя по убыванию score.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    padding = catalog_size
    L.seed_everything(SEED)

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

    sequences = build_sequences(train)
    train_path = work_dir / "train.parquet"
    sequences.to_parquet(train_path, index=False)
    predict_path = work_dir / "predict.parquet"
    sequences.to_parquet(predict_path, index=False)

    transforms = make_default_sasrec_transforms(schema)
    fit_metadata = {"train": {"item_id": {"shape": MAX_SEQ_LEN + 1, "padding": padding}}}
    callbacks = []

    validate_path = None
    if valid is not None:
        # monitor: первое по времени событие holdout каждого пользователя;
        # оно не входило в train, поэтому метрика честная.
        # ground_truth обязан быть list-колонкой длины 1 (так устроен
        # parquet-reader: колонки с "shape" читаются как массивы)
        first_holdout = (
            valid.sort_values(["user_id", "timestamp"]).groupby("user_id")["item_id"].first()
        )
        val_df = sequences[["user_id", "item_id"]].copy()
        val_df["ground_truth"] = val_df["user_id"].map(first_holdout).map(lambda item: [int(item)])
        validate_path = work_dir / "validate.parquet"
        val_df.to_parquet(validate_path, index=False)
        fit_metadata["validate"] = {
            "item_id": {"shape": MAX_SEQ_LEN, "padding": padding},
            "ground_truth": {"shape": 1, "padding": -1},
        }
        callbacks.extend([
            ComputeMetricsCallback(metrics=["recall"], ks=[10], item_count=catalog_size),
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="recall@10", mode="max", dirpath=work_dir / "checkpoints", filename="best"
            ),
        ])

    datamodule = ParquetModule(
        train_path=str(train_path),
        validate_path=str(validate_path) if validate_path else None,
        batch_size=BATCH_SIZE,
        metadata=fit_metadata,
        transforms=transforms,
    )
    model = SasRec.from_params(
        schema,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_sequence_length=MAX_SEQ_LEN,
        dropout=DROPOUT,
    )
    module = LightningModule(model)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        devices=1,
        callbacks=callbacks,
        logger=L.pytorch.loggers.CSVLogger(save_dir=str(work_dir), name="lightning_logs"),
        log_every_n_steps=100,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=datamodule)

    # предсказываем весами лучшей эпохи по monitor-метрике, а не последней
    checkpoint = work_dir / "checkpoints" / "best.ckpt"
    module_to_predict = module
    if valid is not None and checkpoint.is_file():
        module_to_predict = LightningModule.load_from_checkpoint(str(checkpoint), model=model)

    # inference-модуль читает user_id для маппинга предсказаний (пример 09);
    # результат забираем из callback (return_predictions=False)
    predict_callback = PandasTopItemsCallback(
        top_k=catalog_size, query_column="user_id", item_column="item_id", rating_column="score"
    )
    predict_datamodule = ParquetModule(
        predict_path=str(predict_path),
        batch_size=BATCH_SIZE,
        metadata={"predict": {"user_id": {}, "item_id": {"shape": MAX_SEQ_LEN, "padding": padding}}},
        transforms=transforms,
    )
    trainer_predict = L.Trainer(
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
        inference_mode=True,
        callbacks=[predict_callback],
    )
    trainer_predict.predict(module_to_predict, datamodule=predict_datamodule, return_predictions=False)
    scores = predict_callback.get_result()
    return scores.sort_values(["user_id", "score"], ascending=[True, False], kind="stable")


def filter_seen_topk(scores: pd.DataFrame, train: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Убрать уже оценённые items и оставить первые k позиций на пользователя.

    ``scores`` уже отсортированы по убыванию score внутри пользователя,
    поэтому groupby.head сохраняет порядок ранжирования.
    """
    seen = train[["user_id", "item_id"]].drop_duplicates()
    unseen = scores.merge(seen, on=["user_id", "item_id"], how="left", indicator=True)
    unseen = unseen[unseen["_merge"] == "left_only"].drop(columns="_merge")
    top = unseen.groupby("user_id", sort=False).head(k).reset_index(drop=True)
    top["rank"] = top.groupby("user_id", sort=False).cumcount() + 1
    return top


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["9010", "masked10"],
                        choices=sorted(SPLITTERS))
    parser.add_argument("--submission", action="store_true",
                        help="дополнительно обучить на полных events и собрать сабмит")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    events = load_events()
    catalog_size = item_catalog(events)
    print(f"каталог: {catalog_size} items (последний id {catalog_size - 1}, padding {catalog_size})")

    payload: dict = {"splits": {}, "results": {}}
    for split_name in args.splits:
        train, valid = SPLITTERS[split_name](events)
        payload["splits"][split_name] = {
            "train": len(train), "valid": len(valid), "valid_users": int(valid["user_id"].nunique()),
        }
        print(f"\n=== {split_name}: train {len(train):,} / valid {len(valid):,} ===", flush=True)
        scores = fit_predict_sasrec(train, valid, catalog_size, WORK_DIR / split_name, epochs=args.epochs)
        recs = filter_seen_topk(scores, train)
        panel = evaluate(recs, valid, k=10, train_events=train)
        payload["results"][split_name] = {"sasrec_default": {str(k_): float(v) for k_, v in panel.items()}}
        print(f"[{split_name}] sasrec_default: recall@10_macro_relevant="
              f"{panel['recall@10_macro_relevant']:.4f} micro={panel['recall@10_micro']:.4f}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / "e05_sasrec.json"
    if results_path.is_file():  # дозаписываем сплиты, если гоняли по частям
        existing = json.loads(results_path.read_text())
        existing["splits"].update(payload["splits"])
        existing["results"].update(payload["results"])
        payload = existing
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nрезультаты: {results_path}")

    if args.submission:
        print("\n=== обучение на полных events для сабмита ===", flush=True)
        scores = fit_predict_sasrec(events, None, catalog_size, WORK_DIR / "full", epochs=args.epochs)
        recs = filter_seen_topk(scores, events)
        SUBMISSIONS_DIR.mkdir(exist_ok=True)
        out = SUBMISSIONS_DIR / "E05_sasrec.csv"
        save_submission(recs, out)
        print(f"сабмит E05: {out}")


if __name__ == "__main__":
    main()
