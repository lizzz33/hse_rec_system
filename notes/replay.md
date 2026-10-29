# Особенности работы с RePlay

Заметки по библиотеке RePlay (sb-ai-lab) — установке в этот проект и подводным камням API.
Всё проверено на живом smoke-тесте (PopRec + ItemKNN) 2026-10-29.

---

## 1. Установка

### Пакет называется `replay-rec`, а не `replay`

В PyPI пакет `replay` — **чужая библиотека-тёзка** (Maik Messerschmidt, github.com/messersm/replay, версии 0.1.x). Настоящая RePlay от sb-ai-lab публикуется как **`replay-rec`** (импорт — `replay`).

```
# неправильно: pip install replay   <- чужой пакет
# правильно:
uv add "replay-rec[spark,torch]"
```

### Extra `[all]` не существует

У `replay-rec==0.22.0` два extra: `spark` и `torch`. Комбинация `spark,torch` эквивалентна упомянутому в старой документации `[all]`. Без extra ставится только ядро (polars/sklearn), но модели так не работают — см. п. 3.

### Итоговые ограничения версий (зафиксированы в pyproject.toml)

| Зависимость | Пин | Причина |
|---|---|---|
| `replay-rec[spark,torch]` | `>=0.22.0` | — |
| `pandas` | `>=2.3,<3` | replay-rec 0.22.0 требует `pandas<3` (была 3.1 — понижена до 2.3.3) |
| `numpy` | `>=2.4,<2.5` | replay-rec 0.22.0 требует `numpy<2.5` (была 2.5.4 — понижена до 2.4.6) |
| `pyspark` | `>=3.5,<4` | pyspark 4.x требует **Java 17**, на машине только **Java 11** (sudo с паролем, обновить нельзя); стоит 3.5.9 |

Прочее из установки:

- `torch 2.15.0` — CUDA-сборка, **GPU на машине нет**, обучение идёт на CPU;
- `optuna` НЕ ставится → метод `model.optimize()` недоступен (warning `FeatureUnavailableWarning` при импорте; при необходимости — добавить optuna отдельно);
- код нашего проекта (`src/recsys/`) от понижения pandas/numpy не ломается — обычный pandas API.

---

## 2. Spark-сессия

- Сессию создавать через `replay.utils.session_handler.get_spark_session()` — она сама настроит память, `PYSPARK_PYTHON`, локальный master и т.д.
- По умолчанию RePlay под Spark 3.x подхватывает свой Scala-jar с Maven (`replay_2.12-3.1.3.jar`) — это работает. Отключается пустой переменной окружения `REPLAY_JAR_PATH=`.
- Полезные env: `REPLAY_SPARK_CORE_COUNT` (число ядер), `REPLAY_SPARK_MEMORY` (GB памяти драйвера; по умолчанию 70% RAM).

---

## 3. API: Dataset и FeatureSchema

### На вход моделям нужен именно Spark DataFrame

`Dataset` формально принимает pandas / polars / Spark DataFrame, но **модели внутри используют Spark API** (`.select()`, `.distinct()`):
- pandas-датафрейм падает с `AttributeError: 'DataFrame' object has no attribute 'select'`;
- polars-датафрейм падает с `... no attribute 'distinct'`.

Рабочий паттерн:

```python
from replay.utils.session_handler import get_spark_session

spark = get_spark_session()
interactions = spark.createDataFrame(pandas_df)
dataset = Dataset(feature_schema=schema, interactions=interactions)
```

### Схема обязательных колонок

`FeatureSchema` собирается из `FeatureInfo(column, feature_type, feature_hint)`. Хинты: `QUERY_ID`, `ITEM_ID`, `RATING`, `TIMESTAMP`; типы: `CATEGORICAL`, `CATEGORICAL_LIST`, `NUMERICAL`, `NUMERICAL_LIST`.

**Ловушка: без RATING-колонки модели падают с невнятным NPE.** `PopRec` делает `agg(...).alias(self.rating_column)` — при отсутствии RATING-хинта `rating_column = None`, и Spark падает `INTERNAL_ERROR ... NullPointerException` в глубине анализа запроса, без упоминания настоящей причины. Лечится добавлением в данные колонки `rating` (можно константной) и `FeatureInfo("rating", NUMERICAL, FeatureHint.RATING)`.

У нас в `events.csv` rating есть — но помнить про это важно при генерации синтетических/тестовых данных.

### Минимальный рабочий пример (проверен)

```python
import pandas as pd
from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.models import ItemKNN, PopRec
from replay.utils.session_handler import get_spark_session

pdf = pd.DataFrame({
    "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    "item_id": [10, 20, 30, 10, 30, 20, 30, 40, 10, 20],
    "rating": [1.0] * 10,
})
schema = FeatureSchema([
    FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
    FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
    FeatureInfo("rating", FeatureType.NUMERICAL, FeatureHint.RATING),
])
spark = get_spark_session()
dataset = Dataset(feature_schema=schema, interactions=spark.createDataFrame(pdf))

recs = PopRec().fit_predict(dataset, k=3)     # -> Spark DataFrame
recs_pd = recs.toPandas()
```

---

## 4. Имена классов и модулей

- Классические модели: `from replay.models import PopRec, ItemKNN, ALSWrap, Wilson, SLIM, RandomRec, AssociationRulesItemRec, Word2VecRec, ...`
- Нейросетевые: `from replay.models.nn.sequential.sasrec import SasRec`, `from replay.models.nn.sequential.bert4rec import Bert4Rec` — **нижний регистр `Sas`/`Bert`**, не `SASRec`/`BERT4Rec` (на верхнерегистровые имена ImportError с подсказкой).
- Метрики: `replay.metrics`; сплиттеры: `replay.splitters`.

---

## 5. Ссылки

- Репозиторий: https://github.com/sb-ai-lab/RePlay
- PyPI: https://pypi.org/project/replay-rec/
