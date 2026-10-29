# План UV RecSys

## 0. Цель

Построить воспроизводимое решение HSE RecSys Challenge 2026 с качеством выше требуемого baseline по **Recall@10**, при этом сохранить понятный путь экспериментов и уметь объяснить каждое существенное решение.

Мы заранее ведём две ветки:

1. **быстрый сильный baseline** — в том числе SASRec / BERT4Rec;
2. **non-transformer / hybrid pipeline** — collaborative filtering, candidate generation, features, boosting, ensembles.

Так мы не зависим от возможного дополнительного требования преподавателя отказаться от трансформера.

---

## 1. Данные и EDA

Файлы:
- `events.csv`;
- `user_features.csv`;
- `item_features.csv`;
- `submission_sample.csv`.

Проверить:

### Events
- число interactions;
- число пользователей;
- число фильмов;
- колонки и типы;
- временной диапазон;
- распределение числа interactions на пользователя;
- распределение числа interactions на item;
- sparsity user-item матрицы;
- распределение ratings;
- дубликаты user-item;
- повторные оценки;
- долю пользователей с короткой историей;
- cold / warm users и items.

### User features
- набор признаков;
- пропуски;
- cardinality категориальных признаков;
- покрытие пользователей относительно events.

### Item features
- genres / categories / textual или другие признаки;
- пропуски;
- покрытие items относительно events;
- возможность построить content similarity и user preference features.

---

## 2. Валидация

### Основной принцип

Выбираем **один глобальный temporal cutoff** `T`.

```text
event.timestamp < T   -> train
event.timestamp >= T  -> validation
```

Cutoff выбираем после анализа плотности событий так, чтобы:
- validation была достаточно большой;
- у разумного числа пользователей был ground truth;
- training history оставалась достаточной;
- split максимально напоминал скрытый Kaggle test.

### Защита от leakage

На train side рассчитываются:
- user statistics;
- item statistics;
- popularity;
- similarity matrices;
- ALS factors;
- sequential histories;
- features;
- candidate generators.

Validation events используются только как ground truth.

Никакая статистика из validation не должна попадать в train pipeline.

### Локальная метрика

Реализуем `Recall@10` максимально близко к Kaggle.

Отдельно фиксируем:
- micro/macro трактовка, если она неочевидна;
- как обрабатываются пользователи без ground truth;
- исключаются ли уже просмотренные / оценённые items;
- допускаются ли повторные рекомендации.

**Зафиксировано (2026-10-29, реализовано в `src/recsys/metrics.py`):**

- точная формула Kaggle не опубликована: на страницах соревнования только
  «Метрика: recall@10». Поэтому `evaluate()` считает панель всех стандартных
  вариантов рядом; формула Kaggle идентифицируем совпадением варианта со скором
  первых сабмитов (совпадение до 4 знаков на двух разных сабмитах);
- основной вариант до сверки — macro `hits / |ground truth|` (`recall@10_macro_relevant`);
- ground truth — все взаимодействия holdout независимо от rating
  (гипотеза о замаскированных взаимодействиях, раздел 7 в data.md);
  переключение — параметр `min_rating` в `evaluate`;
- пользователи без ground truth исключаются из метрики;
  с ground truth, но без рекомендаций — recall 0;
- повторные рекомендации: дубликат внутри топ-k учитывается один раз и
  «сжигает» слот (диагностика `dup_in_top_share`);
- уже оценённые items метрика не отбрасывает: они не могут быть хитом
  (в events.csv нет повторных пар user-item) и штрафуются автоматически;
  фильтрация просмотренного — зона ответственности модели
  (диагностика `seen_in_top_share`).

---

## 3. Baseline ladder

Цель этапа — не максимизировать качество, а построить проверяемую лестницу моделей.

### E00 — Submission sanity check
- валидный формат сабмита;
- 10 items на пользователя;
- отсутствие дубликатов;
- coverage test users.

### E01 — Global popularity
Топ-10 наиболее популярных фильмов по train.

### E02 — Rating-aware popularity
Проверить несколько простых вариантов:
- interaction count;
- средняя оценка;
- weighted popularity;
- positive-only popularity.

### E03 — Item similarity / ItemKNN
Проверить:
- cosine;
- co-occurrence;
- rating-aware similarity.

### E04 — ALS
Несколько разумных вариантов:
- implicit interactions;
- rating-weighted interactions;
- фильтрация / weighting low ratings.

### E05 — SASRec
RePlay baseline.

Цель: проверить утверждение преподавателя о сильном transformer fit-predict baseline.

### E06 — BERT4Rec
Только если SASRec показывает, что sequential branch действительно перспективна.

GPT4Rec пока не является обязательным этапом: сначала проверяем более дешёвые sequential models.

---

## 4. Candidate generation

После отдельных baseline переходим от соревнования моделей к их сочетанию.

Candidate sources:
- Popularity;
- ItemKNN;
- ALS;
- SASRec;
- BERT4Rec;
- content-based candidates при наличии полезных item features.

Для каждого генератора сохраняем:
- `score`;
- `rank`;
- `source`;
- top-N candidates.

### Главная диагностическая метрика

До reranking считаем **Candidate Recall@N**.

Пример:

```text
ALS                 Recall@200 = ...
KNN                 Recall@200 = ...
SASRec              Recall@200 = ...
ALS ∪ KNN           Recall@200 = ...
ALS ∪ KNN ∪ SASRec  Recall@200 = ...
```

Это позволит понять, какие слабые модели всё равно дают complementary candidates.

---

## 5. Feature engineering

После формирования candidate set строим признаки только на train-side относительно cutoff.

### User features
- число interactions;
- число positive interactions;
- средний rating;
- variance rating;
- activity / recency;
- любимые genres/categories;
- доля взаимодействий по жанрам;
- признаки из `user_features.csv`.

### Item features
- popularity;
- average rating;
- interaction count;
- recent popularity;
- genres/categories;
- признаки из `item_features.csv`.

### User-item features
- был ли item ранее у пользователя;
- число взаимодействий;
- last interaction time;
- user-category affinity;
- user-item / user-genre similarity;
- embedding similarity.

### Candidate-source features
- ALS score / rank;
- KNN score / rank;
- SASRec score / rank;
- BERT4Rec score / rank;
- был ли item найден каждым генератором;
- число генераторов, нашедших candidate.

---

## 6. Ranking

Проверить по возрастанию сложности:

1. один лучший candidate score;
2. hand-crafted weighted score;
3. score blending;
4. rank blending;
5. CatBoost / LightGBM ranker.

Не переходить к 200–300 признакам автоматически. Сначала получить небольшой осмысленный feature set и проверить, что reranking вообще даёт прирост.

---

## 7. Ensembles

Основные гипотезы:
- модели находят разные релевантные items;
- union повышает candidate recall;
- score/rank blending повышает Recall@10;
- CatBoost учится выбирать между источниками лучше фиксированных весов;
- transformer + classical CF дополняют друг друга.

Финальный ensemble строить только после стабильной локальной валидации.

---

## 8. Transformer vs non-transformer

Поддерживаем две независимые линии:

### Transformer
- SASRec;
- BERT4Rec;
- при необходимости GPT4Rec.

### Non-transformer
- popularity;
- similarity / KNN;
- ALS / MF;
- feature engineering;
- boosting;
- ensembles.

Это позволит удовлетворить возможному дополнительному условию преподавателя.

---

## 9. Работа с Kaggle

Не использовать public leaderboard как единственную валидацию.

Каждый существенный сабмит должен иметь:
- experiment id;
- git commit;
- локальную метрику;
- краткое описание изменения;
- Kaggle metric;
- вывод.

При лимите 10 submissions/day сабмитим только содержательные варианты.

---

## 10. Воспроизводимость

К финалу:

- фиксируем random seeds;
- фиксируем зависимости;
- исключаем ручные промежуточные шаги;
- итоговый pipeline запускается одной командой или одним notebook run;
- лучший сабмит можно получить заново из чистого состояния.

Переиспользуемый код помещаем в `src/uv_recsys/`.

Экспериментальные ноутбуки — в `notebooks/`.

---

## 11. Ближайшие действия

1. Разобрать schema всех CSV.
2. Провести EDA.
3. Найти корректный global temporal cutoff.
4. Реализовать leakage-safe split.
5. Реализовать Recall@10.
6. Получить Popularity baseline.
7. Получить ALS.
8. Получить ItemKNN.
9. Получить SASRec через RePlay.
10. Сравнить local и Kaggle.
11. После первых результатов сформировать следующую волну гипотез.
