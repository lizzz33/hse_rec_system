# Данные HSE RecSys Challenge 2026

## 1. Состав датасета

В соревновании четыре файла:

- `events.csv` — история взаимодействий;
- `user_features.csv` — признаки пользователей;
- `item_features.csv` — признаки фильмов;
- `submission_sample.csv` — формат сабмита.

### events.csv

Размер: **894 149 строк × 4 колонки**.

Колонки:

- `user_id` — идентификатор пользователя;
- `item_id` — идентификатор фильма;
- `rating` — оценка от 1 до 5;
- `timestamp` — порядковая временная координата взаимодействия внутри истории пользователя.

Основные размеры:

- пользователей: **6 040**;
- фильмов с хотя бы одним interaction: **3 690**;
- всего фильмов в item features: **3 706**;
- interactions: **894 149**;
- density user-item матрицы: **≈ 4.01%**;
- дубликатов `(user_id, item_id)`: **0**;
- пропусков: **0**.

Распределение ratings:

| Rating | Count | Доля |
|---:|---:|---:|
| 1 | 50 256 | 5.6% |
| 2 | 96 001 | 10.7% |
| 3 | 233 824 | 26.2% |
| 4 | 312 170 | 34.9% |
| 5 | 201 898 | 22.6% |

Полезные пороги:

- `rating >= 3`: ~83.6% interactions;
- `rating >= 4`: ~57.5%;
- `rating == 5`: ~22.6%.

Это делает бинаризацию feedback отдельной содержательной гипотезой:
- все оценки как interaction;
- только `rating >= 3`;
- только `rating >= 4`;
- rating-weighted implicit feedback.

---

## 2. User features

`user_features.csv`: **6 040 пользователей × 3 колонки**.

Признаки:

- `user_id`;
- `gender`: M / F;
- `age`: дискретная возрастная категория.

Gender:

- M: 4 331;
- F: 1 709.

Age принимает только значения:

`1, 18, 25, 35, 45, 50, 56`.

Это не непрерывный точный возраст, а категориальный/групповой признак.

Все пользователи из events покрыты user features. Пропусков нет.

---

## 3. Item features

`item_features.csv`: **3 706 фильмов × 19 колонок**.

Признаки:

- `item_id`;
- `genre_0 ... genre_17` — 18 бинарных жанровых признаков.

На один фильм приходится:

- минимум 1 жанр;
- медиана 1 жанр;
- 75-й перцентиль — 2 жанра;
- максимум — 6 жанров.

Пропусков нет.

16 фильмов присутствуют в item features, но не встречаются в `events.csv`.

### Вывод

Raw-фичей действительно немного, но это **не означает, что у нас мало доступных признаков**.

Большинство сильных признаков можно получить из interaction history:

- user activity;
- item popularity;
- average / std rating пользователя;
- average / std rating фильма;
- recency;
- genre affinity пользователя;
- user × genre statistics;
- demographic × genre statistics;
- item-item co-occurrence;
- ALS / KNN / transformer score;
- rank внутри каждого candidate generator;
- число моделей, которые нашли candidate;
- latent / embedding similarities.

Поэтому 20–50 осмысленных признаков собрать реалистично даже из этого компактного schema.

---

## 4. Sparsity и popularity

### User activity

Число interactions на пользователя:

- min: 14;
- 25%: 38;
- median: 84.5;
- 75%: 186;
- 90%: 357;
- max: 2 056.

То есть cold users в абсолютном смысле почти отсутствуют: даже самый короткий профиль содержит 14 известных взаимодействий.

### Item popularity

Число interactions на item:

- min: 1;
- median: 111;
- 75%: ~316;
- 90%: ~656;
- max: 3 067.

Popularity сильно неоднородна:

- top 1% items дают ~8.6% interactions;
- top 5% — ~28.2%;
- top 10% — ~44.3%;
- top 20% — ~65.1%.

Следствие: popularity baseline должен быть заметным, но возникает сильное пространство для персонализации и борьбы с popularity bias.

---

## 5. Самый важный инсайт: что означает timestamp

Изначальная идея была выбрать **одну глобальную временную точку** и разделить весь датасет:

`past -> cutoff -> future`.

После анализа данных выяснилось, что текущий `timestamp` **не является общей календарной временной шкалой для всех пользователей**.

Наблюдения:

- timestamp лежит в диапазоне `0 ... 2312`;
- почти каждый пользователь начинает историю с `timestamp = 0`;
- 5 471 из 6 040 пользователей имеют первый наблюдаемый event ровно в 0;
- остальные почти всегда начинают с 1–4;
- внутри одного пользователя timestamp уникален;
- `corr(number_of_interactions, max_timestamp) ≈ 0.99975`;
- история пользователя выглядит как `0, 1, 2, 3, 4, 5, 7, 8, ...`.

Следовательно, timestamp практически наверняка является **позицией / относительным временем внутри индивидуальной истории пользователя**, а не общей датой.

### Почему нельзя взять глобальный cutoff по числу timestamp

Например, при `timestamp = 119`:

- train ≈ 445 586 interactions;
- validation ≈ 448 563;
- но в validation остаются только **2 521 пользователя из 6 040**.

То есть такой split просто отберёт пользователей с длинными историями. Это будет сильный selection bias, а не честное деление общего времени.

**Поэтому глобальный числовой cutoff по timestamp использовать не будем.**

---

## 6. Предлагаемый temporal validation split

Так как общей календарной оси в данных нет, сохраняем принцип преподавателя «не смотреть в будущее», но применяем его внутри каждой пользовательской последовательности.

### Основной вариант: chronological 90/10

Для каждого пользователя:

1. сортируем interactions по timestamp;
2. первые 90% истории — train;
3. последние 10% — validation.

Это один общий **relative cutoff = 90% истории** для всех пользователей.

Получается:

- train: **802 016 interactions**;
- validation: **92 133 interactions**;
- validation содержит **всех 6 040 пользователей**;
- median validation interactions/user: **9**;
- cold users в validation: **0**;
- cold validation items относительно train: только **22 item**, всего **28 interactions**.

Это удобно для Recall@10 и почти не создаёт отдельной cold-item задачи.

### Альтернатива: last-10

Можно оставить последние 10 interactions каждого пользователя:

- validation = ровно 60 400 interactions;
- все 6 040 пользователей;
- ровно 10 ground-truth items/user;
- только 6 cold validation items.

Этот split особенно удобен концептуально из-за `Recall@10`, но он сильнее искажает относительную долю holdout для пользователей с очень короткой или очень длинной историей.

### Решение на старте

**Primary local validation: chronological 90/10 per user.**

Last-10 можно использовать как дополнительную robustness-проверку.

Важно: поскольку глобального absolute timestamp нет, невозможно реализовать настоящий единый календарный cutoff. Это ограничение самого предоставленного датасета, а не нашего pipeline.

### Верификация сплита (2026-10-29, `src/recsys/split.py`)

Оба сплита проверены на реальном `events.csv` юнит-тестами (`tests/test_split.py`, все 6 прошли) и инвариантами на полных данных:

- сохранность: `train + valid = 894 149`, объединение пар (user, item) ровно восстанавливает events, дубликатов внутри частей нет;
- leakage: пересечение пар (user, item) train ∩ valid = **0**; у всех **6 040** пользователей `max(train timestamp) < min(valid timestamp)` — train строго раньше valid внутри каждой истории;
- покрытие: все пользователи есть в обоих частях, пустых train-историй нет (минимальная история — 14 событий);
- числа выше подтверждены: 802 016 / 92 133, медиана 9, cold items 22 / 28 interactions (chronological 90/10); 833 749 / 60 400, ровно 10 на пользователя, 6 cold items (last-10).

Нюанс: плотной нумерации `0..n-1` нет — у **5 921 из 6 040** пользователей в диапазоне timestamp есть пропуски (см. раздел 7). Сплит на это не опирается: он ранжирует timestamp внутри пользователя (уникальность внутри user гарантирована в `load_events`), а не предполагает непрерывную нумерацию.

---

## 7. Возможный дополнительный инсайт о скрытом тесте

В `timestamp` внутри пользовательских последовательностей есть пропуски.

Например:

`0, 1, 2, 3, 4, 5, 7, 8, ...`

При этом:

- средняя доля наблюдаемых позиций внутри диапазона пользователя ≈ **90.2%**;
- медианная ≈ **90.24%**.

Это похоже на ситуацию, где часть исходных interactions была скрыта при создании competition test.

Это **гипотеза, а не подтверждённый факт**.

Она важна потому, что Kaggle hidden set потенциально может быть ближе к masked-interaction evaluation, чем к чистому future-only holdout.

Поэтому позже имеет смысл сравнить correlation leaderboard с двумя локальными протоколами:

1. chronological 90/10 — основной leakage-safe protocol;
2. controlled random/masked 10% — только как competition-mimic diagnostic.

Нельзя выбирать protocol только потому, что он лучше коррелирует с public leaderboard после большого количества сабмитов: это приведёт к leaderboard overfitting.

---

## 8. Первые гипотезы, которые следуют непосредственно из данных

### H1. Rating threshold

Какой feedback полезнее для candidate generation:

- все ratings;
- `rating >= 3`;
- `rating >= 4`;
- weighted rating.

Метрика: local Recall@10.

### H2. Popularity

Из-за сильного long tail global popularity должен быть разумным baseline.

Сравнить:
- count popularity;
- average-rating popularity;
- Bayesian / shrinkage rating;
- positive-only popularity.

### H3. Collaborative filtering

Матрица имеет density ≈ 4%, пользователи имеют достаточно длинные истории.

Это хорошие условия для:
- ItemKNN;
- ALS;
- EASE / похожих методов.

### H4. Sequential signal

Timestamp задаёт явный порядок событий.

Проверить SASRec.

Метрика:
- Recall@10;
- дополнительно candidate Recall@N.

### H5. Genre signal

18 жанров позволяют строить:
- user genre profile;
- genre affinity;
- demographic × genre preference;
- content candidates;
- user-item genre similarity.

### H6. Ensemble / two-stage

Разные модели вероятно будут давать complementary candidates.

Проверяем:

`Popularity ∪ KNN ∪ ALS ∪ SASRec -> candidate set -> reranking`.

Сначала измеряем Candidate Recall@N, затем итоговый Recall@10.

---

## 9. Что делать дальше

Следующий минимальный pipeline:

1. зафиксировать код temporal 90/10 split;
2. реализовать Recall@10;
3. сделать E01 Global Popularity;
4. сделать rating-aware popularity;
5. ItemKNN;
6. ALS;
7. SASRec / RePlay;
8. сравнить overlap и Candidate Recall разных моделей;
9. только после этого переходить к feature reranking и ensemble.

Главный принцип: **каждая новая модель должна отвечать на гипотезу и иметь понятное сравнение с предыдущим baseline.**
