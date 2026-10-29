# Журнал сабмитов

Каждая строка — реальная отправка на Kaggle (требования — notes/plan.md, раздел 9).
Команды отправки: `uv run python scripts/submit.py send <file> --exp <ID> --msg "..."`.
Лимит соревнования — **10 сабмитов в день**, поэтому сабмитим только содержательные варианты.

| Дата | Exp | Git | Файл | Local R@10 | Kaggle R@10 | Сообщение | Вывод |
|---|---|---|---|---:|---:|---|---|
| 2026-10-29 | E00 | 719e65f+ | submissions/E00_submission_sample.csv | — | 0.0018 | sanity check: submission sample as-is (upload через web UI) | pipeline подтверждён от формата до скоринга; нижняя планка |
| 2026-10-29 | E01 | 8463b38 | submissions/E01_popularity_count.csv | 0.0367 (9010) | 0.0359 | popularity count, обучение на полных events | −2% к локальной 9010-панели: протокол «хвостовой», делитель \|gt\| |
| 2026-10-29 | E04 | 8463b38 | submissions/E04_als_rating.csv | 0.0658 (9010) | 0.0783 | ALS rating-веса, обучение на полных events | +19% к 9010-панели; masked-гипотеза отвергнута (ждали ~0.17) |
| 2026-10-29 | E05 | 8ee7600 | submissions/E05_sasrec.csv | 0.1602 (9010) | 0.2425 | SASRec e64/len100/ep15 на полных events | baseline_5 (0.0821) пробит ×3; SASRec — ядро финального решения |
