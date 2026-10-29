#!/usr/bin/env python
"""Отправка сабмитов в HSE RecSys Challenge 2026 и журнал сабмитов.

Перед отправкой файл проходит sanity-check (src/recsys/submission.py).
Каждый сабмит записывается в notes/submissions.md с experiment id,
git commit и кратким описанием (notes/plan.md, раздел 9); Kaggle Recall@10
подставляется после скоринга (сразу через `--wait` либо позже через `sync`).

Запуск из корня проекта через uv:

    uv run python scripts/submit.py validate data/submission_sample.csv
    uv run python scripts/submit.py send submissions/E00.csv --exp E00 --msg "..."
    uv run python scripts/submit.py status -n 5
    uv run python scripts/submit.py sync

Аутентификация — Kaggle CLI 2.x, токен `~/.kaggle/access_token`.
Лимит соревнования — 10 сабмитов в день.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recsys.submission import COMPETITION, load_submission, validate_submission

LOG_PATH = PROJECT_ROOT / "notes" / "submissions.md"
USERS_PATH = PROJECT_ROOT / "data" / "submission_sample.csv"
ITEMS_PATH = PROJECT_ROOT / "data" / "item_features.csv"


def kaggle(args: list[str], capture: bool = True) -> str:
    """Вызывает kaggle CLI (должен быть в PATH окружения `uv run`)."""
    binary = shutil.which("kaggle")
    if binary is None:
        sys.exit("kaggle CLI не найден: запускайте через `uv run python scripts/submit.py ...`")
    result = subprocess.run(
        [binary, *args], capture_output=capture, text=True
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout) if capture else "см. вывод выше"
        sys.exit(f"kaggle {' '.join(args)} завершился с ошибкой:\n{detail}")
    return result.stdout or ""


def check_format(path: str) -> None:
    """Sanity-check файла сабмита против данных соревнования."""
    sample = load_submission(USERS_PATH)
    user_ids = {int(u) for u in sample["user_id"]}
    item_ids = set(pd_read_ids(ITEMS_PATH))
    df = load_submission(path)
    stats = validate_submission(df, user_ids=user_ids, item_ids=item_ids)
    print(f"OK: {path} — {stats['users']} пользователей, {stats['unique_items']} уникальных фильмов")


def pd_read_ids(path: str) -> list[int]:
    import pandas as pd

    return pd.read_csv(path, usecols=["item_id"])["item_id"].astype(int).tolist()


def git_ref() -> str:
    """Короткий хэд HEAD с пометкой о незакоммиченных изменениях."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return f"{commit}+" if dirty else commit


def fetch_submissions(limit: int) -> list[dict]:
    """Последние сабмиты пользователя в нормализованном виде."""
    raw = kaggle(["competitions", "submissions", COMPETITION,
                  "--format", "json", "--page-size", str(limit)])
    entries = json.loads(raw)
    if isinstance(entries, dict):
        entries = entries.get("submissions", [entries])
    normalized = []
    for entry in entries:
        normalized.append({
            "date": _field(entry, "date", "submittedAt", "createdAt"),
            "message": _field(entry, "description", "message"),
            "status": _field(entry, "status", "state"),
            "score": _field(entry, "publicScore", "score"),
        })
    return normalized


def _field(entry: dict, *names: str):
    for name in names:
        if entry.get(name) not in (None, ""):
            return entry[name]
    return None


def cmd_send(args: argparse.Namespace) -> None:
    check_format(args.file)
    message = f"{args.exp}: {args.msg}"
    print(f"Отправка: {args.file}\nСообщение: {message!r}\nGit: {git_ref()}")

    submit_args = [
        "competitions", "submit", COMPETITION,
        "-f", args.file, "-m", message, "--wait", str(args.wait),
    ]
    if args.dry_run:
        print(f"[dry-run] kaggle {' '.join(submit_args)}")
        score = None
    else:
        kaggle(submit_args, capture=False)
        score = next(
            (s["score"] for s in fetch_submissions(5)
             if s["message"] == message and s["score"] is not None),
            None,
        )

    kaggle_col = score if score is not None else "—"
    local_col = args.local if args.local is not None else "—"
    row = (f"| {date.today().isoformat()} | {args.exp} | {git_ref()} | "
           f"{args.file} | {local_col} | {kaggle_col} | {args.msg} | — |\n")
    with open(LOG_PATH, "a") as f:
        f.write(row)
    print(f"Записано в {LOG_PATH}: Kaggle R@10 = {kaggle_col}")
    if score is None and not args.dry_run:
        print("Скор ещё не готов — заполните позже: uv run python scripts/submit.py sync")


def cmd_status(args: argparse.Namespace) -> None:
    subs = fetch_submissions(args.n)
    if not subs:
        print("Сабмитов пока нет.")
        return
    for s in subs:
        print(f"{s['date']}  {str(s['status']):<10}  score={s['score']}  {s['message']}")


def cmd_sync(args: argparse.Namespace) -> None:
    """Проставляет Kaggle-скор в строки журнала с пустым значением."""
    lines = LOG_PATH.read_text().splitlines(keepends=True)
    subs = [s for s in fetch_submissions(50) if s["score"] is not None]
    updated = 0
    for i, line in enumerate(lines):
        if "| — |" not in line or "| Дата |" in line:
            continue
        for s in subs:
            if s["message"] and s["message"].split(":", 1)[-1].strip() in line:
                parts = line.split("|")
                if parts[6].strip() == "—":
                    parts[6] = f" {s['score']} "
                    lines[i] = "|".join(parts)
                    updated += 1
                break
    LOG_PATH.write_text("".join(lines))
    print(f"Обновлено строк: {updated}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True, dest="subcommand")

    p_validate = sub.add_parser("validate", help="sanity-check файла сабмита")
    p_validate.add_argument("file")

    p_send = sub.add_parser("send", help="валидация + отправка + запись в журнал")
    p_send.add_argument("file")
    p_send.add_argument("--exp", required=True, help="experiment id, например E00")
    p_send.add_argument("--msg", required=True, help="краткое описание изменения")
    p_send.add_argument("--local", default=None, help="локальная метрика Recall@10")
    p_send.add_argument("--wait", type=int, default=600,
                        help="секунд ожидания скоринга (0 — не ждать)")
    p_send.add_argument("--dry-run", action="store_true")

    p_status = sub.add_parser("status", help="последние сабмиты и скоры")
    p_status.add_argument("-n", type=int, default=5)

    sub.add_parser("sync", help="проставить Kaggle-скоры в журнал")

    args = parser.parse_args()
    {"validate": lambda: check_format(args.file),
     "send": lambda: cmd_send(args),
     "status": lambda: cmd_status(args),
     "sync": lambda: cmd_sync(args)}[args.subcommand]()


if __name__ == "__main__":
    main()
