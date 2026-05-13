#!/usr/bin/env python3
"""
Обнуляет один слот в колонке documents (пустая строка), не трогая urls и порядок слотов.

Индексация слота — с нуля (0 = первый URL, 1 = второй, …).

Пример (бинарный WEBP во втором слоте вопроса 188):
  python simpleqa_verified/clear_document_slot.py \\
    -i simpleqa_verified/simpleqa_verified_with_documents.csv \\
    --question-num 188 --slot 1

После правки пересоберите long_answer / chunks при необходимости.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, List

csv.field_size_limit(sys.maxsize)

_VERIFIED_DIR = Path(__file__).resolve().parent


def _load_docs(raw: str) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    try:
        d = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError("documents не JSON и не Python-list литерал")
    if not isinstance(d, list):
        raise ValueError("documents должен быть списком")
    return ["" if x is None else str(x) for x in d]


def _dump_docs(docs: List[str]) -> str:
    return json.dumps(docs, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Очистить один слот documents[i]")
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=_VERIFIED_DIR / "simpleqa_verified_with_documents.csv",
    )
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument(
        "--question-num",
        type=int,
        required=True,
        help="Номер вопроса как в simpleqa_q_%06d (первая строка данных = 1)",
    )
    p.add_argument(
        "--slot",
        type=int,
        required=True,
        help="Индекс слота с нуля (1 = второй URL)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Не создавать .bak при записи в тот же файл что -i",
    )
    args = p.parse_args()

    inp = args.input.resolve()
    out = (args.output or args.input).resolve()
    row_idx = args.question_num - 1
    slot = args.slot

    if row_idx < 0:
        raise SystemExit("question-num должен быть ≥ 1")
    if slot < 0:
        raise SystemExit("slot должен быть ≥ 0")

    if not inp.is_file():
        raise SystemExit(f"Нет файла: {inp}")

    with inp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "documents" not in fieldnames:
            raise SystemExit("Нужна колонка documents")
        rows: List[dict[str, Any]] = list(reader)

    if row_idx >= len(rows):
        raise SystemExit(f"В файле только {len(rows)} строк данных; question-num слишком большой")

    row = rows[row_idx]
    docs = _load_docs(row.get("documents", ""))
    if slot >= len(docs):
        raise SystemExit(f"В строке только {len(docs)} слотов; slot {slot} вне диапазона")

    prev_len = len(docs[slot])
    if args.dry_run:
        print(
            f"dry-run: строка data-row {row_idx} (question-num {args.question_num}), "
            f"slot {slot}: было {prev_len} символов → будет \"\""
        )
        return

    docs[slot] = ""
    row["documents"] = _dump_docs(docs)

    if not args.no_backup and out == inp:
        bak = inp.with_suffix(inp.suffix + ".bak")
        shutil.copy2(inp, bak)
        print(f"Резервная копия: {bak}")

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(
        f"Готово: question-num {args.question_num}, slot {slot}: "
        f"очищено (было {prev_len} символов) → {out}"
    )


if __name__ == "__main__":
    main()
