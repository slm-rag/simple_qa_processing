#!/usr/bin/env python3
"""
Строит qa_pairs.jsonl из CSV simpleqa_verified (колонки original_index, problem, answer).

Пример строки вывода:
  {"question_id":"simpleqa_q_000001","original_index":"5","question":"...","answer":["..."]}

Запуск:
  python scripts/build_qa_pairs_jsonl.py \\
    -i simpleqa_verified/simpleqa_verified_with_long_answer.csv \\
    -o simpleqa_verified/qa_pairs.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json

csv.field_size_limit(10**7)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV simpleqa_verified → qa_pairs.jsonl")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=ROOT / "simpleqa_verified" / "simpleqa_verified_with_long_answer.csv",
        help="Входной CSV",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "simpleqa_verified" / "qa_pairs.jsonl",
        help="Выходной JSONL",
    )
    parser.add_argument(
        "--id-prefix",
        default="simpleqa_q_",
        help='Префикс question_id (по умолчанию "simpleqa_q_")',
    )
    parser.add_argument(
        "--id-width",
        type=int,
        default=6,
        help="Ширина нумерации в question_id (default: 6 → 000001)",
    )
    args = parser.parse_args()

    inp = args.input.resolve()
    out = args.output.resolve()
    if not inp.is_file():
        print(f"Ошибка: файл не найден: {inp}", file=sys.stderr)
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(inp, encoding="utf-8", newline="") as fin, open(out, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        required = ("original_index", "problem", "answer")
        if reader.fieldnames is None:
            print("Ошибка: пустой или без заголовка CSV", file=sys.stderr)
            sys.exit(1)
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            print(f"Ошибка: в CSV нет колонок: {missing}", file=sys.stderr)
            sys.exit(1)

        for n, row in enumerate(reader, start=1):
            rec = {
                "question_id": f"{args.id_prefix}{n:0{args.id_width}d}",
                "original_index": str(row.get("original_index", "")),
                "question": row.get("problem") or "",
                "answer": [row.get("answer") or ""],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Записано строк: {n} → {out}")


if __name__ == "__main__":
    main()
