#!/usr/bin/env python3
"""
Просмотр нарезки одного документа тем же сплиттером, что и extract_long_answer.py.

Примеры:
  python scripts/preview_chunks.py -i simpleqa_verified/simpleqa_verified_with_documents.csv --row 2 --doc 0
  python scripts/preview_chunks.py --text-file path/to/article.txt
  python scripts/preview_chunks.py --text $'Первый абзац.\\n\\nВторой абзац.'
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# корень репозитория = родитель каталога scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extract_long_answer import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    RECURSIVE_SEPARATORS,
    document_word_chunks,
    parse_documents,
)


def _load_doc_from_csv(path: Path, row_index: int, doc_index: int) -> tuple[str, str]:
    """row_index — 0-based по строкам данных (первая после header = 0)."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i == row_index:
                q = (r.get("problem") or "").strip()
                docs = parse_documents(r.get("documents", ""))
                if doc_index < 0 or doc_index >= len(docs):
                    raise SystemExit(
                        f"В строке {row_index} всего документов: {len(docs)} (запрошен doc={doc_index})"
                    )
                return docs[doc_index], q
    raise SystemExit(f"В файле нет строки данных с индексом {row_index}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Предпросмотр чанков документа")
    parser.add_argument(
        "-i", "--input",
        type=Path,
        help="CSV с колонкой documents (как у пайплайна)",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        help="Сырой текстовый файл вместо CSV",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Индекс строки данных в CSV (0 = первая строка после header)",
    )
    parser.add_argument(
        "--doc",
        type=int,
        default=0,
        help="Индекс документа в списке documents (0-based)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Передать текст аргументом (короткие тесты)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=220,
        help="Сколько символов печатать с начала каждого чанка",
    )
    args = parser.parse_args()

    src_csv = args.input is not None
    src_file = args.text_file is not None
    src_text = bool(args.text)
    if sum([src_csv, src_file, src_text]) != 1:
        parser.error("Укажите ровно один источник: -i, --text-file или --text")

    if args.text_file is not None:
        doc = args.text_file.read_text(encoding="utf-8", errors="replace")
        problem = ""
    elif src_text:
        doc = args.text
        problem = ""
    else:
        assert args.input is not None
        if not args.input.is_file():
            raise SystemExit(f"Нет файла: {args.input}")
        doc, problem = _load_doc_from_csv(args.input, args.row, args.doc)

    print("=== Параметры (extract_long_answer) ===")
    print(f"CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")
    print(f"RECURSIVE_SEPARATORS = {RECURSIVE_SEPARATORS!r}")
    print()
    if problem:
        p = problem[:200] + ("…" if len(problem) > 200 else "")
        print("=== Вопрос (усечено) ===")
        print(p)
        print()
    print("=== Документ ===")
    print(f"длина: {len(doc)} символов, слов ~{len(doc.split())}")
    print(f"абзацев \\n\\n: {doc.count(chr(10)+chr(10))}, переносов \\n: {doc.count(chr(10))}")
    print()

    chunks = document_word_chunks(doc)
    print(f"=== Чанков: {len(chunks)} ===\n")
    n = args.preview
    for i, ch in enumerate(chunks):
        head = ch[:n].replace("\n", "⏎")
        more = "…" if len(ch) > n else ""
        print(f"--- chunk {i} | {len(ch)} sym | {len(ch.split())} words ---")
        print(head + more)
        print()


if __name__ == "__main__":
    main()
