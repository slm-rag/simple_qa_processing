#!/usr/bin/env python3
"""
Строит chunks_collection.jsonl из CSV с колонками urls и chunks
(например simpleqa_verified_with_long_answer.csv).

Одна строка JSONL = один документ (один URL в контексте одного вопроса).

doc_id и chunk_id стабильны: SHA256 от нормализованного URL (и текста чанка),
см. scripts/corpus_ids.py — одинаковый URL → один doc_id в разных вопросах.

Запуск:
  python scripts/build_chunks_collection_jsonl.py \\
    -i simpleqa_verified/simpleqa_verified_with_long_answer.csv \\
    -o simpleqa_verified/chunks_collection.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

csv.field_size_limit(10**7)

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from corpus_ids import make_chunk_id, make_doc_id  # noqa: E402


def parse_urls(urls_str: str) -> list[str]:
    if not urls_str or not str(urls_str).strip():
        return []
    urls = []
    for part in str(urls_str).split(","):
        url = part.strip().rstrip(")")
        if url.startswith("http"):
            urls.append(url)
    return urls


def title_from_url(url: str) -> str:
    """Название страницы для Wikipedia; иначе пустая строка."""
    if not url:
        return ""
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        path = p.path or ""
        if "wikipedia.org" in host and "/wiki/" in path:
            raw = path.split("/wiki/", 1)[-1].split("#")[0]
            return unquote(raw).replace("_", " ")
    except Exception:
        pass
    return ""


def load_chunks_matrix(raw: str) -> list[list[dict]]:
    if not raw or not str(raw).strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[list[dict]] = []
    for item in data:
        if isinstance(item, list):
            row: list[dict] = []
            for cell in item:
                if isinstance(cell, dict) and "text" in cell:
                    row.append(cell)
            out.append(row)
        else:
            out.append([])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV (urls + chunks) → chunks_collection.jsonl")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=ROOT / "simpleqa_verified" / "simpleqa_verified_with_long_answer.csv",
        help="Входной CSV с колонками urls, chunks",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "simpleqa_verified" / "chunks_collection.jsonl",
        help="Выходной JSONL (одна запись на документ)",
    )
    parser.add_argument(
        "--question-id-prefix",
        default="simpleqa_q_",
        help="Как в build_qa_pairs_jsonl.py (--id-prefix)",
    )
    parser.add_argument(
        "--question-id-width",
        type=int,
        default=6,
        help="Ширина номера в question_id",
    )
    parser.add_argument(
        "--doc-id-prefix",
        default="simple_qa_doc_",
        help="Префикс doc_id",
    )
    parser.add_argument(
        "--chunk-id-prefix",
        default="simple_qa_chunk_",
        help="Префикс id чанка",
    )
    parser.add_argument(
        "--hash-len",
        type=int,
        default=16,
        help="Длина hex-суффикса id (от SHA256)",
    )
    args = parser.parse_args()

    inp = args.input.resolve()
    out = args.output.resolve()
    if not inp.is_file():
        print(f"Ошибка: файл не найден: {inp}", file=sys.stderr)
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)

    lines_out = 0

    with open(inp, encoding="utf-8", newline="") as fin, open(out, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            print("Ошибка: CSV без заголовка", file=sys.stderr)
            sys.exit(1)
        for need in ("urls", "chunks"):
            if need not in reader.fieldnames:
                print(f"Ошибка: нет колонки {need!r}", file=sys.stderr)
                sys.exit(1)

        for row_idx, row in enumerate(reader, start=1):
            question_id = f"{args.question_id_prefix}{row_idx:0{args.question_id_width}d}"
            urls = parse_urls(row.get("urls", ""))
            chunks_per_doc = load_chunks_matrix(row.get("chunks", ""))

            n_docs = max(len(urls), len(chunks_per_doc))
            for j in range(n_docs):
                url = urls[j] if j < len(urls) else ""
                cells = chunks_per_doc[j] if j < len(chunks_per_doc) else []

                doc_id = make_doc_id(
                    url, question_id, j, prefix=args.doc_id_prefix, hex_len=args.hash_len
                )
                chunk_list: list[dict] = []
                for ci, cell in enumerate(cells):
                    text = cell.get("text")
                    if text is None:
                        continue
                    chunk_list.append(
                        {
                            "id": make_chunk_id(
                                url,
                                question_id,
                                j,
                                ci,
                                text,
                                prefix=args.chunk_id_prefix,
                                hex_len=args.hash_len,
                            ),
                            "text": text,
                        }
                    )

                rec = {
                    "doc_id": doc_id,
                    "url": url,
                    "title": title_from_url(url),
                    "question_id": question_id,
                    "chunks": chunk_list,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                lines_out += 1

    print(f"Записано документов (строк JSONL): {lines_out} → {out}")


if __name__ == "__main__":
    main()
