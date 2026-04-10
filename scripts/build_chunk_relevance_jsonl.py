#!/usr/bin/env python3
"""
Строит chunk_relevance.jsonl: для каждого вопроса — все документы и все чанки
с меткой relevant 1/0 (только id чанков, без текстов).

Релевантность: текст чанка после normalize_for_dedup совпадает с одним из
фрагментов long_answer (то же правило, что при сборке long_answer в extract_long_answer.py).

Идентификаторы doc_id / chunk_id совпадают с build_chunks_collection_jsonl.py
(хэши от URL и текста чанка, см. corpus_ids.py).

Запуск:
  python scripts/build_chunk_relevance_jsonl.py \\
    -i simpleqa_verified/simpleqa_verified_with_long_answer.csv \\
    -o simpleqa_verified/chunk_relevance.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(10**7)

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from corpus_ids import make_chunk_id, make_doc_id  # noqa: E402
from extract_long_answer import normalize_for_dedup  # noqa: E402


def parse_urls(urls_str: str) -> list[str]:
    if not urls_str or not str(urls_str).strip():
        return []
    urls = []
    for part in str(urls_str).split(","):
        url = part.strip().rstrip(")")
        if url.startswith("http"):
            urls.append(url)
    return urls


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


def long_answer_norm_set(raw: str) -> set[str]:
    try:
        fr = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return set()
    if not isinstance(fr, list):
        return set()
    s: set[str] = set()
    for x in fr:
        t = str(x).strip()
        if not t:
            continue
        n = normalize_for_dedup(t)
        if n:
            s.add(n)
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV → chunk_relevance.jsonl (id + 1/0)")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=ROOT / "simpleqa_verified" / "simpleqa_verified_with_long_answer.csv",
        help="CSV с колонками urls, chunks, long_answer",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "simpleqa_verified" / "chunk_relevance.jsonl",
        help="По одной JSON-строке на вопрос",
    )
    parser.add_argument("--question-id-prefix", default="simpleqa_q_")
    parser.add_argument("--question-id-width", type=int, default=6)
    parser.add_argument("--doc-id-prefix", default="simple_qa_doc_")
    parser.add_argument("--chunk-id-prefix", default="simple_qa_chunk_")
    parser.add_argument("--hash-len", type=int, default=16)
    args = parser.parse_args()

    inp = args.input.resolve()
    out = args.output.resolve()
    if not inp.is_file():
        print(f"Ошибка: файл не найден: {inp}", file=sys.stderr)
        sys.exit(1)

    required = ("urls", "chunks", "long_answer")
    lines_out = 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(inp, encoding="utf-8", newline="") as fin, open(out, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            print("Ошибка: CSV без заголовка", file=sys.stderr)
            sys.exit(1)
        for col in required:
            if col not in reader.fieldnames:
                print(f"Ошибка: нет колонки {col!r}", file=sys.stderr)
                sys.exit(1)

        for row_idx, row in enumerate(reader, start=1):
            question_id = f"{args.question_id_prefix}{row_idx:0{args.question_id_width}d}"
            rel = long_answer_norm_set(row.get("long_answer", ""))
            urls = parse_urls(row.get("urls", ""))
            chunks_per_doc = load_chunks_matrix(row.get("chunks", ""))

            docs_out: list[dict] = []
            n_docs = max(len(urls), len(chunks_per_doc))
            for j in range(n_docs):
                url = urls[j] if j < len(urls) else ""
                cells = chunks_per_doc[j] if j < len(chunks_per_doc) else []

                doc_id = make_doc_id(
                    url, question_id, j, prefix=args.doc_id_prefix, hex_len=args.hash_len
                )
                chunk_rows: list[dict] = []
                for ci, cell in enumerate(cells):
                    text = cell.get("text")
                    if text is None:
                        continue
                    cid = make_chunk_id(
                        url,
                        question_id,
                        j,
                        ci,
                        text,
                        prefix=args.chunk_id_prefix,
                        hex_len=args.hash_len,
                    )
                    nt = normalize_for_dedup(str(text))
                    chunk_rows.append({"id": cid, "relevant": 1 if (nt and nt in rel) else 0})

                docs_out.append({"doc_id": doc_id, "chunks": chunk_rows})

            rec = {"question_id": question_id, "documents": docs_out}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lines_out += 1

    print(f"Вопросов (строк JSONL): {lines_out} → {out}")


if __name__ == "__main__":
    main()
