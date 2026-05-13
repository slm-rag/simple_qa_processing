#!/usr/bin/env python3
"""
Точечно пересачивает только те элементы documents[], где текст совпадает с
заглушкой WikiRoulette (или пустой при --retry-empty). Остальные строки CSV
и соседние слоты не меняются.

Не перезагружает весь датасет — только затронутые URL в затронутых строках.

После успешного обновления CSV пересоберите long_answer и jsonl, например:
  python extract_long_answer.py ...
  python scripts/build_chunks_collection_jsonl.py ...
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, List, Sequence, Tuple

csv.field_size_limit(sys.maxsize)

_VERIFIED_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _VERIFIED_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


def _get_downloader():
    from download_documents_full import OptimizedDocumentDownloader  # noqa: WPS433

    return OptimizedDocumentDownloader(timeout=45, delay=1.5, max_workers=3)

DEFAULT_PLACEHOLDER = "WikiRoulette — Random Wikipedia Pages"

# Известные опечатки в исходных URL (ключ — как в CSV, значение — перед запросом)
URL_REWRITES: dict[str, str] = {
    "https://en.wikipedia.org/wiki/Douglas_Bennett_(cricketer": (
        "https://en.wikipedia.org/wiki/Douglas_Bennett_(cricketer)"
    ),
}


def parse_urls(urls_str: str) -> List[str]:
    if not urls_str or not str(urls_str).strip():
        return []
    urls = []
    for part in str(urls_str).split(","):
        url = part.strip().rstrip(")")
        if url.startswith("http"):
            urls.append(url)
    return urls


def apply_url_rewrite(url: str) -> str:
    u = url.strip()
    return URL_REWRITES.get(u, u)


def load_documents_list(raw: str) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    try:
        docs = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(docs, list):
        return []
    return ["" if x is None else str(x) for x in docs]


def is_placeholder(text: str, placeholder: str) -> bool:
    t = (text or "").strip()
    p = placeholder.strip()
    return bool(t) and (t == p or p in t)


def align_docs_to_urls(docs: List[str], n_urls: int) -> List[str]:
    out = list(docs[:n_urls])
    while len(out) < n_urls:
        out.append("")
    return out


def slots_to_refetch(
    docs: Sequence[str],
    urls: Sequence[str],
    *,
    placeholder: str,
    retry_empty: bool,
) -> List[int]:
    n = len(urls)
    bad: List[int] = []
    for i in range(n):
        text = docs[i] if i < len(docs) else ""
        if is_placeholder(text, placeholder):
            bad.append(i)
            continue
        if retry_empty and not (text or "").strip():
            bad.append(i)
    return bad


def refetch_row_slots(
    downloader: Any,
    urls: List[str],
    docs: List[str],
    indices: List[int],
) -> List[str]:
    out = list(docs)
    for i in indices:
        url = apply_url_rewrite(urls[i])
        if url != urls[i]:
            logger.info("URL rewrite: %s -> %s", urls[i][:70], url[:70])
        try:
            content = downloader.download_document(url)
        except Exception as e:
            logger.warning("Ошибка загрузки %s: %s", url[:80], e)
            content = None
        text = (content or "").strip()
        out[i] = text
        logger.info(
            "Слот %d: новая длина %d символов",
            i,
            len(text),
        )
        time.sleep(2.0)
    return out


def process_csv(
    input_path: Path,
    output_path: Path,
    *,
    placeholder: str,
    retry_empty: bool,
    dry_run: bool,
    backup: bool,
) -> Tuple[int, int, int]:
    """
    Returns:
        (rows_total, rows_with_placeholder, slots_refetched)
    """
    rows_total = 0
    rows_touched = 0
    slots_refetched = 0

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "documents" not in fieldnames or "urls" not in fieldnames:
            raise SystemExit("CSV должен содержать колонки urls и documents")
        rows: List[dict[str, Any]] = list(reader)

    downloader = None
    if not dry_run:
        downloader = _get_downloader()

    for row in rows:
        rows_total += 1
        urls = parse_urls(row.get("urls", ""))
        if not urls:
            continue
        docs = align_docs_to_urls(load_documents_list(row.get("documents", "")), len(urls))
        if len(load_documents_list(row.get("documents", ""))) > len(urls):
            logger.warning(
                "original_index=%s: documents длиннее urls, обрезаю",
                row.get("original_index"),
            )

        if not any(is_placeholder(d, placeholder) for d in docs):
            continue

        bad = slots_to_refetch(docs, urls, placeholder=placeholder, retry_empty=retry_empty)
        if not bad:
            continue

        rows_touched += 1
        logger.info(
            "original_index=%s: перезагрузка слотов %s из %d URL",
            row.get("original_index"),
            bad,
            len(urls),
        )

        if dry_run:
            slots_refetched += len(bad)
            continue

        assert downloader is not None
        docs = refetch_row_slots(downloader, urls, docs, bad)
        slots_refetched += len(bad)
        row["documents"] = json.dumps(docs, ensure_ascii=False)
        # Переписанные URL в CSV не меняем — важен только успешный текст в documents.

    if dry_run:
        return rows_total, rows_touched, slots_refetched

    if backup and output_path.resolve() == input_path.resolve():
        bak = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, bak)
        logger.info("Резервная копия: %s", bak)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return rows_total, rows_touched, slots_refetched


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Точечная догрузка слотов WikiRoulette")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=_VERIFIED_DIR / "simpleqa_verified_with_documents.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="По умолчанию = input (с --backup)",
    )
    parser.add_argument(
        "--placeholder",
        default=DEFAULT_PLACEHOLDER,
        help="Подстрогое совпадение: весь слот или вхождение этой строки",
    )
    parser.add_argument(
        "--retry-empty",
        action="store_true",
        help="В строках, где уже есть заглушка, также перекачать пустые слоты",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Не писать .bak при записи в тот же файл, что и -i",
    )
    args = parser.parse_args()

    inp = args.input.resolve()
    out = (args.output or args.input).resolve()
    if not inp.is_file():
        raise SystemExit(f"Нет файла: {inp}")

    total, touched, slots = process_csv(
        inp,
        out,
        placeholder=args.placeholder,
        retry_empty=args.retry_empty,
        dry_run=args.dry_run,
        backup=not args.no_backup and out == inp,
    )
    print(
        f"Строк всего: {total}; с заглушкой (будут затронуты): {touched}; "
        f"слотов к перекачке: {slots}"
    )
    if args.dry_run:
        print("(dry-run: файлы не менялись)")
    else:
        print(f"Записано: {out}")
        print("Дальше: обновите long_answer и jsonl по вашему пайплайну.")


if __name__ == "__main__":
    main()
