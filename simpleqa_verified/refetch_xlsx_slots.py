#!/usr/bin/env python3
"""
Точечно перезаписывает элементы documents[] для URL с .xlsx: заново качает файл и
пишет текст через openpyxl (как в download_documents_full.OptimizedDocumentDownloader).

Не перезагружает весь датасет — только указанные строки или авто-поиск «сломанных»
слотов (тело начинается с PK после сохранения сырых байтов как текст).

Примеры:

  # строка данных с индексом 382 (вопрос simpleqa_q_000383), все слоты с .xlsx
  python simpleqa_verified/refetch_xlsx_slots.py \\
    -i simpleqa_verified/simpleqa_verified_with_documents.csv \\
    --data-row 382

  # по номеру вопроса как в question_id (…_000383 → 383)
  python simpleqa_verified/refetch_xlsx_slots.py -i ... --question-num 383

  # все строки CSV, где URL оканчивается на .xlsx и текст слота похож на сырой ZIP
  python simpleqa_verified/refetch_xlsx_slots.py -i ... --auto-binary

После обновления CSV пересчитайте long_answer и jsonl:
  python extract_long_answer.py ...
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple

csv.field_size_limit(sys.maxsize)

_VERIFIED_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _VERIFIED_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


def _get_downloader():
    from download_documents_full import OptimizedDocumentDownloader  # noqa: WPS433

    return OptimizedDocumentDownloader(timeout=45, delay=1.5, max_workers=3)


def parse_urls(urls_str: str) -> List[str]:
    if not urls_str or not str(urls_str).strip():
        return []
    urls = []
    for part in str(urls_str).split(","):
        url = part.strip().rstrip(")")
        if url.startswith("http"):
            urls.append(url)
    return urls


def load_documents_list(raw: str) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    raw_stripped = raw.strip()
    try:
        docs = json.loads(raw_stripped)
    except json.JSONDecodeError:
        try:
            docs = ast.literal_eval(raw_stripped)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(docs, list):
        return []
    out: List[str] = []
    for x in docs:
        out.append("" if x is None else str(x))
    return out


def url_is_xlsx(url: str) -> bool:
    path = url.lower().split("?", 1)[0].rstrip("/")
    return path.endswith(".xlsx")


def doc_looks_like_raw_zip_bytes_as_str(doc: str) -> bool:
    """Сырые байты .xlsx попали в CSV как строка — видна сигнатура ZIP."""
    if not doc or len(doc) < 8:
        return False
    return doc.startswith("PK")


def align_docs_to_urls(docs: List[str], n_urls: int) -> List[str]:
    out = list(docs[:n_urls])
    while len(out) < n_urls:
        out.append("")
    return out


def slots_to_refetch_for_row(
    urls: Sequence[str],
    docs: Sequence[str],
    *,
    row_indices_mode: str,
    only_binary: bool,
) -> List[int]:
    """
    row_indices_mode:
      'explicit' — все слоты с .xlsx URL (перезапись целиком).
      'auto_binary' — только .xlsx и похожий на сырой бинарник текст.
    """
    n = len(urls)
    bad: List[int] = []
    for i in range(n):
        if not url_is_xlsx(urls[i]):
            continue
        text = docs[i] if i < len(docs) else ""
        if row_indices_mode == "auto_binary":
            if only_binary and not doc_looks_like_raw_zip_bytes_as_str(text):
                continue
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
        url = urls[i]
        downloader.document_cache.pop(downloader.get_url_hash(url), None)
        try:
            content = downloader.download_document(url)
        except Exception as e:
            logger.warning("Ошибка загрузки %s: %s", url[:88], e)
            content = None
        text = (content or "").strip()
        out[i] = text
        logger.info("Слот %d URL …%s → длина текста %d", i, url[-48:], len(text))
        time.sleep(1.5)
    return out


def process_csv(
    input_path: Path,
    output_path: Path,
    *,
    restrict_rows: Optional[Set[int]],
    auto_binary_all: bool,
    dry_run: bool,
    backup: bool,
) -> Tuple[int, int, int]:
    """
    Returns:
        (rows_total, rows_touched, slots_refetched)
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

    for idx, row in enumerate(rows):
        rows_total += 1
        if restrict_rows is not None and idx not in restrict_rows:
            continue
        urls = parse_urls(row.get("urls", ""))
        if not urls:
            continue
        raw_docs = load_documents_list(row.get("documents", ""))
        docs = align_docs_to_urls(raw_docs, len(urls))
        if len(raw_docs) > len(urls):
            logger.warning(
                "Строка %d original_index=%s: documents длиннее urls, обрезаю",
                idx,
                row.get("original_index"),
            )

        mode = "auto_binary" if auto_binary_all else "explicit"
        bad = slots_to_refetch_for_row(
            urls,
            docs,
            row_indices_mode=mode,
            only_binary=True,
        )
        if not bad:
            continue

        rows_touched += 1
        logger.info(
            "Строка %d (%s): перезагрузка слотов %s",
            idx,
            row.get("original_index"),
            bad,
        )

        if dry_run:
            slots_refetched += len(bad)
            continue

        assert downloader is not None
        docs = refetch_row_slots(downloader, urls, docs, bad)
        slots_refetched += len(bad)
        row["documents"] = json.dumps(docs, ensure_ascii=False)

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

    parser = argparse.ArgumentParser(
        description="Точечная перезагрузка .xlsx → текст в колонке documents"
    )
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
        help="По умолчанию = input (с резервной копией .bak)",
    )
    parser.add_argument(
        "--data-row",
        type=int,
        action="append",
        default=[],
        metavar="N",
        help="Индекс строки данных (0 = первая строка после заголовка), можно повторять",
    )
    parser.add_argument(
        "--question-num",
        type=int,
        action="append",
        default=[],
        metavar="K",
        help="Номер как в simpleqa_q_%06d (например 383 → строка с индексом 382)",
    )
    parser.add_argument(
        "--auto-binary",
        action="store_true",
        help="Пробежать все строки: перезагрузить только .xlsx-слоты с «PK» в тексте",
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

    if args.auto_binary:
        if args.data_row or args.question_num:
            logging.warning(
                "Указан --auto-binary: опции --data-row / --question-num игнорируются"
            )
        restrict = None
    else:
        restrict = set()
        for n in args.data_row:
            restrict.add(n)
        for q in args.question_num:
            restrict.add(q - 1)
        if not restrict:
            raise SystemExit(
                "Укажите --data-row / --question-num или режим --auto-binary "
                "(см. python refetch_xlsx_slots.py --help)"
            )

    total, touched, slots = process_csv(
        inp,
        out,
        restrict_rows=restrict,
        auto_binary_all=bool(args.auto_binary),
        dry_run=args.dry_run,
        backup=not args.no_backup and out == inp,
    )
    print(
        f"Строк всего: {total}; затронуто строк: {touched}; перезагружено слотов: {slots}"
    )
    if args.dry_run:
        print("(dry-run: файлы не менялись)")
    else:
        print(f"Записано: {out}")
        print("Дальше: extract_long_answer.py и сборка jsonl.")


if __name__ == "__main__":
    main()
