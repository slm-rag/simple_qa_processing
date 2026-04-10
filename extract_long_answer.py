#!/usr/bin/env python3
"""
Извлечение релевантных фрагментов (long_answer) из документов для каждого вопроса.
- Документ режется рекурсивным сплиттером (как LangChain RecursiveCharacterTextSplitter):
  сначала по абзацам/строкам, затем предложениям, словам, символам; в поле chunks — JSON:
  [ [ {"id": 0, "text": "..."}, ... ], [...] ] (id сквозной по всем документам строки).
- Отбор кандидатов: BM25 по запросу «вопрос + ответ» + чанки с вхождением ответа; LLM один раз
  (или два при --only-empty-long-answer / relaxed) по короткому пулу.
- long_answer — список уникальных фрагментов (без дубликатов после нормализации).

LLM: OpenRouter API (openai/gpt-4o по умолчанию). Ключ: OPENROUTER_API_KEY в окружении
или в файле .env в корне репозитория (подгружается через python-dotenv).
  OPENROUTER_REQUEST_TIMEOUT — таймаут ответа в секундах (по умолчанию 180).
"""

import argparse
import ast
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Один чанк в JSON и для RAG: id уникален в пределах строки CSV (все документы вопроса)
ChunkRecord = Dict[str, Any]

# Увеличиваем лимит размера поля CSV
csv.field_size_limit(10**7)

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_VERIFIED_DOCS = _REPO_ROOT / "simpleqa_verified" / "simpleqa_verified_with_documents.csv"
_DEFAULT_VERIFIED_LONG = _REPO_ROOT / "simpleqa_verified" / "simpleqa_verified_with_long_answer.csv"

# Размер чанка и перекрытие в символах (как в LangChain; ~бывшие 400/80 слов)
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
# Порядок важен: сначала крупные единицы, чтобы сохранять структуру абзацев
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
MAX_CHUNKS_FOR_LLM = 8  # макс. чанков в одном запросе к LLM
BM25_TOP_K = 18  # сколько лучших по BM25 рассматривать при сборке пула (до среза)
MAX_CANDIDATES_NORMAL = 8  # размер пула → один вызов LLM
MAX_CANDIDATES_RELAXED = 16  # два окна по 8 при «втором проходе»

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-4o"
OPENROUTER_HTTP_RETRIES = 5


def _openrouter_request_timeout() -> float:
    """Секунды на HTTP-запрос; можно OPENROUTER_REQUEST_TIMEOUT в env или .env (после load_dotenv)."""
    return float(os.environ.get("OPENROUTER_REQUEST_TIMEOUT", "180"))


def parse_documents(documents_str: str) -> List[str]:
    """Парсит строку с документами в список."""
    if not documents_str or documents_str == '[]' or documents_str.strip() in ("''", '""'):
        return []
    try:
        parsed = ast.literal_eval(documents_str)
        if isinstance(parsed, list):
            return [str(doc) for doc in parsed if doc and str(doc).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
        return []
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(documents_str)
            if isinstance(parsed, list):
                return [str(doc) for doc in parsed if doc and str(doc).strip()]
            return []
        except Exception:
            return []


def normalize_for_dedup(text: str) -> str:
    """Нормализация текста для сравнения на дубликаты."""
    if not text:
        return ''
    return re.sub(r'\s+', ' ', text.strip())


def deduplicate_fragments(fragments: List[str]) -> List[str]:
    """Удаляет дубликаты после нормализации. Пустые фрагменты не добавляются."""
    seen = set()
    result = []
    for f in fragments:
        s = f.strip()
        if not s:
            continue
        norm = normalize_for_dedup(s)
        if norm and norm not in seen:
            seen.add(norm)
            result.append(s)
    return result


def _split_text_with_regex(text: str, separator: str) -> List[str]:
    """Деление по regex-шаблону разделителя (ветка keep_separator=False из LangChain)."""
    if not separator:
        return list(text)
    splits = re.split(separator, text)
    return [s for s in splits if s]


def _join_docs(docs: List[str], separator: str, *, strip_whitespace: bool = True) -> Optional[str]:
    text = separator.join(docs)
    if strip_whitespace:
        text = text.strip()
    return text if text else None


def _merge_splits(
    splits: Iterable[str],
    separator: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    length_function: Any = len,
    strip_whitespace: bool = True,
) -> List[str]:
    """Склейка мелких кусков в чанки ≤ chunk_size с перекрытием (логика LangChain TextSplitter)."""
    separator_len = length_function(separator)
    docs: List[str] = []
    current_doc: List[str] = []
    total = 0
    for d in splits:
        len_ = length_function(d)
        if total + len_ + (separator_len if len(current_doc) > 0 else 0) > chunk_size:
            if len(current_doc) > 0:
                doc = _join_docs(current_doc, separator, strip_whitespace=strip_whitespace)
                if doc is not None:
                    docs.append(doc)
                while total > chunk_overlap or (
                    total + len_ + (separator_len if len(current_doc) > 0 else 0)
                    > chunk_size
                    and total > 0
                ):
                    total -= length_function(current_doc[0]) + (
                        separator_len if len(current_doc) > 1 else 0
                    )
                    current_doc = current_doc[1:]
        current_doc.append(d)
        total += len_ + (separator_len if len(current_doc) > 1 else 0)
    doc = _join_docs(current_doc, separator, strip_whitespace=strip_whitespace)
    if doc is not None:
        docs.append(doc)
    return docs


def _recursive_split_text(
    text: str,
    separators: List[str],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    is_separator_regex: bool = False,
) -> List[str]:
    """
    Рекурсивное разбиение как RecursiveCharacterTextSplitter (langchain_text_splitters).
    """
    final_chunks: List[str] = []
    separator = separators[-1]
    new_separators: List[str] = []
    for i, s_ in enumerate(separators):
        separator_ = s_ if is_separator_regex else re.escape(s_)
        if not s_:
            separator = s_
            break
        if re.search(separator_, text):
            separator = s_
            new_separators = separators[i + 1 :]
            break

    separator_pattern = separator if is_separator_regex else re.escape(separator)
    splits = _split_text_with_regex(text, separator_pattern)
    merge_sep = separator
    good_splits: List[str] = []
    for s in splits:
        if len(s) < chunk_size:
            good_splits.append(s)
        else:
            if good_splits:
                merged_text = _merge_splits(
                    good_splits,
                    merge_sep,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                final_chunks.extend(merged_text)
                good_splits = []
            if not new_separators:
                final_chunks.append(s)
            else:
                final_chunks.extend(
                    _recursive_split_text(
                        s,
                        new_separators,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        is_separator_regex=is_separator_regex,
                    )
                )
    if good_splits:
        merged_text = _merge_splits(
            good_splits,
            merge_sep,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        final_chunks.extend(merged_text)
    return [c for c in final_chunks if c and c.strip()]


def document_word_chunks(document: str) -> List[str]:
    """Сырой текст чанков одного документа (рекурсивный сплиттер, без id)."""
    if not document or not document.strip():
        return []
    return _recursive_split_text(
        document.strip(),
        RECURSIVE_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def document_chunks_with_ids(documents: List[str]) -> List[List[ChunkRecord]]:
    """
    Для каждого документа — список чанков {"id", "text"}.
    id: целое, 0..N-1 в пределах одной строки (вопроса), без коллизий между документами.
    """
    out: List[List[ChunkRecord]] = []
    gid = 0
    for doc in documents:
        texts = document_word_chunks(doc)
        doc_list: List[ChunkRecord] = []
        for t in texts:
            doc_list.append({"id": gid, "text": t})
            gid += 1
        out.append(doc_list)
    return out


def count_words(text: str) -> int:
    """Подсчёт слов в тексте."""
    return len(text.split()) if text else 0


def tokenize_for_bm25(text: str) -> List[str]:
    """Токены для BM25 (слова в нижнем регистре)."""
    return re.findall(r'\b\w+\b', text.lower()) if text else []


def bm25_ordered_chunk_ids(
    question: str,
    answer: str,
    chunk_records: List[ChunkRecord],
    top_k: int,
) -> List[int]:
    """Идентификаторы чанков по убыванию BM25; запрос = вопрос + ответ."""
    query = f'{question} {answer}'.strip() if answer else question
    texts = [str(rec.get('text', '')) for rec in chunk_records]
    if not query or not texts or top_k <= 0:
        return [rec['id'] for rec in chunk_records[:top_k]]
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [tokenize_for_bm25(t) for t in texts]
        tokenized_query = tokenize_for_bm25(query)
        if not tokenized_query:
            return [rec['id'] for rec in chunk_records[:top_k]]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
        idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [chunk_records[i]['id'] for i in idx_sorted[:top_k]]
    except Exception:
        return [rec['id'] for rec in chunk_records[:top_k]]


def build_candidate_chunks(
    question: str,
    answer: str,
    chunk_records: List[ChunkRecord],
    target_count: int,
) -> List[ChunkRecord]:
    """
    Пул для LLM: сначала чанки с подстрокой ответа (порядок по id), затем BM25 до target_count.
    """
    if not chunk_records or target_count <= 0:
        return []
    by_id = {int(rec['id']): rec for rec in chunk_records}
    answer_l = answer.lower().strip() if answer else ''
    hits: List[ChunkRecord] = []
    if answer_l:
        for rec in sorted(chunk_records, key=lambda r: int(r['id'])):
            if answer_l in str(rec.get('text', '')).lower():
                hits.append(rec)
            if len(hits) >= target_count:
                break
    seen = {int(c['id']) for c in hits}
    ordered: List[ChunkRecord] = list(hits)
    ranked_ids = bm25_ordered_chunk_ids(
        question, answer, chunk_records, top_k=max(BM25_TOP_K, target_count)
    )
    for cid in ranked_ids:
        if len(ordered) >= target_count:
            break
        if cid not in seen and cid in by_id:
            seen.add(cid)
            ordered.append(by_id[cid])
    if len(ordered) < target_count:
        for rec in sorted(chunk_records, key=lambda r: int(r['id'])):
            cid = int(rec['id'])
            if cid not in seen:
                seen.add(cid)
                ordered.append(rec)
                if len(ordered) >= target_count:
                    break
    return ordered[:target_count]


def openrouter_pick_best_chunk(
    api_key: str,
    question: str,
    answer: str,
    chunk_records: List[ChunkRecord],
    model: str = OPENROUTER_MODEL,
) -> int:
    """
    Просит LLM (OpenRouter) выбрать номер чанка (1-based), который содержит ответ.
    chunk_records — срез списка с полем "text".
    Возвращает индекс (0-based) в chunk_records или -1.
    """
    if not chunk_records:
        return -1
    import requests

    batch = chunk_records[:MAX_CHUNKS_FOR_LLM]
    formatted = "\n\n".join(
        f"[{i+1}]\n{rec['text'][:1500]}" for i, rec in enumerate(batch)
    )
    prompt = f"""Question: {question}
Expected short answer: {answer}

Text fragments:
{formatted}

Which fragment [1], [2], ... contains the answer? The fragment must be actual content (not metadata, navigation, title page, or table of contents). Reply with only the number (1, 2, 3...) or 0 if none contain the answer."""

    import time
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/simple_qa",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
    }
    resp = None
    req_timeout = _openrouter_request_timeout()
    for attempt in range(OPENROUTER_HTTP_RETRIES):
        try:
            resp = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=req_timeout,
            )
            resp.raise_for_status()
            break
        except requests.exceptions.Timeout as e:
            if attempt < OPENROUTER_HTTP_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(
                    f"\nOpenRouter: таймаут после {OPENROUTER_HTTP_RETRIES} попыток "
                    f"(timeout={req_timeout}s): {e}",
                    file=sys.stderr,
                )
                return -1
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            if attempt < OPENROUTER_HTTP_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            r = getattr(e, 'response', None)
            status = r.status_code if r is not None else 0
            if status in (429, 502, 503, 504) and attempt < OPENROUTER_HTTP_RETRIES - 1:
                time.sleep(2 ** attempt)
            elif status in (429, 502, 503, 504):
                raise
            else:
                err_body = (r.text[:500] if r is not None else str(e))
                print(f"\nOpenRouter HTTP {status}: {err_body}", file=sys.stderr)
                return -1
    data = resp.json()
    content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

    numbers = re.findall(r'\b(\d+)\b', content)
    if numbers:
        n = int(numbers[0])
        if 1 <= n <= len(batch):
            return n - 1
        if n == 0:
            return -1
    return -1


def llm_select_chunks_for_document(
    api_key: str,
    question: str,
    answer: str,
    all_chunk_records: List[ChunkRecord],
    model: str = OPENROUTER_MODEL,
    *,
    relaxed: bool = False,
) -> List[ChunkRecord]:
    """
    BM25 + ответ в тексте → короткий пул; 1 вызов LLM (2-е окно при relaxed, если первое дало 0).
    При отказе LLM — лучший кандидат по пулу (не пустой long_answer без LLM-дубля).
    """
    if not all_chunk_records:
        return []
    pool_limit = MAX_CANDIDATES_RELAXED if relaxed else MAX_CANDIDATES_NORMAL
    pool = build_candidate_chunks(question, answer, all_chunk_records, pool_limit)
    if not pool:
        return []

    def _one_llm(batch: List[ChunkRecord]) -> List[ChunkRecord]:
        if not batch:
            return []
        idx = openrouter_pick_best_chunk(api_key, question, answer, batch, model=model)
        if 0 <= idx < len(batch):
            return [batch[idx]]
        return []

    first = pool[:MAX_CHUNKS_FOR_LLM]
    picked = _one_llm(first)
    if not picked and relaxed and len(pool) > MAX_CHUNKS_FOR_LLM:
        picked = _one_llm(pool[MAX_CHUNKS_FOR_LLM:MAX_CANDIDATES_RELAXED])
    if not picked and pool:
        return [pool[0]]
    return picked


def _long_answer_nonempty(raw: str) -> bool:
    try:
        v = json.loads(raw or '[]')
    except json.JSONDecodeError:
        return False
    if not isinstance(v, list):
        return False
    return any(str(x).strip() for x in v)


def process_row(
    row: dict,
    api_key: Optional[str],
    use_llm: bool = True,
    model: str = OPENROUTER_MODEL,
    *,
    relaxed: bool = False,
) -> Tuple[List[str], List[List[ChunkRecord]]]:
    """
    Для каждого документа строит чанки (с id), отбор BM25+ответ, затем LLM по короткому пулу.
    Возвращает (фрагменты для long_answer, список списков {id, text} по документам).
    """
    documents = parse_documents(row.get('documents', ''))
    question = str(row.get('problem', '')).strip()
    answer = str(row.get('answer', '')).strip()

    if not documents or not question:
        return [], []

    chunks_by_doc = document_chunks_with_ids(documents)
    fragments: List[str] = []

    for doc_chunks in chunks_by_doc:
        if not doc_chunks:
            continue
        if use_llm and api_key:
            selected = llm_select_chunks_for_document(
                api_key, question, answer, doc_chunks, model=model, relaxed=relaxed
            )
        else:
            cand = build_candidate_chunks(
                question, answer, doc_chunks, MAX_CANDIDATES_NORMAL
            )
            if cand:
                selected = [cand[0]]
            else:
                selected = [doc_chunks[0]]
        for rec in selected:
            t = str(rec.get("text", "")).strip()
            if t:
                fragments.append(t)

    return deduplicate_fragments(fragments), chunks_by_doc


def main():
    parser = argparse.ArgumentParser(description='Извлечение long_answer из документов')
    parser.add_argument(
        '--input', '-i',
        default=str(_DEFAULT_VERIFIED_DOCS),
        help='Входной CSV',
    )
    parser.add_argument(
        '--output', '-o',
        default=str(_DEFAULT_VERIFIED_LONG),
        help='Выходной CSV',
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Без LLM (лучший чанк по BM25+ответу или первый чанк документа)'
    )
    parser.add_argument(
        '--model', '-m',
        default=OPENROUTER_MODEL,
        help=f'Модель OpenRouter (по умолчанию: {OPENROUTER_MODEL})'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=0,
        help='Ограничить количество обрабатываемых строк (0 = все)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Продолжить с последней сохранённой позиции (читает output, пропускает обработанные)'
    )
    parser.add_argument(
        '--relaxed',
        action='store_true',
        help='Пул 16 кандидатов и второй вызов LLM при пустом ответе первого (слишком пустой long_answer реже)'
    )
    parser.add_argument(
        '--only-empty-long-answer',
        action='store_true',
        help='Только строки с пустым long_answer в существующем -o; включает режим relaxed для них'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=50,
        help='Сохранять прогресс каждые N строк (по умолчанию 50)'
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / '.env')
    except ImportError:
        pass

    api_key = None
    if not args.no_llm:
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            print(
                'Ошибка: для LLM нужен OPENROUTER_API_KEY. '
                'Укажите в окружении, в .env в корне репозитория (OPENROUTER_API_KEY=...), '
                'или используйте --no-llm. Установите пакет: pip install python-dotenv'
            )
            sys.exit(1)
        print(f'Используется OpenRouter: {args.model}')

    print(f'Чтение {args.input}...')
    rows = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for col in ('long_answer', 'chunks'):
            if col not in fieldnames:
                fieldnames.append(col)
        for row in reader:
            row.setdefault('long_answer', '[]')
            row.setdefault('chunks', '[]')
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break

    prev_rows: Optional[List[dict]] = None
    skip_count = 0
    if args.only_empty_long_answer:
        if not os.path.isfile(args.output):
            print('Ошибка: --only-empty-long-answer требует существующий файл -o.')
            sys.exit(1)
        with open(args.output, 'r', encoding='utf-8') as f:
            prev_rows = list(csv.DictReader(f))
        if args.resume:
            print('Примечание: при --only-empty-long-answer флаг --resume не используется.')
        if len(prev_rows) < len(rows):
            print(
                f'Предупреждение: в {args.output} строк {len(prev_rows)}, во входе {len(rows)} — '
                'для индексов ≥ len(prev) long_answer будет пересчитан без копирования.'
            )
    elif args.resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            resume_rows = list(csv.DictReader(f))
        skip_count = len(resume_rows)
        if skip_count > 0:
            for i, r in enumerate(resume_rows):
                if i < len(rows):
                    rows[i]['long_answer'] = r.get('long_answer', '[]')
                    rows[i]['chunks'] = r.get('chunks', '[]')
            print(f'Resume: пропуск {skip_count} уже обработанных строк')
            if skip_count >= len(rows):
                print('Все строки уже обработаны. Для перезапуска удалите выходной файл или запустите без --resume.')
                with open(args.output, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                    writer.writeheader()
                    writer.writerows(rows)
                filled = sum(1 for r in rows if _long_answer_nonempty(r.get('long_answer', '[]')))
                print(f'Готово. Заполнено long_answer: {filled}/{len(rows)}')
                sys.exit(0)

    relaxed = bool(args.relaxed or args.only_empty_long_answer)

    total = len(rows)
    print(f'Обработка {total} строк... (relaxed={relaxed})')

    from tqdm import tqdm
    save_every = max(1, args.save_every)
    for i, row in enumerate(tqdm(rows, desc='Извлечение long_answer')):
        if args.only_empty_long_answer and prev_rows is not None and i < len(prev_rows):
            if _long_answer_nonempty(prev_rows[i].get('long_answer', '[]')):
                rows[i]['long_answer'] = prev_rows[i].get('long_answer', '[]')
                rows[i]['chunks'] = prev_rows[i].get('chunks', '[]')
                continue
        if i < skip_count:
            continue
        fragments, chunks_by_doc = process_row(
            row, api_key, use_llm=not args.no_llm, model=args.model, relaxed=relaxed
        )
        row['long_answer'] = json.dumps(fragments, ensure_ascii=False)
        row['chunks'] = json.dumps(chunks_by_doc, ensure_ascii=False)
        if (i + 1) % save_every == 0:
            with open(args.output, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                writer.writerows(rows[: i + 1])
            tqdm.write(f'Сохранено {i + 1}/{total}')

    print(f'Сохранение в {args.output}...')
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    filled = sum(1 for r in rows if _long_answer_nonempty(r.get('long_answer', '[]')))
    print(f'Готово. С long_answer (есть хотя бы один непустой фрагмент): {filled}/{total}')
    unfilled = [r for r in rows if not _long_answer_nonempty(r.get('long_answer', '[]'))]
    if unfilled:
        no_doc = sum(1 for r in unfilled if not parse_documents(r.get('documents', '')))
        with_docs = len(unfilled) - no_doc
        print(
            f'  Пустой long_answer: {len(unfilled)} строк '
            f'(без documents: {no_doc}; с documents, но не получилось извлечь: {with_docs}).'
        )
        if with_docs:
            print('  Для второй попытки по «дырам» с документами: --only-empty-long-answer --relaxed')


if __name__ == '__main__':
    main()
