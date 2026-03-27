#!/usr/bin/env python3
"""
Извлечение релевантных фрагментов (long_answer) из документов для каждого вопроса.
- Самостоятельно определяет наличие ответа в документе (поиск подстроки)
- Для документа с ответом: окно вокруг позиции ответа (вариант 1)
- Для остальных документов: разбивка на чанки + BM25 + выбор через LLM (вариант 2)
- long_answer — список уникальных фрагментов (без дубликатов после нормализации)

LLM: OpenRouter API (openai/gpt-4o по умолчанию). Требуется OPENROUTER_API_KEY.
"""

import argparse
import ast
import csv
import json
import os
import re
import sys
from typing import List, Optional, Tuple

# Увеличиваем лимит размера поля CSV
csv.field_size_limit(10**7)

# Константы
CONTEXT_CHARS = 400  # символов до и после ответа для окна
CHUNK_WORDS = 400    # слов в чанке при разбивке
CHUNK_OVERLAP_WORDS = 80
MAX_CHUNKS_FOR_LLM = 8  # макс. чанков для выбора (ограничение контекста)
BM25_TOP_K = 5  # BM25 предфильтр: сколько сегментов подавать в LLM
MAX_PARAGRAPH_WORDS = 600  # абзац больше — считаем "слишком большим", используем чанки
MIN_PARAGRAPHS = 2  # минимум абзацев, чтобы использовать разбивку по абзацам
ADJACENT_PARAGRAPHS = 1  # соседних абзацев с каждой стороны для контекста (1 = до 3 абзацев)
MAX_EXPANDED_WORDS = 500  # макс. слов при расширении контекстом

# OpenRouter: сильная модель OpenAI
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-4o"


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
            import json
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


# Паттерны для определения мусорных фрагментов (навигация, меню, заголовки)
_JUNK_WORDS = re.compile(
    r'\b(sign\s*up|login|sign\s*in|logout|profile|privacy\s*policy|terms\s*of\s*use|'
    r'contact\s*us|navigation|toggle|menu|cookie|newsletter|subscribe)\b',
    re.I
)


def is_likely_junk(text: str, answer: str = '') -> bool:
    """
    Определяет, похож ли фрагмент на навигацию/меню/заголовок.
    Такие фрагменты отфильтровываются из результата.
    Если в фрагменте есть ответ — не считаем мусором (контент важнее навигации).
    """
    if not text or not text.strip():
        return True
    text_lower = text.lower()
    words = text.split()
    # Фрагмент содержит ответ — сохраняем, даже если есть навигация
    if answer and answer.strip() and answer.lower() in text_lower:
        return False
    # Очень короткий фрагмент без точки — скорее заголовок или пункт меню
    if len(words) < 12 and '.' not in text:
        return True
    # Много типичных слов навигации
    if len(_JUNK_WORDS.findall(text_lower)) >= 3:
        return True
    # Много разделителей | или · (пункты меню, навигация)
    if text.count('|') + text.count('·') >= 8:
        return True
    # Начинается с типичной навигации
    nav_start = text_lower[:150]
    if any(nav_start.startswith(p) for p in ('skip to', 'toggle navigation', 'sign up')):
        return True
    return False


def extract_context_window(
    document: str,
    answer: str,
    pos_chars: int,
    context_chars: int = CONTEXT_CHARS
) -> str:
    """
    Извлекает окно контекста вокруг позиции ответа.
    Пытается обрезать по границам предложений.
    """
    if not document or pos_chars is None or pos_chars == '':
        return ''
    try:
        pos = int(pos_chars)
    except (ValueError, TypeError):
        return ''
    start = max(0, pos - context_chars)
    end = min(len(document), pos + len(str(answer)) + context_chars)
    chunk = document[start:end]
    # Расширяем до границ предложений
    if start > 0:
        before = document[max(0, start - 200):start]
        for sep in '.!?。！？\n':
            last = before.rfind(sep)
            if last >= 0:
                start = max(0, start - 200 + last + 1)
                break
    if end < len(document):
        after = document[end:min(len(document), end + 200)]
        for sep in '.!?。！？\n':
            first = after.find(sep)
            if first >= 0:
                end = end + first + 1
                break
    return document[start:end].strip()


def count_words(text: str) -> int:
    """Подсчёт слов в тексте."""
    return len(text.split()) if text else 0


def split_into_paragraphs(text: str) -> List[str]:
    """Разбивает текст на абзацы по двойному переносу строки."""
    if not text or not text.strip():
        return []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs


def split_into_chunks(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    """Разбивает текст на перекрывающиеся чанки по словам."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_words])
        if chunk.strip():
            chunks.append(chunk.strip())
        if i + chunk_words >= len(words):
            break
    return chunks


def tokenize_for_bm25(text: str) -> List[str]:
    """Простая токенизация для BM25 (слова, нижний регистр)."""
    return re.findall(r'\b\w+\b', text.lower()) if text else []


def bm25_filter_segments(
    question: str,
    segments: List[str],
    top_k: int = BM25_TOP_K,
    answer: str = '',
) -> List[str]:
    """
    BM25 предфильтр: возвращает top_k наиболее релевантных сегментов.
    Запрос: вопрос + ответ (если задан).
    """
    query = f'{question} {answer}'.strip() if answer else question
    if not segments or not query:
        return segments
    if len(segments) <= top_k:
        return segments
    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [tokenize_for_bm25(s) for s in segments]
        tokenized_query = tokenize_for_bm25(query)
        if not tokenized_query:
            return segments[:top_k]
        bm25 = BM25Okapi(tokenized_corpus)
        top_indices = sorted(
            range(len(segments)),
            key=lambda i: bm25.get_scores(tokenized_query)[i],
            reverse=True
        )[:top_k]
        return [segments[i] for i in sorted(top_indices)]  # сохраняем порядок по релевантности
    except ImportError:
        return segments[:top_k]
    except Exception:
        return segments[:top_k]


def expand_segment_with_context(
    best_segment: str,
    doc_segments: List[str],
    max_words: int = MAX_EXPANDED_WORDS,
    adjacent: int = ADJACENT_PARAGRAPHS,
) -> str:
    """
    Расширяет выбранный сегмент соседними абзацами для контекста.
    Возвращает объединённый текст (best + prev + next), ограниченный max_words.
    """
    if not doc_segments or best_segment not in doc_segments:
        return best_segment
    idx = doc_segments.index(best_segment)
    start = max(0, idx - adjacent)
    end = min(len(doc_segments), idx + adjacent + 1)
    expanded = '\n\n'.join(doc_segments[start:end])
    if count_words(expanded) <= max_words:
        return expanded
    # Обрезаем с конца, оставляя max_words
    words = expanded.split()
    return ' '.join(words[:max_words])


def get_segments(text: str) -> List[str]:
    """
    Возвращает сегменты для выбора через LLM.
    Сначала пробует абзацы; если их нет или они слишком большие — чанки.
    """
    paragraphs = split_into_paragraphs(text)
    if len(paragraphs) >= MIN_PARAGRAPHS:
        # Проверяем, что абзацы не слишком большие
        if all(count_words(p) <= MAX_PARAGRAPH_WORDS for p in paragraphs):
            return paragraphs
    # Fallback: чанки по словам
    return split_into_chunks(text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)


def get_paragraph_containing_position(text: str, pos_chars: int) -> Optional[str]:
    """
    Возвращает абзац, содержащий позицию pos_chars.
    Если абзацев нет (один сплошной текст) — возвращает None.
    """
    paragraphs = split_into_paragraphs(text)
    if len(paragraphs) < MIN_PARAGRAPHS:
        return None
    cumul = 0
    for p in paragraphs:
        cumul += len(p) + 2  # +2 за \n\n
        if cumul > pos_chars:
            return p
    return paragraphs[-1] if paragraphs else None


def openrouter_pick_best_chunk(
    api_key: str,
    question: str,
    answer: str,
    chunks: List[str],
    model: str = OPENROUTER_MODEL,
) -> int:
    """
    Просит LLM (OpenRouter) выбрать номер чанка (1-based), который содержит ответ.
    Возвращает индекс (0-based) или -1 если ни один не подходит.
    """
    if not chunks:
        return -1
    import requests

    formatted = "\n\n".join(f"[{i+1}]\n{c[:1500]}" for i, c in enumerate(chunks[:MAX_CHUNKS_FOR_LLM]))
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
    for attempt in range(4):
        try:
            resp = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            break
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            if attempt < 3:
                time.sleep(2 ** attempt)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            r = getattr(e, 'response', None)
            status = r.status_code if r is not None else 0
            if status in (429, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt)
            elif status in (429, 502, 503, 504):
                raise
            else:
                # 400 и др. — не ретраим, возвращаем fallback (ни один фрагмент)
                err_body = (r.text[:500] if r is not None else str(e))
                print(f"\nOpenRouter HTTP {status}: {err_body}", file=sys.stderr)
                return -1
    data = resp.json()
    content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

    numbers = re.findall(r'\b(\d+)\b', content)
    if numbers:
        n = int(numbers[0])
        if 1 <= n <= len(chunks):
            return n - 1
        if n == 0:
            return -1
    return 0  # fallback: первый чанк


def _get_fragment_variant1(
    doc: str,
    answer: str,
    pos_chars: int,
) -> str:
    """Вариант 1: извлечение по позиции ответа (документ с совпадением)."""
    para = get_paragraph_containing_position(doc, pos_chars)
    if para and count_words(para) <= MAX_PARAGRAPH_WORDS:
        return para
    return extract_context_window(doc, answer, pos_chars)


def _get_fragment_variant2(
    doc: str,
    question: str,
    answer: str,
    api_key: Optional[str],
    use_llm: bool,
    model: str = OPENROUTER_MODEL,
) -> str:
    """Вариант 2: BM25 + LLM (OpenRouter) для документа без явного совпадения."""
    doc_segments = get_segments(doc)
    if not doc_segments:
        return ''
    top_segments = bm25_filter_segments(question, doc_segments, BM25_TOP_K, answer)
    if not top_segments:
        return ''
    # Отфильтровываем мусор (навигация, меню)
    # Приоритет: 1) не-мусор, 2) с ответом (если всё мусор — берём то, где есть ответ)
    candidates = [s for s in top_segments if not is_likely_junk(s, '')]
    if not candidates and answer:
        candidates = [s for s in top_segments if answer.lower() in s.lower()]
    if not candidates:
        extended = bm25_filter_segments(question, doc_segments, top_k=min(15, len(doc_segments)), answer=answer)
        candidates = [s for s in extended if not is_likely_junk(s, '')]
        if not candidates and answer:
            candidates = [s for s in extended if answer.lower() in s.lower()]
    segments_for_pick = candidates if candidates else top_segments
    if use_llm and api_key:
        best_idx = openrouter_pick_best_chunk(
            api_key, question, answer, segments_for_pick, model=model
        )
        best = segments_for_pick[best_idx] if best_idx >= 0 else ''
    else:
        best = segments_for_pick[0] if segments_for_pick else ''
    if best:
        return expand_segment_with_context(best, doc_segments)
    return ''


def process_row(
    row: dict,
    api_key: Optional[str],
    use_llm: bool = True,
    model: str = OPENROUTER_MODEL,
) -> List[str]:
    """
    Обрабатывает одну строку. Для каждого документа находит релевантный фрагмент.
    Возвращает список уникальных фрагментов (без дубликатов после нормализации).
    """
    documents = parse_documents(row.get('documents', ''))
    question = str(row.get('problem', '')).strip()
    answer = str(row.get('answer', '')).strip()

    if not documents or not question:
        return []

    fragments = []
    for i, doc in enumerate(documents):
        candidate = ''
        # Самостоятельно определяем: есть ли ответ в документе
        if answer and answer.lower() in doc.lower():
            pos_int = doc.lower().find(answer.lower())
            candidate = _get_fragment_variant1(doc, answer, pos_int)
        else:
            candidate = _get_fragment_variant2(
                doc, question, answer, api_key, use_llm, model=model
            )

        if candidate and candidate.strip() and not is_likely_junk(candidate, answer):
            fragments.append(candidate.strip())

    return deduplicate_fragments(fragments)


def main():
    parser = argparse.ArgumentParser(description='Извлечение long_answer из документов')
    parser.add_argument(
        '--input', '-i',
        default='/home/dolganov/simple_qa/simple_qa_test_set_with_documents.csv',
        help='Входной CSV'
    )
    parser.add_argument(
        '--output', '-o',
        default='/home/dolganov/simple_qa/simple_qa_test_set_with_long_answer.csv',
        help='Выходной CSV'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Не использовать LLM (только извлечение по позиции/первый чанк)'
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
        '--save-every',
        type=int,
        default=50,
        help='Сохранять прогресс каждые N строк (по умолчанию 50)'
    )
    args = parser.parse_args()

    api_key = None
    if not args.no_llm:
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            print('Ошибка: для LLM нужен OPENROUTER_API_KEY. Установите переменную окружения или используйте --no-llm.')
            sys.exit(1)
        print(f'Используется OpenRouter: {args.model}')

    print(f'Чтение {args.input}...')
    rows = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        if 'long_answer' not in fieldnames:
            fieldnames.append('long_answer')
        for row in reader:
            row.setdefault('long_answer', '[]')
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break

    skip_count = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            resume_rows = list(csv.DictReader(f))
        skip_count = len(resume_rows)
        if skip_count > 0:
            for i, r in enumerate(resume_rows):
                if i < len(rows):
                    rows[i]['long_answer'] = r.get('long_answer', '[]')
            print(f'Resume: пропуск {skip_count} уже обработанных строк')
            if skip_count >= len(rows):
                print('Все строки уже обработаны. Для перезапуска удалите выходной файл или запустите без --resume.')
                with open(args.output, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                    writer.writeheader()
                    writer.writerows(rows)
                filled = sum(1 for r in rows if json.loads(r.get('long_answer', '[]')))
                print(f'Готово. Заполнено long_answer: {filled}/{len(rows)}')
                sys.exit(0)

    total = len(rows)
    print(f'Обработка {total} строк...')

    from tqdm import tqdm
    save_every = max(1, args.save_every)
    for i, row in enumerate(tqdm(rows, desc='Извлечение long_answer')):
        if i < skip_count:
            continue
        fragments = process_row(
            row, api_key, use_llm=not args.no_llm, model=args.model
        )
        row['long_answer'] = json.dumps(fragments, ensure_ascii=False)
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

    filled = sum(1 for r in rows if json.loads(r.get('long_answer', '[]')))
    print(f'Готово. Заполнено long_answer (непустой список): {filled}/{total}')


if __name__ == '__main__':
    main()
