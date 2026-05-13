#!/usr/bin/env python3
"""
Извлечение релевантных фрагментов (long_answer) из документов для каждого вопроса.
- Документ режется рекурсивным сплиттером (как LangChain RecursiveCharacterTextSplitter):
  сначала по абзацам/строкам, затем предложениям, словам, символам; затем очень короткие чанки
  приклеиваются к соседям (после завершённого предложения — к следующему, иначе — к предыдущему;
  склейка с соседом через двойной перевод строки; абзацы внутри кусков не трогаются). Допускаются чанки длиннее CHUNK_SIZE.
  В поле chunks — JSON: [ [ {"id": 0, "text": "..."}, ... ], [...] ] (id сквозной по документам строки).
- Отбор кандидатов: BM25 по запросу «вопрос + ответ» + чанки с вхождением ответа; LLM один раз
  (или два при --only-empty-long-answer / relaxed) по короткому пулу.
  С --relaxed-embeddings и relaxed: позиции пула 9–16 добираются по cosine similarity эмбеддингов (sentence-transformers).
  GPU для эмбеддингов: переменная SIMPLEQA_EMBEDDING_DEVICE (например cuda:1 или 1); иначе sbert берёт устройство по умолчанию (часто cuda:0).
  По умолчанию запрос с structured output
  (response_format json_schema: {fragment}), при отказе провайдера — тот же промпт без схемы и разбор числа из текста.
  Совпадение ответа с чанком: дословная подстрока; затем нормализованный текст (NFKC, €/euro, запятые в числах);
  совпадение дат (dateutil + набор строковых форматов); совпадение чисел (все числа из «ядра» ответа есть в чанке).
  При отказе LLM для документа, где ответ подтверждается этими правилами — fallback на первый такой чанк.
  Если answer в строке CSV пустой — проверка не применяется.
- long_answer — список уникальных фрагментов (без дубликатов после нормализации).

LLM: OpenRouter API (google/gemini-2.5-pro по умолчанию). Ключ: OPENROUTER_API_KEY в окружении
или в файле .env в корне репозитория (подгружается через python-dotenv).
  OPENROUTER_REQUEST_TIMEOUT — таймаут ответа в секундах (по умолчанию 180).
"""

import argparse
import ast
import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dateutil import parser as date_parser
from dateutil.parser import ParserError

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
# После сплиттера: чанк «слишком маленький», если символов меньше MIN_CHUNK_CHARS или слов меньше MIN_CHUNK_WORDS.
# Часть чанков после склейки может стать длиннее CHUNK_SIZE — это нормально.
MIN_CHUNK_CHARS = 120
MIN_CHUNK_WORDS = 10
_PREV_SENTENCE_COMPLETE = re.compile(r'[.!?…]["\'")\]]*\s*$', re.UNICODE)
MAX_CHUNKS_FOR_LLM = 8  # макс. чанков в одном запросе к LLM
BM25_TOP_K = 18  # сколько лучших по BM25 рассматривать при сборке пула (до среза)
MAX_CANDIDATES_NORMAL = 8  # размер пула → один вызов LLM
MAX_CANDIDATES_RELAXED = 16  # два окна по 8 при «втором проходе»
EMBED_MODEL_DEFAULT = "all-MiniLM-L6-v2"  # sentence-transformers; для --relaxed-embeddings
EMBED_CHUNK_CHAR_CAP = 8000  # обрезка текста чанка при энкодинге

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-2.5-pro"
OPENROUTER_HTTP_RETRIES = 5


_ST_MODEL: Optional[Any] = None
_ST_MODEL_KEY: Optional[Tuple[str, str]] = None  # (model_name, device_key для кэша)


def _embedding_torch_device() -> Optional[str]:
    """
    Устройство для SentenceTransformer. Env SIMPLEQA_EMBEDDING_DEVICE:
    cuda:1, cpu, cuda — или одна цифра «1» → cuda:1. Пусто — поведение sentence-transformers по умолчанию.
    """
    raw = (os.environ.get("SIMPLEQA_EMBEDDING_DEVICE") or "").strip()
    if not raw:
        return None
    low = raw.lower()
    if low in ("cpu", "cuda"):
        return low
    if re.fullmatch(r"\d+", raw):
        return f"cuda:{raw}"
    if low.startswith("cuda:") or low.startswith("mps"):
        return low
    return raw


def _get_sentence_transformer(model_name: str) -> Any:
    global _ST_MODEL, _ST_MODEL_KEY
    device = _embedding_torch_device()
    cache_dev = device if device is not None else ""
    key = (model_name, cache_dev)
    if _ST_MODEL is not None and _ST_MODEL_KEY == key:
        return _ST_MODEL
    from sentence_transformers import SentenceTransformer

    kwargs: Dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    _ST_MODEL = SentenceTransformer(model_name, **kwargs)
    _ST_MODEL_KEY = key
    return _ST_MODEL


def enrich_relaxed_pool_with_embeddings(
    question: str,
    answer: str,
    pool: List[ChunkRecord],
    all_chunk_records: List[ChunkRecord],
    target_len: int,
    embedding_model: str,
) -> List[ChunkRecord]:
    """
    При relaxed: первые MAX_CHUNKS_FOR_LLM слотов пула оставляем как из BM25/hits;
    оставшиеся до target_len заполняем чанками с наибольшим cosine similarity эмбеддинга
    к запросу «вопрос + ответ». При нехватке — добираем из хвоста исходного пула.
    """
    if not pool or not all_chunk_records or target_len <= 0:
        return pool
    try:
        import numpy as np
    except ImportError:
        print(
            "extract_long_answer: для --relaxed-embeddings нужен numpy (идёт с sentence-transformers).",
            file=sys.stderr,
        )
        return pool
    try:
        model = _get_sentence_transformer(embedding_model)
    except ImportError as e:
        print(
            f"extract_long_answer: {e} Установите: pip install sentence-transformers",
            file=sys.stderr,
        )
        return pool
    head_n = min(MAX_CHUNKS_FOR_LLM, len(pool))
    head = pool[:head_n]
    head_ids = {int(r["id"]) for r in head}

    query = f"{question}\n{answer}".strip() if (answer or "").strip() else question
    texts = [str(c.get("text", ""))[:EMBED_CHUNK_CHAR_CAP] for c in all_chunk_records]
    qv = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )[0]
    evs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    scores = evs @ qv
    order = np.argsort(-scores)

    out: List[ChunkRecord] = list(head)
    seen = set(head_ids)
    for ix in order:
        if len(out) >= target_len:
            break
        rec = all_chunk_records[int(ix)]
        cid = int(rec["id"])
        if cid in seen:
            continue
        seen.add(cid)
        out.append(rec)
    if len(out) < target_len:
        for rec in pool[head_n:]:
            if len(out) >= target_len:
                break
            cid = int(rec["id"])
            if cid in seen:
                continue
            seen.add(cid)
            out.append(rec)
    return out[:target_len]


def _openrouter_request_timeout() -> float:
    """Секунды на HTTP-запрос; можно OPENROUTER_REQUEST_TIMEOUT в env или .env (после load_dotenv)."""
    return float(os.environ.get("OPENROUTER_REQUEST_TIMEOUT", "180"))


def _documents_list_slots(parsed: List[Any]) -> List[str]:
    """Один элемент списка documents → строка текста или \"\" (место сохраняется под индексом URL)."""
    out: List[str] = []
    for doc in parsed:
        if doc is None:
            out.append("")
            continue
        s = str(doc)
        out.append(s if s.strip() else "")
    return out


def parse_documents(documents_str: str) -> List[str]:
    """
    Парсит строку с документами в список.
    Длина списка совпадает с числом слотов в исходном массиве (пустой слот → \"\"),
    чтобы колонка chunks в CSV выравнивалась по индексу с urls.
    """
    if not documents_str or documents_str == '[]' or documents_str.strip() in ("''", '""'):
        return []
    try:
        parsed = ast.literal_eval(documents_str)
        if isinstance(parsed, list):
            return _documents_list_slots(parsed)
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
        return []
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(documents_str)
            if isinstance(parsed, list):
                return _documents_list_slots(parsed)
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


def _prev_sentence_complete(text: str) -> bool:
    """Предыдущий кусок заканчивается завершённым предложением — короткий фрагмент чаще заголовок к следующему."""
    t = text.rstrip()
    if not t:
        return False
    return bool(_PREV_SENTENCE_COMPLETE.search(t))


def _glue_adjacent_chunks(a: str, b: str) -> str:
    """Склейка двух чанков с пустой строкой между блоками (сохраняет разделение абзацев)."""
    a = a.rstrip()
    b = b.lstrip()
    if not a:
        return b
    if not b:
        return a
    return f"{a}\n\n{b}"


def _absorb_short_document_chunks(
    chunks: List[str],
    *,
    min_chars: int = MIN_CHUNK_CHARS,
    min_words: int = MIN_CHUNK_WORDS,
) -> List[str]:
    """
    Убирает слишком короткие чанки: к предыдущему (обрывок продолжения) или к следующему (заголовок).
    Первый чанк документа клеится только вперёд, последний — только назад.
    """
    chunks = [c for c in chunks if c and c.strip()]
    if len(chunks) <= 1:
        return chunks

    def undersized(t: str) -> bool:
        s = t.strip()
        return len(s) < min_chars or len(s.split()) < min_words

    max_iter = len(chunks) * 6 + 20
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        try:
            i = next(idx for idx, ch in enumerate(chunks) if undersized(ch))
        except StopIteration:
            break
        if len(chunks) == 1:
            break
        if i == 0:
            chunks[0] = _glue_adjacent_chunks(chunks[0], chunks[1])
            del chunks[1]
        elif i == len(chunks) - 1:
            chunks[-2] = _glue_adjacent_chunks(chunks[-2], chunks[-1])
            del chunks[-1]
        elif _prev_sentence_complete(chunks[i - 1]):
            chunks[i] = _glue_adjacent_chunks(chunks[i], chunks[i + 1])
            del chunks[i + 1]
        else:
            chunks[i - 1] = _glue_adjacent_chunks(chunks[i - 1], chunks[i])
            del chunks[i]
    return chunks


def document_word_chunks(document: str) -> List[str]:
    """Сырой текст чанков одного документа (рекурсивный сплиттер + поглощение слишком коротких, без id)."""
    if not document or not document.strip():
        return []
    raw = _recursive_split_text(
        document.strip(),
        RECURSIVE_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return _absorb_short_document_chunks(raw)


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
    Пул для LLM: сначала чанки, где ответ согласуется с текстом (подстрока / нормализация / дата / числа),
    """
    if not chunk_records or target_count <= 0:
        return []
    by_id = {int(rec['id']): rec for rec in chunk_records}
    hits: List[ChunkRecord] = []
    if (answer or "").strip():
        for rec in sorted(chunk_records, key=lambda r: int(r['id'])):
            if answer_supported_by_chunk(answer, str(rec.get('text', ''))):
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


def _answer_core_before_paren(answer: str) -> str:
    """Часть ответа до '(' — отрезает хвост ' (acceptable range: …)' и т.п."""
    s = (answer or "").strip()
    if not s:
        return s
    if "(" not in s:
        return s
    head = s[: s.index("(")].strip()
    return head if head else s


def _normalize_loose_text(s: str) -> str:
    """Нормализация для мягкого in: unicode, пробелы, €, запятые в числах, слово euro(s)."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    for ch in ("\u00a0", "\u2009", "\u202f"):
        s = s.replace(ch, " ")
    s = s.replace("€", " eur ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"\beuros?\b", "", s)
    s = re.sub(r"\beur\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_NUM_TOKEN_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _numeric_tokens_from_text(s: str) -> List[float]:
    out: List[float] = []
    for m in _NUM_TOKEN_RE.finditer(s):
        t = m.group(0).replace(",", "")
        try:
            out.append(float(t))
        except ValueError:
            continue
    return out


def _numbers_in_chunk_support_answer(answer_core: str, chunk: str) -> bool:
    """
    Все «скалярные» числа из ядра ответа должны найтись в чанке (с точностью float).
    Пропускаем очень длинные целые (>12 цифр), чтобы не ловить ID/хэши.
    """
    an = _numeric_tokens_from_text(answer_core)
    required = [x for x in an if abs(x) < 10**12]
    if not required:
        return False
    cn = _numeric_tokens_from_text(chunk)
    if not cn:
        return False
    cset = list(cn)
    for x in required:
        if not any(math.isclose(x, y, rel_tol=0, abs_tol=1e-5) for y in cset):
            return False
    return True


def _looks_datey(s: str) -> bool:
    sl = (s or "").lower()
    if re.search(r"\d{1,4}[-/.]\d{1,4}([-/.]\d{1,4})?", s):
        return True
    return any(
        m in sl
        for m in (
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        )
    )


def _date_in_chunk_matches_answer(answer: str, chunk: str) -> bool:
    """Парсинг даты из ответа и поиск типичных строковых представлений в чанке."""
    core = _answer_core_before_paren(answer).strip()
    if len(core) < 4:
        return False
    cl = (chunk or "").lower()
    try:
        dt = date_parser.parse(core, fuzzy=False)
    except (ParserError, ValueError, OverflowError, TypeError):
        if not _looks_datey(core):
            return False
        try:
            dt = date_parser.parse(core, fuzzy=True)
        except (ParserError, ValueError, OverflowError, TypeError):
            return False

    variants: Set[str] = set()
    variants.add(dt.strftime("%b %d, %Y").lower())
    variants.add(dt.strftime("%b %d %Y").lower())
    variants.add(dt.strftime("%B %d, %Y").lower())
    variants.add(dt.strftime("%d %B %Y").lower())
    variants.add(dt.strftime("%d %b %Y").lower())
    variants.add(dt.strftime("%B %Y").lower())
    variants.add(dt.strftime("%b %Y").lower())
    variants.add(dt.strftime("%Y-%m-%d").lower())
    variants.add(f"{dt.month}/{dt.day}/{dt.year}")
    variants.add(f"{dt.day}/{dt.month}/{dt.year}")

    if any(len(v) >= 6 and v in cl for v in variants):
        return True
    if core.isdigit() and len(core) == 4 and core in cl:
        return True
    return False


def answer_supported_by_chunk(answer: str, chunk_text: str) -> bool:
    """
    Пустой answer — не фильтруем. Иначе: подстрока; нормализованная подстрока;
    совпадение даты; все числа из ядра ответа присутствуют в чанке как числа.
    """
    a = (answer or "").strip()
    if not a:
        return True
    chunk = str(chunk_text or "")
    al = a.lower()
    c_low = chunk.lower()
    if al in c_low:
        return True

    norm_a = _normalize_loose_text(a)
    norm_c = _normalize_loose_text(chunk)
    if len(norm_a) >= 3 and norm_a in norm_c:
        return True

    if _date_in_chunk_matches_answer(a, chunk):
        return True

    core = _answer_core_before_paren(a)
    if core and _numbers_in_chunk_support_answer(core, chunk):
        return True

    return False


def _validated_llm_pick(picked: List[ChunkRecord], answer: str) -> List[ChunkRecord]:
    if not picked:
        return []
    if answer_supported_by_chunk(answer, str(picked[0].get("text", ""))):
        return picked
    return []


def _openrouter_chunk_pick_response_format(n_fragments: int) -> Dict[str, Any]:
    """
    Structured output для OpenRouter: см. https://openrouter.ai/docs/guides/features/structured-outputs
    fragment — 1-based индекс фрагмента или 0 если ни один не подходит.
    """
    nf = max(1, int(n_fragments))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "chunk_pick",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "fragment": {
                        "type": "integer",
                        "description": (
                            "1-based index of the text fragment that contains the expected answer; "
                            "0 if none of the listed fragments contain it."
                        ),
                        "minimum": 0,
                        "maximum": nf,
                    },
                },
                "required": ["fragment"],
                "additionalProperties": False,
            },
        },
    }


def _parse_llm_chunk_index(content: str, n_fragments: int) -> int:
    """
    Из message.content: JSON {"fragment": k} (structured) или первое целое в тексте (legacy).
    Возвращает 0-based индекс в batch или -1.
    """
    text = (content or "").strip()
    if not text:
        return -1
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "fragment" in obj:
                k = int(obj["fragment"])
                if k == 0:
                    return -1
                if 1 <= k <= n_fragments:
                    return k - 1
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        k = int(numbers[0])
        if 1 <= k <= n_fragments:
            return k - 1
        if k == 0:
            return -1
    return -1


def openrouter_pick_best_chunk(
    api_key: str,
    question: str,
    answer: str,
    chunk_records: List[ChunkRecord],
    model: str = OPENROUTER_MODEL,
    *,
    use_structured_output: bool = True,
) -> int:
    """
    Просит LLM (OpenRouter) выбрать номер чанка (1-based), который содержит ответ.
    С use_structured_output: в теле запроса передаётся response_format json_schema
    (объект с ключом fragment: int); иначе — тот же промпт, ответ разбирается как число в тексте.
    При HTTP 400 из-за неподдержки схемы выполняется повтор без response_format.
    chunk_records — срез списка с полем "text".
    Возвращает индекс (0-based) в chunk_records или -1.
    """
    if not chunk_records:
        return -1
    import requests

    batch = chunk_records[:MAX_CHUNKS_FOR_LLM]
    nbatch = len(batch)
    formatted = "\n\n".join(
        f"[{i+1}]\n{rec['text'][:1500]}" for i, rec in enumerate(batch)
    )
    prompt = f"""Question: {question}
Expected short answer: {answer}

Text fragments:
{formatted}

Which fragment [1], [2], ... contains the answer? The fragment must be actual content (not metadata, navigation, title page, or table of contents). Reply with only the number (1, 2, 3...) or 0 if none contain the answer."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/simple_qa",
    }
    payload_base: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0,
    }
    req_timeout = _openrouter_request_timeout()
    data: Optional[dict] = None
    for attempt in range(OPENROUTER_HTTP_RETRIES):
        try:
            try_structured = bool(use_structured_output)
            while True:
                payload = dict(payload_base)
                if try_structured:
                    payload["response_format"] = _openrouter_chunk_pick_response_format(nbatch)
                resp = requests.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=req_timeout,
                )
                if resp.status_code == 400 and try_structured and use_structured_output:
                    err_low = (resp.text or "").lower()
                    if any(
                        s in err_low
                        for s in (
                            "json_schema",
                            "structured",
                            "response_format",
                            "invalid schema",
                            "unsupported",
                        )
                    ):
                        try_structured = False
                        continue
                resp.raise_for_status()
                data = resp.json()
                break
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
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ContentDecodingError,
        ) as e:
            if attempt < OPENROUTER_HTTP_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            print(
                f"\nOpenRouter: обрыв соединения / chunked после {OPENROUTER_HTTP_RETRIES} попыток: {e}",
                file=sys.stderr,
            )
            return -1
        except requests.exceptions.HTTPError as e:
            r = getattr(e, 'response', None)
            status = r.status_code if r is not None else 0
            if status in (429, 502, 503, 504) and attempt < OPENROUTER_HTTP_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            if status in (429, 502, 503, 504):
                raise
            err_body = (r.text[:500] if r is not None else str(e))
            print(f"\nOpenRouter HTTP {status}: {err_body}", file=sys.stderr)
            return -1
        except ValueError as e:
            # resp.json() — битый JSON в теле
            if attempt < OPENROUTER_HTTP_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"\nOpenRouter: не удалось разобрать JSON ответа: {e}", file=sys.stderr)
            return -1

    if not isinstance(data, dict):
        return -1
    content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

    return _parse_llm_chunk_index(content, nbatch)


def llm_select_chunks_for_document(
    api_key: str,
    question: str,
    answer: str,
    all_chunk_records: List[ChunkRecord],
    model: str = OPENROUTER_MODEL,
    *,
    relaxed: bool = False,
    use_structured_output: bool = True,
    relaxed_embeddings: bool = False,
    embedding_model: str = EMBED_MODEL_DEFAULT,
) -> List[ChunkRecord]:
    """
    BM25 + ответ в тексте → короткий пул; 1 вызов LLM (2-е окно при relaxed, если первое дало 0).
    С relaxed_embeddings и relaxed: слоты пула после первых MAX_CHUNKS_FOR_LLM добираются по эмбеддингам.
    Если ответ уже поддерживается каким-то чанком (answer_supported_by_chunk: подстрока, нормализация,
    даты, числа) — сразу берём первый такой чанк по id, без LLM и без эмбеддингов.
    """
    if not all_chunk_records:
        return []
    if (answer or "").strip():
        for rec in sorted(all_chunk_records, key=lambda r: int(r["id"])):
            if answer_supported_by_chunk(answer, str(rec.get("text", ""))):
                return [rec]
    pool_limit = MAX_CANDIDATES_RELAXED if relaxed else MAX_CANDIDATES_NORMAL
    pool = build_candidate_chunks(question, answer, all_chunk_records, pool_limit)
    if not pool:
        return []
    if relaxed and relaxed_embeddings:
        pool = enrich_relaxed_pool_with_embeddings(
            question,
            answer,
            pool,
            all_chunk_records,
            pool_limit,
            embedding_model,
        )

    def _one_llm(batch: List[ChunkRecord]) -> List[ChunkRecord]:
        if not batch:
            return []
        idx = openrouter_pick_best_chunk(
            api_key,
            question,
            answer,
            batch,
            model=model,
            use_structured_output=use_structured_output,
        )
        if 0 <= idx < len(batch):
            return [batch[idx]]
        return []

    first = pool[:MAX_CHUNKS_FOR_LLM]
    picked = _validated_llm_pick(_one_llm(first), answer)
    if not picked and relaxed and len(pool) > MAX_CHUNKS_FOR_LLM:
        picked = _validated_llm_pick(
            _one_llm(pool[MAX_CHUNKS_FOR_LLM:MAX_CANDIDATES_RELAXED]), answer
        )
    if picked:
        return picked
    return []


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
    use_structured_output: bool = True,
    relaxed_embeddings: bool = False,
    embedding_model: str = EMBED_MODEL_DEFAULT,
) -> Tuple[List[str], List[List[ChunkRecord]]]:
    """
    Для каждого документа строит чанки (с id), отбор BM25+ответ и LLM по короткому пулу
    (или при --no-llm: первый чанк с подстрокой answer; при пустом answer — первый кандидат BM25).
    Возвращает (фрагменты для long_answer, список списков {id, text} по документам).
    """
    documents = parse_documents(row.get('documents', ''))
    question = str(row.get('problem', '')).strip()
    answer = str(row.get('answer', '')).strip()

    if not question:
        return [], []
    if not documents:
        return [], []

    chunks_by_doc = document_chunks_with_ids(documents)
    fragments: List[str] = []

    for doc_chunks in chunks_by_doc:
        if not doc_chunks:
            continue
        if use_llm and api_key:
            selected = llm_select_chunks_for_document(
                api_key,
                question,
                answer,
                doc_chunks,
                model=model,
                relaxed=relaxed,
                use_structured_output=use_structured_output,
                relaxed_embeddings=relaxed_embeddings,
                embedding_model=embedding_model,
            )
        elif answer.strip():
            selected = []
            for rec in sorted(doc_chunks, key=lambda r: int(r["id"])):
                if answer_supported_by_chunk(answer, str(rec.get("text", ""))):
                    selected = [rec]
                    break
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
        '--relaxed-embeddings',
        action='store_true',
        help='Вместе с --relaxed (или --only-empty-long-answer): слоты 9–16 пула — по cosine similarity '
        'эмбеддингов чанка к «вопрос + ответ»; нужен пакет sentence-transformers. '
        'GPU: SIMPLEQA_EMBEDDING_DEVICE=cuda:1 или =1 (см. докстринг модуля).',
    )
    parser.add_argument(
        '--embedding-model',
        default=EMBED_MODEL_DEFAULT,
        help=f'Модель sentence-transformers для --relaxed-embeddings (по умолчанию: {EMBED_MODEL_DEFAULT})',
    )
    parser.add_argument(
        '--only-empty-long-answer',
        action='store_true',
        help='Только строки с пустым long_answer в существующем -o; включает режим relaxed для них'
    )
    parser.add_argument(
        '--no-structured-output',
        action='store_true',
        help='Не передавать response_format json_schema в OpenRouter (только свободный текст и разбор числа)',
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
    relaxed_embeddings = bool(args.relaxed_embeddings)
    if relaxed_embeddings and not relaxed:
        print(
            'Предупреждение: --relaxed-embeddings без --relaxed / --only-empty-long-answer не используется.',
            file=sys.stderr,
        )
        relaxed_embeddings = False

    total = len(rows)
    print(
        f'Обработка {total} строк... (relaxed={relaxed}, relaxed_embeddings={relaxed_embeddings})'
    )
    if relaxed_embeddings:
        ed = _embedding_torch_device()
        if ed:
            print(f'  Модель эмбеддингов: {args.embedding_model} (SIMPLEQA_EMBEDDING_DEVICE={ed})')
        else:
            print(
                f'  Модель эмбеддингов: {args.embedding_model} '
                f'(устройство по умолчанию; для другой GPU: SIMPLEQA_EMBEDDING_DEVICE=cuda:1 или =1)'
            )

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
            row,
            api_key,
            use_llm=not args.no_llm,
            model=args.model,
            relaxed=relaxed,
            use_structured_output=not args.no_structured_output,
            relaxed_embeddings=relaxed_embeddings,
            embedding_model=args.embedding_model,
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
        no_doc = sum(
            1
            for r in unfilled
            if not any(d.strip() for d in parse_documents(r.get('documents', '')))
        )
        with_docs = len(unfilled) - no_doc
        print(
            f'  Пустой long_answer: {len(unfilled)} строк '
            f'(без documents: {no_doc}; с documents, но не получилось извлечь: {with_docs}).'
        )
        if with_docs:
            print(
                '  Для второй попытки по «дырам» с документами: '
                '--only-empty-long-answer --relaxed [--relaxed-embeddings]'
            )


if __name__ == '__main__':
    main()
