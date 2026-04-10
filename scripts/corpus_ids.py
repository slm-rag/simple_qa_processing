"""
Стабильные doc_id / chunk_id для корпуса: один URL → один doc_id везде;
один и тот же чанк (тот же URL-слот, индекс и нормализованный текст) → один chunk_id.

doc_id = prefix + sha256(stable_doc_key)[:hex_len]
chunk_id = prefix + sha256(doc_key + idx + normalize_for_dedup(text))[:hex_len]

Пустой URL (редкий выравнивающий слот): ключ __no_url__:question_id:doc_index —
ид уникален в рамках вопроса, между вопросами не шарится.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse


def _normalize_for_dedup(text: str) -> str:
    """Как extract_long_answer.normalize_for_dedup (без импорта корня репо)."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def normalize_url(url: str) -> str:
    if not (url or "").strip():
        return ""
    p = urlparse(url.strip())
    scheme = (p.scheme or "https").lower()
    netloc = (p.netloc or "").lower()
    path = unquote(p.path or "")
    if path not in ("", "/") and path.endswith("/"):
        path = path.rstrip("/")
    q = sorted(parse_qsl(p.query, keep_blank_values=True))
    query = urlencode(q)
    return urlunparse((scheme, netloc, path, "", query, ""))


def stable_doc_key(url: str, question_id: str, doc_index: int) -> str:
    u = (url or "").strip()
    if u.startswith("http"):
        return normalize_url(u)
    return f"__no_url__:{question_id}:{doc_index}"


def make_doc_id(
    url: str,
    question_id: str,
    doc_index: int,
    *,
    prefix: str = "simple_qa_doc_",
    hex_len: int = 16,
) -> str:
    key = stable_doc_key(url, question_id, doc_index)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:hex_len]
    return f"{prefix}{h}"


def make_chunk_id(
    url: str,
    question_id: str,
    doc_index: int,
    chunk_index: int,
    text: str,
    *,
    prefix: str = "simple_qa_chunk_",
    hex_len: int = 16,
) -> str:
    doc_key = stable_doc_key(url, question_id, doc_index)
    nt = _normalize_for_dedup(str(text or ""))
    payload = f"{doc_key}\0{chunk_index}\0{nt}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:hex_len]
    return f"{prefix}{h}"
