#!/usr/bin/env python3
"""
Извлечение абзаца (long_answer) из документов для каждого вопроса.
- Для answer_found_in_documents: окно вокруг позиции ответа + проверка LLM
- Для остальных: разбивка на чанки + выбор лучшего через LLM (Qwen3-8B)
"""

import csv
import ast
import re
import argparse
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
# Qwen2.5-7B-Instruct: совместим с Python 3.8 и transformers 4.37+
# Qwen3-8B требует transformers>=4.51 и Python>=3.9
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


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


def bm25_filter_segments(question: str, segments: List[str], top_k: int = BM25_TOP_K) -> List[str]:
    """
    BM25 предфильтр: возвращает top_k наиболее релевантных сегментов по вопросу.
    Уменьшает нагрузку на LLM.
    """
    if not segments or not question:
        return segments
    if len(segments) <= top_k:
        return segments
    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [tokenize_for_bm25(s) for s in segments]
        tokenized_query = tokenize_for_bm25(question)
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


def load_llm():
    """Загружает модель и токенизатор."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model


def llm_verify_paragraph(
    tokenizer,
    model,
    question: str,
    answer: str,
    paragraph: str,
    enable_thinking: bool = False
) -> bool:
    """
    Спрашивает LLM: содержит ли абзац ответ на вопрос?
    Возвращает True если LLM считает, что да.
    """
    prompt = f"""Текст:
{paragraph[:2000]}

Вопрос: {question}
Ожидаемый краткий ответ: {answer}

Содержит ли приведённый текст этот ответ в явном или неявном виде? Ответь только одним словом: Да или Нет."""

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    import torch
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return 'да' in response.lower() or 'yes' in response.lower()


def llm_pick_best_chunk(
    tokenizer,
    model,
    question: str,
    answer: str,
    chunks: List[str],
    enable_thinking: bool = False
) -> int:
    """
    Просит LLM выбрать номер чанка (1-based), который содержит ответ.
    Возвращает индекс (0-based) или -1 если ни один не подходит.
    """
    if not chunks:
        return -1
    formatted = "\n\n".join(f"[{i+1}]\n{c[:1500]}" for i, c in enumerate(chunks[:MAX_CHUNKS_FOR_LLM]))
    prompt = f"""Вопрос: {question}
Ожидаемый краткий ответ: {answer}

Фрагменты текста:
{formatted}

В каком по счёту фрагменте [1], [2], ... содержится ответ на вопрос? Ответь только номером (1, 2, 3...) или 0, если ни в одном."""

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    import torch
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    # Извлекаем число
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        n = int(numbers[0])
        if 1 <= n <= len(chunks):
            return n - 1
        if n == 0:
            return -1
    return 0  # fallback: первый чанк


def process_row(
    row: dict,
    tokenizer,
    model,
    use_llm: bool = True,
    enable_thinking: bool = False
) -> str:
    """
    Обрабатывает одну строку, возвращает long_answer.
    """
    documents = parse_documents(row.get('documents', ''))
    question = str(row.get('problem', '')).strip()
    answer = str(row.get('answer', '')).strip()
    answer_found = str(row.get('answer_found_in_documents', '')).strip().lower() in ('true', '1', 'yes')
    pos_chars = row.get('answer_position_chars', '')
    doc_idx = row.get('answer_found_in_doc_index', '')

    if not documents or not question:
        return ''

    # Собираем все документы в один текст для поиска
    all_docs = documents
    candidate = ''

    if answer_found and pos_chars != '' and doc_idx != '':
        try:
            idx = int(doc_idx)
            pos = int(pos_chars)
            if 0 <= idx < len(documents):
                doc = documents[idx]
                # Сначала пробуем абзац, содержащий ответ
                para = get_paragraph_containing_position(doc, pos)
                if para and count_words(para) <= MAX_PARAGRAPH_WORDS:
                    candidate = para
                else:
                    candidate = extract_context_window(doc, answer, pos_chars)
        except (ValueError, TypeError):
            pass

        if not candidate and documents:
            # Fallback: ищем во всех документах
            for doc in documents:
                if answer.lower() in doc.lower():
                    pos = doc.lower().find(answer.lower())
                    para = get_paragraph_containing_position(doc, pos)
                    candidate = para if para else extract_context_window(doc, answer, pos)
                    break

    if not candidate:
        # Ответ не найден явно — абзацы или чанки, BM25 предфильтр, выбор через LLM
        combined = ' '.join(all_docs)
        segments = get_segments(combined)
        if not segments:
            return ''
        # BM25 предфильтр: подаём в LLM только top-K релевантных сегментов
        segments = bm25_filter_segments(question, segments, BM25_TOP_K)
        if use_llm and tokenizer is not None and model is not None:
            best_idx = llm_pick_best_chunk(tokenizer, model, question, answer, segments, enable_thinking)
            if best_idx >= 0:
                candidate = segments[best_idx]
            else:
                candidate = segments[0]  # fallback
        else:
            candidate = segments[0]

    # Финальная проверка LLM для answer_found
    if candidate and use_llm and tokenizer is not None and model is not None and answer_found:
        if not llm_verify_paragraph(tokenizer, model, question, answer, candidate, enable_thinking):
            # LLM не подтвердил — можно попробовать расширить окно или оставить как есть
            # Оставляем candidate, т.к. answer точно есть в документе
            pass

    return candidate


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
        '--thinking',
        action='store_true',
        help='Включить режим рассуждений у Qwen3 (медленнее, но точнее)'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=0,
        help='Ограничить количество обрабатываемых строк (0 = все)'
    )
    args = parser.parse_args()

    tokenizer, model = None, None
    if not args.no_llm:
        print(f'Загрузка модели {MODEL_NAME}...')
        tokenizer, model = load_llm()
        print('Модель загружена.')

    print(f'Чтение {args.input}...')
    rows = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) + ['long_answer']
        for row in reader:
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break

    total = len(rows)
    print(f'Обработка {total} строк...')

    from tqdm import tqdm
    for row in tqdm(rows, desc='Извлечение long_answer'):
        row['long_answer'] = process_row(row, tokenizer, model, use_llm=not args.no_llm, enable_thinking=args.thinking)

    print(f'Сохранение в {args.output}...')
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    filled = sum(1 for r in rows if r.get('long_answer', '').strip())
    print(f'Готово. Заполнено long_answer: {filled}/{total}')


if __name__ == '__main__':
    main()
