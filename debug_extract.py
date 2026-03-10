#!/usr/bin/env python3
"""
Отладочный скрипт для анализа извлечения long_answer.
Запускает extract_long_answer на нескольких примерах с подробным выводом.
"""

import csv
import json
import sys
from extract_long_answer import (
    parse_documents,
    get_segments,
    bm25_filter_segments,
    _get_fragment_variant2,
    process_row,
    count_words,
)


def debug_row(row_idx: int, row: dict, use_llm: bool = False):
    """Печатает отладочную информацию для одной строки."""
    documents = parse_documents(row.get('documents', ''))
    question = str(row.get('problem', '')).strip()
    answer = str(row.get('answer', '')).strip()

    print("\n" + "=" * 80)
    print(f"СТРОКА {row_idx + 1}")
    print("=" * 80)
    print(f"Вопрос: {question}")
    print(f"Ответ:  {answer}")
    print(f"Документов: {len(documents)}")
    print(f"Ответ в документах? {answer.lower() in ' '.join(documents).lower()}")

    for doc_idx, doc in enumerate(documents):
        print(f"\n--- Документ {doc_idx + 1} ({count_words(doc)} слов) ---")
        # Первые 200 символов
        preview = doc[:300].replace('\n', ' ')
        print(f"Начало: {preview}...")

        segments = get_segments(doc)
        print(f"Сегментов (абзацев/чанков): {len(segments)}")

        if segments:
            top = bm25_filter_segments(question, segments, 5, answer)
            print(f"BM25 top-5 (по вопросу+ответу):")
            for i, s in enumerate(top):
                contains_answer = answer.lower() in s.lower() if answer else False
                marker = " ✓ ОТВЕТ ЕСТЬ" if contains_answer else ""
                print(f"  [{i+1}] ({count_words(s)} сл.) {s[:120].replace(chr(10), ' ')}...{marker}")


def main():
    input_path = "/home/dolganov/simple_qa/simple_qa_test_set_with_documents.csv"
    n_rows = 5
    use_llm = "--llm" in sys.argv  # по умолчанию без LLM для скорости

    print("Чтение CSV...")
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= n_rows - 1:
                break

    print(f"Отладка {len(rows)} строк (use_llm={use_llm})")
    for i, row in enumerate(rows):
        debug_row(i, row, use_llm)

    # Запуск полного process_row для сравнения
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТ process_row (без LLM)")
    print("=" * 80)
    tokenizer, model = None, None
    for i, row in enumerate(rows):
        fragments = process_row(row, tokenizer, model, use_llm=False)
        print(f"\nСтрока {i+1} -> {len(fragments)} фрагментов:")
        for j, f in enumerate(fragments):
            print(f"  [{j+1}] ({count_words(f)} сл.) {f[:150].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
