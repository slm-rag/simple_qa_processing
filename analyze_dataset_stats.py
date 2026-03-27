#!/usr/bin/env python3
"""
Скрипт для анализа статистики датасета с long_answer.
Поддерживает simple_qa_test_set и simpleqa_verified (колонки urls или metadata).
"""

import argparse
import csv
import json

csv.field_size_limit(10**7)

import pandas as pd
import ast
import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import statistics

def parse_documents(documents_str: str) -> List[str]:
    """Парсит строку с документами в список."""
    if pd.isna(documents_str) or documents_str == '' or documents_str == '[]':
        return []
    try:
        if isinstance(documents_str, str):
            documents_str = documents_str.strip()
            # Если это пустой список
            if documents_str == '[]' or documents_str == "''" or documents_str == '""':
                return []
            
            # Сначала пытаемся использовать ast.literal_eval (безопаснее для Python литералов)
            try:
                parsed = ast.literal_eval(documents_str)
                if isinstance(parsed, list):
                    return [str(doc) for doc in parsed if doc and str(doc).strip()]
                elif isinstance(parsed, str) and parsed:
                    # Если это одна строка, возвращаем как список с одним элементом
                    return [parsed]
                return []
            except (ValueError, SyntaxError):
                # Если ast.literal_eval не сработал, пробуем JSON
                try:
                    # Заменяем одинарные кавычки на двойные для JSON (осторожно)
                    # Но это может сломать строки с одинарными кавычками внутри
                    parsed = json.loads(documents_str)
                    if isinstance(parsed, list):
                        return [str(doc) for doc in parsed if doc and str(doc).strip()]
                    return []
                except (json.JSONDecodeError, ValueError):
                    # Если и JSON не сработал, возможно это просто одна строка
                    if documents_str and documents_str.strip():
                        return [documents_str]
                    return []
        else:
            # Если это уже список
            if isinstance(documents_str, list):
                return [str(doc) for doc in documents_str if doc and str(doc).strip()]
            return []
    except Exception as e:
        # Если ничего не получилось, возвращаем пустой список
        return []


def parse_long_answer(long_answer_str: str) -> List[str]:
    """Парсит строку с long_answer в список фрагментов."""
    if pd.isna(long_answer_str) or long_answer_str == '' or long_answer_str == '[]':
        return []
    try:
        if isinstance(long_answer_str, str):
            long_answer_str = long_answer_str.strip()
            if long_answer_str == '[]' or long_answer_str in ("''", '""'):
                return []
            try:
                parsed = ast.literal_eval(long_answer_str)
            except (ValueError, SyntaxError):
                parsed = json.loads(long_answer_str)
            if isinstance(parsed, list):
                return [str(frag) for frag in parsed if frag and str(frag).strip()]
            return []
        if isinstance(long_answer_str, list):
            return [str(frag) for frag in long_answer_str if frag and str(frag).strip()]
        return []
    except Exception:
        return []


def parse_urls_column(urls_str: str) -> List[str]:
    """Парсит колонку urls (simpleqa_verified: список или строка через запятую)."""
    if pd.isna(urls_str) or urls_str == '' or urls_str == '[]':
        return []
    try:
        if isinstance(urls_str, str):
            urls_str = urls_str.strip()
            if urls_str.startswith('['):
                parsed = ast.literal_eval(urls_str) if "'" in urls_str or '"' in urls_str else json.loads(urls_str)
                return [str(u).strip().rstrip(')') for u in parsed if u and str(u).strip().startswith('http')]
            urls = re.findall(r'https?://[^\s,\]]+', urls_str)
            return [u.rstrip(')') for u in urls]
        if isinstance(urls_str, list):
            return [str(u).strip().rstrip(')') for u in urls_str if u and str(u).strip().startswith('http')]
    except Exception:
        pass
    return []


def parse_metadata(metadata_str: str) -> Dict[str, Any]:
    """Парсит metadata для получения URLs."""
    if pd.isna(metadata_str) or metadata_str == '':
        return {'urls': []}
    try:
        if isinstance(metadata_str, str):
            # Пытаемся распарсить как JSON
            try:
                metadata = json.loads(metadata_str)
            except:
                # Если не JSON, пытаемся извлечь URLs с помощью regex
                urls = re.findall(r'https?://[^\s\'"]+', metadata_str)
                metadata = {'urls': urls}
        else:
            metadata = metadata_str
        return metadata if isinstance(metadata, dict) else {'urls': []}
    except Exception as e:
        return {'urls': []}

def is_wikipedia_url(url: str) -> bool:
    """Проверяет, является ли URL ссылкой на Wikipedia."""
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower()
        wikipedia_domains = [
            'en.wikipedia.org', 'ru.wikipedia.org', 'de.wikipedia.org',
            'fr.wikipedia.org', 'es.wikipedia.org', 'it.wikipedia.org',
            'ja.wikipedia.org', 'zh.wikipedia.org', 'pt.wikipedia.org',
            'pl.wikipedia.org', 'www.wikipedia.org'
        ]
        return any(hostname.endswith(d) for d in wikipedia_domains) and '/wiki/' in parsed.path
    except Exception:
        return False


def get_document_format(url: str) -> str:
    """Определяет формат документа по URL."""
    url_lower = url.lower()
    if url_lower.endswith('.pdf'):
        return 'pdf'
    elif url_lower.endswith(('.html', '.htm')):
        return 'html'
    elif url_lower.endswith('.txt'):
        return 'txt'
    else:
        # Пытаемся определить по домену или пути
        parsed = urlparse(url)
        path = parsed.path.lower()
        if '.pdf' in path:
            return 'pdf'
        elif '.html' in path or '.htm' in path:
            return 'html'
        elif '.txt' in path:
            return 'txt'
        else:
            return 'unknown'

def count_words(text: str) -> int:
    """Подсчитывает количество слов в тексте."""
    if not text or pd.isna(text):
        return 0
    # Разбиваем по пробелам и фильтруем пустые строки
    words = [w for w in text.split() if w.strip()]
    return len(words)

def find_answer_position(answer: str, document: str) -> Optional[Tuple[int, int]]:
    """
    Находит позицию ответа в документе.
    Возвращает (позиция_в_словах, позиция_в_символах) или None.
    """
    if not answer or not document or pd.isna(answer) or pd.isna(document):
        return None
    
    answer_clean = str(answer).strip()
    document_clean = str(document)
    
    # Ищем точное вхождение (регистронезависимо)
    answer_lower = answer_clean.lower()
    document_lower = document_clean.lower()
    
    pos = document_lower.find(answer_lower)
    if pos == -1:
        return None
    
    # Подсчитываем позицию в словах от начала документа
    text_before = document_clean[:pos]
    words_before = count_words(text_before)
    
    return (words_before, pos)

def analyze_row(row: pd.Series) -> Dict[str, Any]:
    """Анализирует одну строку датасета."""
    result = {
        'num_documents': 0,
        'document_formats': [],
        'avg_document_length_words': 0.0,
        'answer_found_in_documents': False,
        'answer_position_words': None,
        'answer_position_chars': None,
        'answer_found_in_doc_index': None,
        'num_long_answer_fragments': 0,
        'total_long_answer_words': 0,
        'avg_long_answer_fragment_words': 0.0,
        'answer_found_in_long_answer': False,
        'num_documents_from_wikipedia': 0,
        'num_documents_from_other': 0,
    }
    
    # Парсим документы
    documents = parse_documents(row.get('documents', ''))
    result['num_documents'] = len(documents)
    
    # Парсим long_answer
    long_answer_fragments = parse_long_answer(row.get('long_answer', ''))
    result['num_long_answer_fragments'] = len(long_answer_fragments)
    
    if long_answer_fragments:
        frag_lengths = [count_words(f) for f in long_answer_fragments]
        result['total_long_answer_words'] = sum(frag_lengths)
        result['avg_long_answer_fragment_words'] = statistics.mean(frag_lengths)
        answer = row.get('answer', '')
        if answer:
            answer_lower = str(answer).strip().lower()
            result['answer_found_in_long_answer'] = any(
                answer_lower in frag.lower() for frag in long_answer_fragments
            )
    
    if len(documents) == 0:
        return result
    
    # Определяем форматы документов и источники (Wikipedia / другие)
    metadata = parse_metadata(row.get('metadata', ''))
    urls = metadata.get('urls', [])
    if not urls or not isinstance(urls, list):
        urls = parse_urls_column(row.get('urls', ''))
    formats = []
    wikipedia_count = 0
    other_count = 0
    for i, url in enumerate(urls):
        fmt = get_document_format(url)
        formats.append(fmt)  # всегда добавляем (включая unknown), чтобы сходилось с числом документов
        # Документы и URL в одном порядке; считаем только успешно загруженные (непустые)
        if i < len(documents) and documents[i] and str(documents[i]).strip():
            if is_wikipedia_url(url):
                wikipedia_count += 1
            else:
                other_count += 1
    result['document_formats'] = formats if formats else ['unknown']
    result['num_documents_from_wikipedia'] = wikipedia_count
    result['num_documents_from_other'] = other_count
    
    # Вычисляем среднюю длину документов в словах
    doc_lengths = [count_words(doc) for doc in documents]
    if doc_lengths:
        result['avg_document_length_words'] = statistics.mean(doc_lengths)
    
    # Ищем ответ в документах
    answer = row.get('answer', '')
    if answer:
        for idx, doc in enumerate(documents):
            pos = find_answer_position(answer, doc)
            if pos:
                result['answer_found_in_documents'] = True
                result['answer_position_words'] = pos[0]
                result['answer_position_chars'] = pos[1]
                result['answer_found_in_doc_index'] = idx
                break
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Анализ статистики датасета с long_answer')
    parser.add_argument('-i', '--input', default='/home/dolganov/simple_qa/simple_qa_test_set_with_long_answer.csv',
                        help='Входной CSV')
    parser.add_argument('-o', '--output', default=None,
                        help='Обновить датасет колонками статистики (по умолчанию не сохранять)')
    parser.add_argument('-s', '--stats', default=None,
                        help='Файл отчёта статистики (по умолчанию: dataset_statistics.txt рядом с входным)')
    args = parser.parse_args()

    from pathlib import Path
    input_file = args.input
    output_file = args.output
    stats_file = args.stats or str(Path(input_file).parent / 'dataset_statistics.txt')

    print(f"Загрузка датасета из {input_file}...")
    # Читаем CSV файл частями для экономии памяти
    chunk_size = 1000
    all_results = []
    total_rows = 0
    
    # Сначала подсчитаем общее количество строк
    print("Подсчет общего количества строк...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # -1 для заголовка
    
    print(f"Всего строк: {total_rows}")
    print("Обработка датасета...")
    
    # Читаем и обрабатываем по частям
    for chunk in tqdm(pd.read_csv(input_file, chunksize=chunk_size), 
                          total=(total_rows // chunk_size + 1), 
                          desc="Обработка"):
        chunk_results = []
        for idx, row in chunk.iterrows():
            try:
                analysis = analyze_row(row)
                chunk_results.append(analysis)
            except Exception as e:
                print(f"Ошибка при обработке строки {idx}: {e}")
                chunk_results.append({
                    'num_documents': 0,
                    'document_formats': [],
                    'avg_document_length_words': 0.0,
                    'answer_found_in_documents': False,
                    'answer_position_words': None,
                    'answer_position_chars': None,
                    'answer_found_in_doc_index': None,
                    'num_long_answer_fragments': 0,
                    'total_long_answer_words': 0,
                    'avg_long_answer_fragment_words': 0.0,
                    'answer_found_in_long_answer': False,
                    'num_documents_from_wikipedia': 0,
                    'num_documents_from_other': 0,
                })
        all_results.extend(chunk_results)
    
    if output_file:
        print("Добавление новых колонок в датасет...")
        df = pd.read_csv(input_file)
        cols_to_remove = ['num_documents', 'document_formats', 'avg_document_length_words',
                          'answer_found_in_documents', 'answer_position_words',
                          'answer_position_chars', 'answer_found_in_doc_index',
                          'num_long_answer_fragments', 'total_long_answer_words',
                          'avg_long_answer_fragment_words', 'answer_found_in_long_answer',
                          'num_documents_from_wikipedia', 'num_documents_from_other']
        for col in cols_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
        df['num_documents'] = [r['num_documents'] for r in all_results]
        df['document_formats'] = [','.join(r['document_formats']) if r['document_formats'] else '' for r in all_results]
        df['avg_document_length_words'] = [r['avg_document_length_words'] for r in all_results]
        df['answer_found_in_documents'] = [r['answer_found_in_documents'] for r in all_results]
        df['answer_position_words'] = [r['answer_position_words'] if r['answer_position_words'] is not None else '' for r in all_results]
        df['answer_position_chars'] = [r['answer_position_chars'] if r['answer_position_chars'] is not None else '' for r in all_results]
        df['answer_found_in_doc_index'] = [r['answer_found_in_doc_index'] if r['answer_found_in_doc_index'] is not None else '' for r in all_results]
        df['num_long_answer_fragments'] = [r['num_long_answer_fragments'] for r in all_results]
        df['total_long_answer_words'] = [r['total_long_answer_words'] for r in all_results]
        df['avg_long_answer_fragment_words'] = [r['avg_long_answer_fragment_words'] for r in all_results]
        df['answer_found_in_long_answer'] = [r['answer_found_in_long_answer'] for r in all_results]
        df['num_documents_from_wikipedia'] = [r['num_documents_from_wikipedia'] for r in all_results]
        df['num_documents_from_other'] = [r['num_documents_from_other'] for r in all_results]
        print(f"Сохранение обновленного датасета в {output_file}...")
        df.to_csv(output_file, index=False)
    
    # Вычисляем статистику
    print("Вычисление статистики...")
    num_docs_list = [r['num_documents'] for r in all_results]
    questions_without_docs = sum(1 for n in num_docs_list if n == 0)
    avg_doc_lengths = [r['avg_document_length_words'] for r in all_results if r['avg_document_length_words'] > 0]
    answer_found_count = sum(1 for r in all_results if r['answer_found_in_documents'])
    answer_positions_words = [r['answer_position_words'] for r in all_results
                          if r['answer_position_words'] is not None]

    # Статистика по источникам документов
    total_wikipedia_docs = sum(r['num_documents_from_wikipedia'] for r in all_results)
    total_other_docs = sum(r['num_documents_from_other'] for r in all_results)
    total_docs_all = total_wikipedia_docs + total_other_docs

    # Статистика по long_answer
    num_fragments_list = [r['num_long_answer_fragments'] for r in all_results]
    questions_without_long_answer = sum(1 for n in num_fragments_list if n == 0)
    avg_frag_lengths = [r['avg_long_answer_fragment_words'] for r in all_results if r['avg_long_answer_fragment_words'] > 0]
    answer_in_long_answer_count = sum(1 for r in all_results if r['answer_found_in_long_answer'])

    # Статистика по форматам
    all_formats = []
    for r in all_results:
        all_formats.extend(r['document_formats'])
    format_counts = {}
    for fmt in all_formats:
        format_counts[fmt] = format_counts.get(fmt, 0) + 1

    dataset_name = Path(input_file).name
    stats_report = f"""
СТАТИСТИКА ПО ДАТАСЕТУ: {dataset_name}
{'='*60}

1. КОЛИЧЕСТВО ДОКУМЕНТОВ НА ВОПРОС:
   Среднее: {statistics.mean(num_docs_list):.2f}
   Медиана: {statistics.median(num_docs_list):.2f}
   Максимальное: {max(num_docs_list)}
   Минимальное: {min(num_docs_list)}
   Стандартное отклонение: {statistics.stdev(num_docs_list) if len(num_docs_list) > 1 else 0:.2f}

2. КОЛИЧЕСТВО ВОПРОСОВ БЕЗ ДОКУМЕНТОВ: {questions_without_docs}
   Процент от общего числа: {questions_without_docs / len(all_results) * 100:.2f}%

3. СРЕДНЯЯ ДЛИНА ДОКУМЕНТА В СЛОВАХ:
   Среднее: {statistics.mean(avg_doc_lengths) if avg_doc_lengths else 0:.2f}
   Медиана: {statistics.median(avg_doc_lengths) if avg_doc_lengths else 0:.2f}
   Максимальная: {max(avg_doc_lengths) if avg_doc_lengths else 0:.0f}
   Минимальная: {min(avg_doc_lengths) if avg_doc_lengths else 0:.0f}

4. ТОЧНОЕ ВХОЖДЕНИЕ ОТВЕТА В ДОКУМЕНТ:
   Количество вопросов с найденным ответом: {answer_found_count}
   Процент от общего числа: {answer_found_count / len(all_results) * 100:.2f}%
   
   Статистика по позициям ответа (в словах от начала документа):
   Среднее: {statistics.mean(answer_positions_words) if answer_positions_words else 0:.2f}
   Медиана: {statistics.median(answer_positions_words) if answer_positions_words else 0:.2f}
   Максимальная: {max(answer_positions_words) if answer_positions_words else 0:.0f}
   Минимальная: {min(answer_positions_words) if answer_positions_words else 0:.0f}

5. LONG_ANSWER (извлечённые фрагменты):
   Количество фрагментов на вопрос:
   Среднее: {statistics.mean(num_fragments_list):.2f}
   Медиана: {statistics.median(num_fragments_list):.2f}
   Максимальное: {max(num_fragments_list)}
   Минимальное: {min(num_fragments_list)}
   
   Вопросов без long_answer: {questions_without_long_answer}
   Процент от общего числа: {questions_without_long_answer / len(all_results) * 100:.2f}%
   
   Средняя длина фрагмента (слов):
   Среднее: {statistics.mean(avg_frag_lengths) if avg_frag_lengths else 0:.2f}
   Медиана: {statistics.median(avg_frag_lengths) if avg_frag_lengths else 0:.2f}
   
   Ответ найден в long_answer: {answer_in_long_answer_count}
   Процент от общего числа: {answer_in_long_answer_count / len(all_results) * 100:.2f}%

6. ФОРМАТЫ ДОКУМЕНТОВ:
"""
    for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
        stats_report += f"   {fmt}: {count} ({count / len(all_formats) * 100:.2f}%)\n"

    total_formats = sum(len(r['document_formats']) for r in all_results)
    stats_report += f"""
7. ИСТОЧНИКИ ДОКУМЕНТОВ (успешно загруженные):
   Из Wikipedia: {total_wikipedia_docs:,}
   Из других источников: {total_other_docs:,}
   Всего документов: {total_docs_all:,}
"""
    if total_docs_all > 0:
        stats_report += f"   Wikipedia: {total_wikipedia_docs / total_docs_all * 100:.1f}%  |  Другие: {total_other_docs / total_docs_all * 100:.1f}%\n"
    stats_report += f"\n   Сводка: всего URL/слотов (форматы): {total_formats:,}  |  успешно загружено (источники): {total_docs_all:,}\n"
    
    stats_report += f"\n{'='*60}\n"
    stats_report += f"Всего обработано строк: {len(all_results)}\n"
    
    print(stats_report)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(stats_report)
    
    print(f"\nСтатистика сохранена в {stats_file}")
    if output_file:
        print(f"Обновленный датасет сохранен в {output_file}")

if __name__ == '__main__':
    main()

