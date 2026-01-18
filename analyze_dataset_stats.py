#!/usr/bin/env python3
"""
Скрипт для анализа статистики датасета simple_qa_test_set_with_documents.csv
"""

import pandas as pd
import json
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
        'answer_found_in_doc_index': None
    }
    
    # Парсим документы
    documents = parse_documents(row.get('documents', ''))
    result['num_documents'] = len(documents)
    
    if len(documents) == 0:
        return result
    
    # Определяем форматы документов из metadata
    metadata = parse_metadata(row.get('metadata', ''))
    urls = metadata.get('urls', [])
    formats = []
    for url in urls:
        fmt = get_document_format(url)
        if fmt != 'unknown':
            formats.append(fmt)
    result['document_formats'] = formats if formats else ['unknown']
    
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
    input_file = '/home/dolganov/simple_qa/simple_qa_test_set_with_documents.csv'
    output_file = '/home/dolganov/simple_qa/simple_qa_test_set_with_documents.csv'
    stats_file = '/home/dolganov/simple_qa/dataset_statistics.txt'
    
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
                    'answer_found_in_doc_index': None
                })
        all_results.extend(chunk_results)
    
    print("Добавление новых колонок в датасет...")
    # Перечитываем весь датасет для добавления колонок
    df = pd.read_csv(input_file)
    
    # Удаляем старые колонки статистики, если они есть
    cols_to_remove = ['num_documents', 'document_formats', 'avg_document_length_words', 
                      'answer_found_in_documents', 'answer_position_words', 
                      'answer_position_chars', 'answer_found_in_doc_index']
    for col in cols_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Добавляем новые колонки
    df['num_documents'] = [r['num_documents'] for r in all_results]
    df['document_formats'] = [','.join(r['document_formats']) if r['document_formats'] else '' for r in all_results]
    df['avg_document_length_words'] = [r['avg_document_length_words'] for r in all_results]
    df['answer_found_in_documents'] = [r['answer_found_in_documents'] for r in all_results]
    df['answer_position_words'] = [r['answer_position_words'] if r['answer_position_words'] is not None else '' for r in all_results]
    df['answer_position_chars'] = [r['answer_position_chars'] if r['answer_position_chars'] is not None else '' for r in all_results]
    df['answer_found_in_doc_index'] = [r['answer_found_in_doc_index'] if r['answer_found_in_doc_index'] is not None else '' for r in all_results]
    
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
    
    # Статистика по форматам
    all_formats = []
    for r in all_results:
        all_formats.extend(r['document_formats'])
    format_counts = {}
    for fmt in all_formats:
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    # Генерируем отчет
    stats_report = f"""
СТАТИСТИКА ПО ДАТАСЕТУ: simple_qa_test_set_with_documents.csv
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

5. ФОРМАТЫ ДОКУМЕНТОВ:
"""
    for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
        stats_report += f"   {fmt}: {count} ({count / len(all_formats) * 100:.2f}%)\n"
    
    stats_report += f"\n{'='*60}\n"
    stats_report += f"Всего обработано строк: {len(all_results)}\n"
    
    print(stats_report)
    
    # Сохраняем отчет
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(stats_report)
    
    print(f"\nСтатистика сохранена в {stats_file}")
    print(f"Обновленный датасет сохранен в {output_file}")

if __name__ == '__main__':
    main()

