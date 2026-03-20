#!/usr/bin/env python3
"""
Скачивание документов для датасета google/simpleqa-verified.
Загружает датасет из HuggingFace, скачивает документы по URL и сохраняет в CSV.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Добавляем корень проекта в path для импорта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from download_documents_full import OptimizedDocumentDownloader

# Настройка логирования
LOG_DIR = Path(__file__).resolve().parent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'download_progress.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_urls(urls_str: str) -> List[str]:
    """
    Парсит строку с URL (через запятую) в список.
    Убирает лишние символы вроде trailing ')'.
    """
    if not urls_str or not str(urls_str).strip():
        return []
    urls = []
    for part in str(urls_str).split(','):
        url = part.strip().rstrip(')')
        if url.startswith('http'):
            urls.append(url)
    return urls


def process_dataset(
    output_file: Optional[str] = None,
    limit: int = 0,
    save_every: int = 50,
) -> str:
    """
    Загружает google/simpleqa-verified, скачивает документы, сохраняет в CSV.
    
    Args:
        output_file: путь к выходному CSV (по умолчанию в папке скрипта)
        limit: ограничить количество строк (0 = все)
        save_every: сохранять промежуточный результат каждые N строк
    
    Returns:
        путь к сохранённому файлу
    """
    if output_file is None:
        output_file = str(LOG_DIR / 'simpleqa_verified_with_documents.csv')

    logger.info('Загружаю датасет google/simpleqa-verified...')
    ds = load_dataset('google/simpleqa-verified', split='eval')
    df = ds.to_pandas()

    if limit > 0:
        df = df.head(limit)
        logger.info(f'Ограничено {limit} строками')

    logger.info(f'Загружено {len(df)} строк')

    # Добавляем колонку documents
    df['documents'] = None
    downloader = OptimizedDocumentDownloader()
    failed_downloads = 0

    for idx in tqdm(range(len(df)), desc='Скачивание документов'):
        row = df.iloc[idx]
        urls_str = row.get('urls', '')
        urls = parse_urls(urls_str)

        if not urls:
            df.at[idx, 'documents'] = json.dumps([], ensure_ascii=False)
            failed_downloads += 1
            continue

        documents = downloader.download_documents_parallel(urls)
        df.at[idx, 'documents'] = json.dumps(documents, ensure_ascii=False)

        if not documents:
            failed_downloads += 1

        if (idx + 1) % save_every == 0:
            temp_file = output_file + '.temp'
            df.to_csv(temp_file, index=False)
            logger.info(f'Промежуточное сохранение: {idx + 1} строк')

    df.to_csv(output_file, index=False)

    # Удаляем временный файл
    temp_file = output_file + '.temp'
    if os.path.exists(temp_file):
        os.remove(temp_file)

    total = len(df)
    successful = total - failed_downloads
    logger.info(f'Готово. Успешно: {successful}/{total}, без документов: {failed_downloads}')

    return output_file


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Скачивание документов для google/simpleqa-verified')
    parser.add_argument('-o', '--output', default=None, help='Выходной CSV')
    parser.add_argument('-n', '--limit', type=int, default=0, help='Ограничить количество строк (0=все)')
    parser.add_argument('--save-every', type=int, default=50, help='Сохранять каждые N строк')
    args = parser.parse_args()

    output = process_dataset(
        output_file=args.output,
        limit=args.limit,
        save_every=args.save_every,
    )
    print(f'\nРезультат сохранён: {output}')


if __name__ == '__main__':
    main()
