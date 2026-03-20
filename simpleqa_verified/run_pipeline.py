#!/usr/bin/env python3
"""
Полный пайплайн для google/simpleqa-verified:
1. Скачивание документов
2. Извлечение long_answer
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def main():
    parser = argparse.ArgumentParser(description='Пайплайн: документы + long_answer для simpleqa-verified')
    parser.add_argument('-n', '--limit', type=int, default=0, help='Ограничить строк (0=все)')
    parser.add_argument('--no-llm', action='store_true', help='Без LLM при извлечении long_answer')
    parser.add_argument('--skip-download', action='store_true', help='Пропустить скачивание (если уже есть)')
    args = parser.parse_args()

    docs_csv = SCRIPT_DIR / 'simpleqa_verified_with_documents.csv'
    long_answer_csv = SCRIPT_DIR / 'simpleqa_verified_with_long_answer.csv'

    # Шаг 1: Скачивание документов
    if not args.skip_download:
        print('=== Шаг 1: Скачивание документов ===')
        cmd = [sys.executable, str(SCRIPT_DIR / 'download_documents_verified.py'), '-o', str(docs_csv)]
        if args.limit:
            cmd.extend(['-n', str(args.limit)])
        if subprocess.run(cmd, cwd=str(PROJECT_ROOT)) != 0:
            sys.exit(1)
    else:
        if not docs_csv.exists():
            print(f'Ошибка: файл {docs_csv} не найден. Запустите без --skip-download.')
            sys.exit(1)
        print('Пропуск скачивания (--skip-download)')

    # Шаг 2: Извлечение long_answer
    print('\n=== Шаг 2: Извлечение long_answer ===')
    cmd = [
        sys.executable, str(PROJECT_ROOT / 'extract_long_answer.py'),
        '-i', str(docs_csv),
        '-o', str(long_answer_csv),
    ]
    if args.no_llm:
        cmd.append('--no-llm')
    if args.limit:
        cmd.extend(['-n', str(args.limit)])

    if subprocess.run(cmd, cwd=str(PROJECT_ROOT)) != 0:
        sys.exit(1)

    print(f'\nГотово. Результат: {long_answer_csv}')


if __name__ == '__main__':
    main()
