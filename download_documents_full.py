#!/usr/bin/env python3
"""
Оптимизированный скрипт для скачивания документов на всем датасете.
"""

import pandas as pd
import requests
import time
import json
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import PyPDF2
import io
from typing import List, Dict, Any, Optional
import logging
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from collections import defaultdict
import signal
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/dolganov/simple_qa/download_progress.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedDocumentDownloader:
    def __init__(self, timeout=15, delay=0.1, max_workers=5):
        self.timeout = timeout
        self.delay = delay
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        # Кэш для уже скачанных документов
        self.document_cache = {}
        # Статистика
        self.stats = {
            'total_requests': 0,
            'cached_requests': 0,
            'successful_downloads': 0,
            'failed_downloads': 0
        }
    
    def get_url_hash(self, url: str) -> str:
        """Создает хэш URL для кэширования."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def download_document(self, url: str) -> Optional[str]:
        """
        Скачивает документ по URL с кэшированием.
        """
        # Проверяем кэш
        url_hash = self.get_url_hash(url)
        if url_hash in self.document_cache:
            self.stats['cached_requests'] += 1
            return self.document_cache[url_hash]
        
        self.stats['total_requests'] += 1
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Определяем тип контента
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                content = self._extract_pdf_text(response.content)
            elif 'html' in content_type or url.lower().endswith(('.html', '.htm')):
                content = self._extract_html_text(response.text)
            elif 'text' in content_type or url.lower().endswith('.txt'):
                content = response.text
            else:
                # Пытаемся обработать как HTML
                try:
                    content = self._extract_html_text(response.text)
                except:
                    content = response.text
            
            # Сохраняем в кэш
            if content:
                self.document_cache[url_hash] = content
                self.stats['successful_downloads'] += 1
                return content
            else:
                self.stats['failed_downloads'] += 1
                return None
                    
        except requests.exceptions.RequestException as e:
            self.stats['failed_downloads'] += 1
            return None
        except Exception as e:
            self.stats['failed_downloads'] += 1
            return None
    
    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Извлекает текст из PDF."""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            return ""
    
    def _extract_html_text(self, html_content: str) -> str:
        """Извлекает текст из HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Удаляем скрипты и стили
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Получаем текст
            text = soup.get_text()
            
            # Очищаем текст
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            return html_content
    
    def download_documents_parallel(self, urls: List[str]) -> List[str]:
        """
        Скачивает документы параллельно.
        """
        documents = []
        
        # Фильтруем и очищаем URL
        clean_urls = []
        for url in urls:
            if not url or not url.strip():
                continue
            url = url.strip()
            if url.startswith('http'):
                clean_urls.append(url)
        
        if not clean_urls:
            return documents
        
        # Скачиваем параллельно
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Создаем задачи
            future_to_url = {
                executor.submit(self.download_document, url): url 
                for url in clean_urls
            }
            
            # Собираем результаты
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    doc_content = future.result()
                    if doc_content:
                        documents.append(doc_content)
                except Exception as e:
                    logger.warning(f"Ошибка при обработке {url}: {e}")
                
                # Небольшая задержка между запросами
                time.sleep(self.delay)
        
        return documents

def signal_handler(sig, frame):
    """Обработчик сигнала для корректного завершения."""
    logger.info("Получен сигнал завершения. Сохраняю промежуточные результаты...")
    sys.exit(0)

def process_dataset_full(input_file: str, output_file: str):
    """
    Обработка полного датасета с сохранением промежуточных результатов.
    """
    # Настраиваем обработчик сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Загружаю датасет из {input_file}")
    
    # Загружаем CSV
    df = pd.read_csv(input_file)
    logger.info(f"Загружено {len(df)} строк")
    
    # Создаем объект для скачивания
    downloader = OptimizedDocumentDownloader()
    
    # Добавляем поле documents
    df['documents'] = None
    failed_downloads = 0
    
    # Обрабатываем с прогресс-баром
    for idx in tqdm(range(len(df)), desc="Обработка строк"):
        row = df.iloc[idx]
        
        try:
            # Парсим metadata для получения URLs
            metadata_str = row['metadata']
            if isinstance(metadata_str, str):
                # Пытаемся распарсить как JSON
                try:
                    metadata = json.loads(metadata_str)
                except:
                    # Если не JSON, пытаемся извлечь URLs с помощью regex
                    urls = re.findall(r'https?://[^\s\'\"]+', metadata_str)
                    metadata = {'urls': urls}
            else:
                metadata = metadata_str
            
            urls = metadata.get('urls', [])
            if not urls:
                df.at[idx, 'documents'] = []
                failed_downloads += 1
                continue
            
            # Скачиваем документы параллельно
            documents = downloader.download_documents_parallel(urls)
            df.at[idx, 'documents'] = documents
            
            if not documents:
                failed_downloads += 1
            
            # Сохраняем промежуточные результаты каждые 100 строк
            if (idx + 1) % 100 == 0:
                temp_file = f"{output_file}.temp"
                df.to_csv(temp_file, index=False)
                logger.info(f"Сохранен промежуточный результат: {idx + 1} строк обработано")
                
                # Логируем статистику
                logger.info(f"Статистика на строке {idx + 1}:")
                logger.info(f"  Кэшированных запросов: {downloader.stats['cached_requests']}")
                logger.info(f"  Успешных загрузок: {downloader.stats['successful_downloads']}")
                logger.info(f"  Неудачных загрузок: {downloader.stats['failed_downloads']}")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке строки {idx + 1}: {e}")
            df.at[idx, 'documents'] = []
            failed_downloads += 1
    
    # Сохраняем финальный результат
    logger.info(f"Сохраняю финальный результат в {output_file}")
    df.to_csv(output_file, index=False)
    
    # Удаляем временный файл
    temp_file = f"{output_file}.temp"
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # Статистика
    total_rows = len(df)
    successful_downloads = total_rows - failed_downloads
    
    logger.info(f"Обработка завершена!")
    logger.info(f"Всего строк: {total_rows}")
    logger.info(f"Успешно обработано: {successful_downloads}")
    logger.info(f"Не удалось скачать документы: {failed_downloads}")
    logger.info(f"Процент неудачных загрузок: {failed_downloads/total_rows*100:.2f}%")
    
    # Статистика кэширования
    logger.info(f"Статистика кэширования:")
    logger.info(f"Всего запросов: {downloader.stats['total_requests']}")
    logger.info(f"Кэшированных запросов: {downloader.stats['cached_requests']}")
    logger.info(f"Успешных загрузок: {downloader.stats['successful_downloads']}")
    logger.info(f"Неудачных загрузок: {downloader.stats['failed_downloads']}")
    
    # Сохраняем итоговый отчет
    save_final_report(total_rows, successful_downloads, failed_downloads, downloader.stats, output_file)
    
    return failed_downloads

def save_final_report(total_rows, successful_downloads, failed_downloads, stats, output_file):
    """
    Сохраняет итоговый отчет о результатах обработки.
    """
    report_file = output_file.replace('.csv', '_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ИТОГОВЫЙ ОТЧЕТ ОБ ОБРАБОТКЕ ДАТАСЕТА\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Дата и время завершения: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Исходный файл: simple_qa_test_set.csv\n")
        f.write(f"Результирующий файл: {os.path.basename(output_file)}\n\n")
        
        f.write("ОБЩАЯ СТАТИСТИКА:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Всего строк в датасете: {total_rows:,}\n")
        f.write(f"Успешно обработано: {successful_downloads:,}\n")
        f.write(f"Не удалось скачать документы: {failed_downloads:,}\n")
        f.write(f"Процент неудачных загрузок: {failed_downloads/total_rows*100:.2f}%\n\n")
        
        f.write("СТАТИСТИКА СКАЧИВАНИЯ ДОКУМЕНТОВ:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Всего HTTP запросов: {stats['total_requests']:,}\n")
        f.write(f"Кэшированных запросов: {stats['cached_requests']:,}\n")
        f.write(f"Успешных загрузок: {stats['successful_downloads']:,}\n")
        f.write(f"Неудачных загрузок: {stats['failed_downloads']:,}\n")
        f.write(f"Количество URL, по которым не удалось получить документы: {stats['failed_downloads']:,}\n")
        
        if stats['total_requests'] > 0:
            cache_efficiency = stats['cached_requests'] / (stats['cached_requests'] + stats['total_requests']) * 100
            f.write(f"Эффективность кэширования: {cache_efficiency:.2f}%\n")
        
        f.write("\n")
        
        f.write("ДЕТАЛИ ОБРАБОТКИ:\n")
        f.write("-" * 20 + "\n")
        f.write("• Обрабатывались документы по URL из поля 'urls' в метаданных\n")
        f.write("• Поддерживаемые форматы: HTML, PDF, текстовые файлы\n")
        f.write("• Использовалось параллельное скачивание (до 5 потоков)\n")
        f.write("• Применялось кэширование для избежания повторных запросов\n")
        f.write("• Промежуточные результаты сохранялись каждые 100 строк\n\n")
        
        f.write("ФАЙЛЫ РЕЗУЛЬТАТОВ:\n")
        f.write("-" * 20 + "\n")
        f.write(f"• Основной результат: {os.path.basename(output_file)}\n")
        f.write(f"• Лог обработки: download_progress.log\n")
        f.write(f"• Вывод программы: download_output.log\n")
        f.write(f"• Этот отчет: {os.path.basename(report_file)}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО\n")
        f.write("=" * 60 + "\n")
    
    logger.info(f"Итоговый отчет сохранен в: {report_file}")

if __name__ == "__main__":
    input_file = "/home/dolganov/simple_qa/simple_qa_test_set.csv"
    output_file = "/home/dolganov/simple_qa/simple_qa_test_set_with_documents.csv"
    
    failed_count = process_dataset_full(input_file, output_file)
    print(f"\nИтоговая статистика:")
    print(f"Количество примеров с недоступными документами: {failed_count}")
