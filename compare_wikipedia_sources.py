#!/usr/bin/env python3
"""
Скрипт для сравнения контента Wikipedia, полученного двумя способами:
1. Через MediaWiki API (prop=extracts) - текущий способ в download_documents_full.py
2. Через прямую загрузку HTML страницы - старый способ
"""

import requests
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
import sys

# Добавляем путь к модулю
sys.path.insert(0, '/home/dolganov/simple_qa')
from download_documents_full import OptimizedDocumentDownloader


def get_via_api(url: str):
    """Получает контент через Wikipedia API (как в download_documents_full.py)."""
    downloader = OptimizedDocumentDownloader()
    content = downloader.download_wikipedia_via_api(url)
    return content, len(content) if content else 0


def get_via_direct_html(url: str):
    """Получает контент через прямую загрузку HTML (старый способ)."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    response = session.get(url, timeout=15)
    response.raise_for_status()
    
    # Используем тот же метод _extract_html_text что и в downloader
    downloader = OptimizedDocumentDownloader()
    content = downloader._extract_html_text(response.text)
    return content, len(content) if content else 0


def extract_main_content(html: str) -> str:
    """
    Извлекает основной контент статьи из HTML Wikipedia.
    Ищем div с id='mw-content-text' или body content.
    """
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()
    
    # Wikipedia хранит основной контент в mw-content-text
    content_div = soup.find('div', id='mw-content-text')
    if content_div:
        text = content_div.get_text()
    else:
        text = soup.get_text()
    
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return ' '.join(chunk for chunk in chunks if chunk)


def get_via_direct_html_main_content(url: str):
    """
    Получает контент через HTML, но извлекает только основной контент статьи
    (без навигации, инфобокса и т.д.) - более честное сравнение.
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    response = session.get(url, timeout=15)
    response.raise_for_status()
    
    content = extract_main_content(response.text)
    return content, len(content) if content else 0


def main():
    url = "https://en.wikipedia.org/wiki/The_Oceanography_Society"
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    
    print("=" * 70)
    print(f"Сравнение способов получения контента Wikipedia")
    print(f"URL: {url}")
    print("=" * 70)
    
    # 1. Через API
    print("\n1. Загрузка через MediaWiki API (текущий способ)...")
    api_content, api_len = get_via_api(url)
    print(f"   Длина: {api_len:,} символов")
    
    # 2. Через прямую загрузку (полная страница)
    print("\n2. Загрузка через прямую загрузку HTML (полная страница)...")
    html_content, html_len = get_via_direct_html(url)
    print(f"   Длина: {html_len:,} символов")
    
    # 3. Через HTML, только основной контент
    print("\n3. Загрузка через HTML (только основной контент mw-content-text)...")
    main_content, main_len = get_via_direct_html_main_content(url)
    print(f"   Длина: {main_len:,} символов")
    
    # Сравнение
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ:")
    print("=" * 70)
    
    if api_content and main_content:
        # Проверяем, содержится ли API контент в основном контенте
        api_start = api_content[:200] if len(api_content) > 200 else api_content
        if api_start in main_content:
            print("✓ Начало API контента СОВПАДАЕТ с HTML контентом")
        else:
            print("✗ Начало API контента ОТЛИЧАЕТСЯ от HTML контента")
        
        # Показываем первые 500 символов каждого для сравнения
        print("\n--- Начало API контента (первые 500 символов): ---")
        print(api_content[:500] if api_content else "(пусто)")
        print("\n--- Начало HTML контента (первые 500 символов): ---")
        print(main_content[:500] if main_content else "(пусто)")
        
        # Разница в длине
        diff = main_len - api_len
        if diff > 0:
            print(f"\n⚠ API возвращает на {diff:,} символов МЕНЬШЕ чем HTML")
            print(f"  (API: {api_len:,} vs HTML: {main_len:,})")
            if api_len <= 1200:
                print(f"  Возможно, сработало ограничение exchars=1200 в Wikipedia API!")
        elif diff < 0:
            print(f"\n  HTML возвращает на {-diff:,} символов меньше (возможно из-за инфобокса)")
        else:
            print("\n  Длины совпадают")
    
    # Проверяем ограничение API
    print("\n" + "=" * 70)
    print("ПРИМЕЧАНИЕ: MediaWiki API prop=extracts имеет ограничение exchars=1200 символов")
    print("Без указания exchars может возвращаться усечённый контент.")
    print("=" * 70)


if __name__ == "__main__":
    main()
