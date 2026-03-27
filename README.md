# Simple QA Dataset Processor

Инструмент для обогащения датасета вопросов-ответов (QA) путем автоматического скачивания и извлечения текста из документов по URL из метаданных.

## Описание

Проект обрабатывает CSV датасет с вопросами и ответами, извлекает URL документов из поля `metadata`, скачивает эти документы и добавляет их текстовое содержимое в новое поле `documents`. Поддерживаются форматы: HTML, PDF и текстовые файлы.

### Датасеты

- **Исходный датасет**: `simple_qa_test_set.csv` - входной датасет с вопросами, ответами и метаданными (URL документов)
- **С документами**: `simple_qa_test_set_with_documents.csv` - обогащенный датасет с полем `documents` (тексты скачанных документов)
- **С long_answer**: `simple_qa_test_set_with_long_answer.csv` - датасет с полем `long_answer` (список релевантных фрагментов из документов)

### google/simpleqa-verified (HuggingFace)

Скрипты для датасета [google/simpleqa-verified](https://huggingface.co/datasets/google/simpleqa-verified) находятся в папке `simpleqa_verified/`:

- `simpleqa_verified/download_documents_verified.py` — загрузка датасета и скачивание документов
- `simpleqa_verified/run_pipeline.py` — полный пайплайн (документы + long_answer)

## Возможности

- ✅ Автоматическое скачивание документов по URL
- ✅ Поддержка форматов: HTML, PDF, текстовые файлы
- ✅ Параллельная обработка (до 5 потоков)
- ✅ Кэширование документов для избежания повторных запросов
- ✅ Извлечение релевантных фрагментов (long_answer) из документов для каждого вопроса
- ✅ Прогресс-бар с отображением статуса обработки
- ✅ Промежуточное сохранение результатов каждые 100 строк
- ✅ Детальное логирование процесса
- ✅ Генерация итогового отчета
- ✅ Корректная обработка сигналов завершения

## Требования

- Python 3.8+
- Для скачивания документов: `pandas`, `requests`, `beautifulsoup4`, `PyPDF2`, `tqdm`
- Для извлечения long_answer: `rank-bm25`, OpenRouter API (переменная `OPENROUTER_API_KEY`)

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd simple_qa
```

2. Создайте виртуальное окружение и установите зависимости:
```bash
python -m venv env
# Windows (PowerShell):
.\env\Scripts\Activate.ps1
# Windows (cmd):
env\Scripts\activate.bat
# Linux/Mac:
source env/bin/activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

(включает зависимости для скачивания документов и извлечения long_answer)

## Структура входного датасета

Входной CSV файл должен содержать следующие колонки:

- `metadata` - JSON строка с метаданными, включая поле `urls` со списком URL документов
- `problem` - вопрос
- `answer` - ответ

Пример строки:
```csv
metadata,problem,answer
"{'topic': 'Science', 'urls': ['https://example.com/doc1.pdf', 'https://example.com/doc2.html']}",What is the topic?,Science
```

### Базовое использование

```bash
python download_documents_full.py
```

По умолчанию скрипт обрабатывает:
- Входной файл: `simple_qa_test_set.csv`
- Выходной файл: `simple_qa_test_set_with_documents.csv`

### Запуск в фоновом режиме

Для длительной обработки больших датасетов используйте скрипт запуска в фоне:

```bash
./run_background.sh
```

Это запустит процесс в фоне и сохранит его PID в `download_pid.txt`.

### Проверка прогресса

Для проверки статуса обработки используйте:

```bash
./check_progress.sh
```

Скрипт покажет:
- Статус процесса (запущен/остановлен)
- Последние записи из логов
- Размер промежуточных и финальных файлов

### Настройка параметров

Вы можете изменить параметры в скрипте `download_documents_full.py`:

```python
# Параметры скачивания
timeout=15          # Таймаут запроса в секундах
delay=0.1          # Задержка между запросами в секундах
max_workers=5      # Количество параллельных потоков

# Пути к файлам
input_file = "/path/to/input.csv"
output_file = "/path/to/output.csv"
```

### Обработка google/simpleqa-verified

```bash
# Полный пайплайн (скачивание + long_answer)
python simpleqa_verified/run_pipeline.py

# Только скачивание документов
python simpleqa_verified/download_documents_verified.py

# С ограничением (например, 10 строк для теста)
python simpleqa_verified/run_pipeline.py -n 10

# Без LLM при извлечении long_answer (быстрее)
python simpleqa_verified/run_pipeline.py --no-llm
```

Результаты сохраняются в `simpleqa_verified/`:
- `simpleqa_verified_with_documents.csv` — документы
- `simpleqa_verified_with_long_answer.csv` — фрагменты long_answer

### Извлечение long_answer

Скрипт `extract_long_answer.py` извлекает релевантные фрагменты из документов для каждого вопроса. Требует датасет с полем `documents` (результат `download_documents_full.py` или `download_documents_verified.py`).

**Логика:**
- Для документа с совпадением ответа — извлечение окна/абзаца вокруг позиции ответа
- Для остальных документов — разбивка на сегменты, BM25-ранжирование, выбор лучшего чанка через LLM (OpenRouter, по умолчанию `openai/gpt-4o`)
- Результат — список уникальных фрагментов (дубликаты после нормализации удаляются)

```bash
# Требуется OPENROUTER_API_KEY (получить на https://openrouter.ai)
export OPENROUTER_API_KEY=sk-or-...
python extract_long_answer.py -i simple_qa_test_set_with_documents.csv -o simple_qa_test_set_with_long_answer.csv
```

Опции:
- `--no-llm` — без LLM (только BM25, первый чанк)
- `--model` — модель OpenRouter (по умолчанию: openai/gpt-4o)
- `-n N` — обработать только N строк

Поле `long_answer` сохраняется как JSON-список строк: `["фрагмент 1", "фрагмент 2"]`.

## Результаты

После завершения обработки создаются следующие файлы:

1. **`simple_qa_test_set_with_documents.csv`** — результат скачивания с полем `documents`
2. **`simple_qa_test_set_with_documents_report.txt`** — отчет со статистикой скачивания
3. **`simple_qa_test_set_with_long_answer.csv`** — результат извлечения фрагментов с полем `long_answer`
4. **`download_progress.log`** — лог процесса скачивания

### Структура выходных файлов

**simple_qa_test_set_with_documents.csv** — все исходные колонки плюс:
- `documents` — список текстов скачанных документов (JSON массив строк)

**simple_qa_test_set_with_long_answer.csv** — все колонки из входного CSV плюс:
- `long_answer` — список релевантных фрагментов из документов (JSON массив строк, без дубликатов)

## Статистика обработки

Отчет включает:
- Общее количество обработанных строк
- Количество успешно обработанных строк
- Количество неудачных загрузок
- Статистику HTTP запросов
- Эффективность кэширования

## Особенности реализации

- **Параллельная обработка**: Используется `ThreadPoolExecutor` для параллельного скачивания документов
- **Кэширование**: Документы кэшируются в памяти для избежания повторных запросов к одинаковым URL
- **Устойчивость**: Промежуточные результаты сохраняются каждые 100 строк, что позволяет продолжить обработку при сбое
- **Обработка ошибок**: Все ошибки логируются, но не останавливают процесс обработки

### Мониторинг процесса

```bash
# Проверить статус
./check_progress.sh

# Следить за логами в реальном времени
tail -f download_progress.log

# Остановить процесс
kill $(cat download_pid.txt)
```

## Устранение неполадок

### Проблема: Медленная обработка
- Увеличьте `max_workers` для большего количества параллельных потоков
- Уменьшите `delay` между запросами (осторожно, чтобы не перегрузить серверы)

### Проблема: Много неудачных загрузок
- Проверьте доступность URL в датасете
- Увеличьте `timeout` для медленных серверов
- Проверьте логи на наличие специфических ошибок

### Проблема: Процесс прервался
- Промежуточные результаты сохраняются в `*.csv.temp`
- Можно восстановить данные из временного файла
- Перезапустите скрипт - он продолжит с места остановки (если реализовано) или начнет заново

