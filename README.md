# SimpleQA Verified — пайплайн и JSONL

Репозиторий для датасета **[google/simpleqa-verified](https://huggingface.co/datasets/google/simpleqa-verified)** на Hugging Face: скачивание текстов по URL из поля `urls`, извлечение релевантных фрагментов (`long_answer`, `chunks`) и сборка JSONL для экспериментов с RAG.

---

## Установка

- **Python** 3.8+
- Зависимости:

```bash
pip install -r requirements.txt
```

- **LLM** (шаг извлечения `long_answer` с OpenRouter): задайте ключ окружения `OPENROUTER_API_KEY` или положите его в файл `.env` в корне репозитория (подхватывается `python-dotenv` в `extract_long_answer.py`).

Режим без LLM: флаг `--no-llm` у `run_pipeline.py` / `extract_long_answer.py` (BM25 и эвристики, без вызова API).

---

## Быстрый старт

Полный цикл «документы → long_answer» одной командой:

```bash
python simpleqa_verified/run_pipeline.py
```

Дальше — сборка JSONL из итогового CSV (см. ниже).

---

## Пайплайн по шагам

### 1. Документы и извлечение фрагментов

| Действие | Команда |
|----------|---------|
| Всё подряд | `python simpleqa_verified/run_pipeline.py` |
| Только скачивание | `python simpleqa_verified/download_documents_verified.py` |
| Только long_answer (если CSV с документами уже есть) | `python extract_long_answer.py` (пути по умолчанию — `simpleqa_verified/`) |

Запуск по частям вручную:

```bash
python simpleqa_verified/download_documents_verified.py \
  -o simpleqa_verified/simpleqa_verified_with_documents.csv

python extract_long_answer.py \
  -i simpleqa_verified/simpleqa_verified_with_documents.csv \
  -o simpleqa_verified/simpleqa_verified_with_long_answer.csv
```

Полезные опции `run_pipeline.py`:

| Опция | Назначение |
|--------|------------|
| `-n N` | Обработать только N строк |
| `--skip-download` | Пропустить скачивание, если CSV с документами уже готов |
| `--no-llm` | Без OpenRouter на шаге извлечения |
| `-m` / `--model` | Модель OpenRouter (по умолчанию `openai/gpt-4o`) |
| `--resume` | Продолжить запись `long_answer` с последней позиции |
| `--relaxed` | Дополнительный проход LLM при пустом ответе (см. `--help` у скриптов) |

**Артефакты** (каталог `simpleqa_verified/`, тяжёлые файлы обычно в `.gitignore`):

| Файл | Содержимое |
|------|------------|
| `simpleqa_verified_with_documents.csv` | Исходные колонки HF + `documents` (тексты по URL) |
| `simpleqa_verified_with_long_answer.csv` | Плюс `long_answer` и `chunks` (JSON) |

### 2. Скрипты сборки JSONL

**Вход для всех:** `simpleqa_verified/simpleqa_verified_with_long_answer.csv` (ключевые колонки `urls`, `chunks`, при необходимости `long_answer`, `documents`). Пути задаются через `-i` / `-o`.

**Идентификаторы:** `doc_id` и `chunk_id` считаются в `scripts/corpus_ids.py` (хэш от нормализованного URL и от пары «чанк + индекс + текст»), поэтому одна и та же страница в разных вопросах получает один и тот же `doc_id`. Префиксы и длину hex можно менять флагами (`--doc-id-prefix`, `--chunk-id-prefix`, `--hash-len` и т.д., см. `--help` у каждого скрипта).

**Запуск подряд:**

```bash
python scripts/build_qa_pairs_jsonl.py
python scripts/build_chunks_collection_jsonl.py
python scripts/build_chunk_relevance_jsonl.py
python scripts/build_documents_collection_jsonl.py
```

| Скрипт | Выход по умолчанию | Что делает |
|--------|-------------------|------------|
| `scripts/build_qa_pairs_jsonl.py` | `simpleqa_verified/qa_pairs.jsonl` | Одна строка на вопрос: `question_id`, `original_index`, `question`, **`answer` как список из одной строки** `["…"]` (удобно для единого формата «несколько эталонов»). |
| `scripts/build_chunks_collection_jsonl.py` | `simpleqa_verified/chunks_collection.jsonl` | **Одна строка на документ** (каждый URL в строке CSV): `doc_id`, `url`, `title` (для Wikipedia — из пути), `question_id`, `chunks` — массив объектов `{"id", "text"}`. |
| `scripts/build_chunk_relevance_jsonl.py` | `simpleqa_verified/chunk_relevance.jsonl` | **Одна строка на вопрос:** `question_id`, `documents` — для каждого документа вопроса список чанков **только с полями** `id` и `relevant` (`1` / `0`). Релевантность: нормализованный текст чанка совпадает с одним из фрагментов `long_answer` (как при сборке пайплайна). |
| `scripts/build_documents_collection_jsonl.py` | `simpleqa_verified/documents_collection.jsonl` | **Одна строка на вопрос:** только **релевантные** документы (есть хотя бы один чанк с меткой релевантности как выше): `question_id`, `documents` — массив `{"doc_id", "text"}` с **полным** текстом из колонки `documents` CSV. |

**Отладка нарезки** (тот же сплиттер, что в `extract_long_answer.py`):

```bash
python scripts/preview_chunks.py -i simpleqa_verified/simpleqa_verified_with_documents.csv --row 0 --doc 0
```

### 3. Статистика по CSV (опционально)

```bash
python analyze_dataset_stats.py \
  -i simpleqa_verified/simpleqa_verified_with_long_answer.csv
```

Отчёт по умолчанию рядом с входным файлом (`dataset_statistics.txt` или путь через `-s`).

---

## Структура репозитория

| Путь | Роль |
|------|------|
| `simpleqa_verified/download_documents_verified.py` | Загрузка сплита HF и вызов скачивания в CSV |
| `simpleqa_verified/run_pipeline.py` | Скачивание + `extract_long_answer` подряд |
| `download_documents_full.py` | Класс `OptimizedDocumentDownloader` (HTML, PDF, Wikipedia API, порядок URL совпадает со списком ссылок) |
| `extract_long_answer.py` | Нарезка чанков, BM25, LLM, поля `long_answer` и `chunks` |
| `scripts/corpus_ids.py` | Стабильные `doc_id` / `chunk_id` от URL и текста чанка |
| `scripts/build_qa_pairs_jsonl.py` | Вопрос–ответ в JSONL |
| `scripts/build_chunks_collection_jsonl.py` | Чанки по документам (с текстами) |
| `scripts/build_chunk_relevance_jsonl.py` | Метки релевантности чанков |
| `scripts/build_documents_collection_jsonl.py` | Полные тексты релевантных документов |
| `scripts/preview_chunks.py` | Просмотр чанков одного документа в CSV |

Колонки итогового CSV для verified включают как минимум `problem`, `answer`, `urls`, `documents`; после извлечения — `long_answer` и `chunks` (JSON-массивы).

---

## Переменные окружения

| Переменная | Где используется |
|------------|------------------|
| `OPENROUTER_API_KEY` | Вызовы OpenRouter в `extract_long_answer.py` |
| `OPENROUTER_REQUEST_TIMEOUT` | Таймаут HTTP (секунды), по умолчанию 180 |
