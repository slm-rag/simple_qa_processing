#!/bin/bash

# Скрипт для проверки прогресса обработки датасета

echo "=== Статус обработки датасета ==="
echo ""

# Проверяем, запущен ли процесс
if [ -f "download_pid.txt" ]; then
    PID=$(cat download_pid.txt)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Процесс запущен (PID: $PID)"
    else
        echo "❌ Процесс не запущен"
    fi
else
    echo "❌ Файл PID не найден"
fi

echo ""

# Показываем последние строки логов
echo "=== Последние записи в логах ==="
if [ -f "download_output.log" ]; then
    echo "--- download_output.log (последние 10 строк) ---"
    tail -10 download_output.log
else
    echo "Файл download_output.log не найден"
fi

echo ""

if [ -f "download_progress.log" ]; then
    echo "--- download_progress.log (последние 10 строк) ---"
    tail -10 download_progress.log
else
    echo "Файл download_progress.log не найден"
fi

echo ""

# Проверяем промежуточные файлы
echo "=== Промежуточные результаты ==="
if [ -f "simple_qa_test_set_with_documents.csv.temp" ]; then
    LINES=$(wc -l < simple_qa_test_set_with_documents.csv.temp)
    SIZE=$(du -h simple_qa_test_set_with_documents.csv.temp | cut -f1)
    echo "✅ Временный файл найден: $LINES строк, размер: $SIZE"
else
    echo "❌ Временный файл не найден"
fi

if [ -f "simple_qa_test_set_with_documents.csv" ]; then
    LINES=$(wc -l < simple_qa_test_set_with_documents.csv)
    SIZE=$(du -h simple_qa_test_set_with_documents.csv | cut -f1)
    echo "✅ Финальный файл найден: $LINES строк, размер: $SIZE"
else
    echo "❌ Финальный файл не найден"
fi

if [ -f "simple_qa_test_set_with_documents_report.txt" ]; then
    echo "✅ Итоговый отчет найден: simple_qa_test_set_with_documents_report.txt"
    echo "   Для просмотра отчета: cat simple_qa_test_set_with_documents_report.txt"
else
    echo "❌ Итоговый отчет не найден (обработка еще не завершена)"
fi

echo ""
echo "=== Команды для управления ==="
echo "Остановить процесс: kill \$(cat download_pid.txt)"
echo "Следить за логами: tail -f download_output.log"
echo "Проверить статус: ps aux | grep download_documents_full"
