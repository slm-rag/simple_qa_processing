#!/bin/bash

# Скрипт для запуска обработки датасета в фоне
# Работает даже после закрытия терминала

echo "Запуск обработки датасета в фоне..."

# Активируем виртуальное окружение и запускаем скрипт
source venv/bin/activate

# Запускаем с nohup для работы в фоне
nohup python download_documents_full.py > download_output.log 2>&1 &

# Получаем PID процесса
PID=$!

echo "Процесс запущен с PID: $PID"
echo "Логи сохраняются в: download_output.log"
echo "Прогресс также сохраняется в: download_progress.log"
echo ""
echo "Для проверки статуса используйте:"
echo "  ps aux | grep $PID"
echo "  tail -f download_output.log"
echo "  tail -f download_progress.log"
echo ""
echo "Для остановки процесса используйте:"
echo "  kill $PID"
echo ""
echo "Промежуточные результаты сохраняются в:"
echo "  simple_qa_test_set_with_documents.csv.temp"
echo ""
echo "После завершения будет создан отчет:"
echo "  simple_qa_test_set_with_documents_report.txt"

# Сохраняем PID в файл
echo $PID > download_pid.txt
echo "PID сохранен в download_pid.txt"
