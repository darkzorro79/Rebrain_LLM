#!/bin/bash

# Скрипт для запуска системы семантического поиска по Kubernetes CHANGELOG

echo "=== Установка зависимостей ==="
pip install -r requirements.txt

echo "=== Проверка переменных окружения ==="
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ GOOGLE_API_KEY не установлен"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY не установлен"
    exit 1
fi

if [ -z "$DB_DSN" ]; then
    echo "❌ DB_DSN не установлен"
    exit 1
fi

echo "✅ Все переменные окружения настроены"

echo "=== Запуск индексации ==="
python3 indexer.py

if [ $? -eq 0 ]; then
    echo "✅ Индексация завершена успешно"
    echo ""
    echo "Теперь можно задавать вопросы:"
    echo "python3 rag.py \"Что нового в работе с API?\""
    echo "python3 rag.py \"Какие изменения в SIG Instrumentation?\""
else
    echo "❌ Ошибка индексации"
    exit 1
fi