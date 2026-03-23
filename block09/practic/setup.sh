#!/bin/bash

# Скрипт быстрого развертывания RAG системы техподдержки

echo "=== Развертывание RAG системы техподдержки ==="

# Проверяем наличие API ключа
if [ -z "$API_KEY" ]; then
    echo "⚠️  Ошибка: Переменная окружения API_KEY не установлена!"
    echo "Установите её командой:"
    echo 'export API_KEY="sk-FbZmrDQdnK1HpKSKXo97Rg"'
    echo ""
    echo "Или создайте .env файл:"
    echo "cp .env.example .env"
    echo "source .env"
    exit 1
fi

echo "✅ API_KEY установлен"

# Проверяем наличие Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не найден. Установите Docker и попробуйте снова."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не найден. Установите Docker Compose и попробуйте снова."
    exit 1
fi

echo "✅ Docker найден"

# Устанавливаем Python зависимости
echo "📦 Устанавливаем Python зависимости..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Ошибка установки зависимостей"
    exit 1
fi

echo "✅ Зависимости установлены"

# Запускаем базу данных
echo "🐳 Запускаем PostgreSQL с pgvector..."
cd base && docker-compose up -d

if [ $? -ne 0 ]; then
    echo "❌ Ошибка запуска базы данных"
    exit 1
fi

cd ..

# Ждем запуска базы данных
echo "⏳ Ждем запуска базы данных (10 секунд)..."
sleep 10

# Запускаем индексацию
echo "🔍 Запускаем индексацию базы знаний..."
python indexer.py

if [ $? -ne 0 ]; then
    echo "❌ Ошибка индексации"
    exit 1
fi

echo ""
echo "🎉 Развертывание завершено успешно!"
echo ""
echo "Теперь можно задавать вопросы:"
echo 'python rag_chat.py "Как восстановить пароль?"'
echo 'python rag_chat.py "Проблемы с принтером"'
echo 'python rag_chat.py "Часы работы офиса"'