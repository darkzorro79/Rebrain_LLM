#!/usr/bin/env python3
"""Проверка конфигурации системы RAG"""

import os
import sys

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env файл загружен")
except ImportError:
    print("⚠️  python-dotenv не установлен, используются системные переменные")

print("\n=== Проверка переменных окружения ===\n")

# Check required variables
required_vars = {
    "OPENAI_API_KEY": "Ключ для OpenAI-совместимого API",
    "DB_DSN": "Строка подключения к PostgreSQL"
}

optional_vars = {
    "OPENAI_BASE_URL": ("URL OpenAI-совместимого API", "https://llm.rebrainme.study"),
    "GOOGLE_API_KEY": ("Google API Key (опционально)", None)
}

all_ok = True

for var, description in required_vars.items():
    value = os.getenv(var)
    if value:
        # Show partial value for security
        if "KEY" in var or "PASSWORD" in var:
            display = value[:10] + "..." if len(value) > 10 else value
        else:
            display = value
        print(f"✅ {var}: {display}")
        print(f"   {description}")
    else:
        print(f"❌ {var}: НЕ УСТАНОВЛЕН")
        print(f"   {description}")
        all_ok = False

print()

for var, (description, default) in optional_vars.items():
    value = os.getenv(var, default)
    if value:
        if "KEY" in var:
            display = value[:10] + "..." if len(value) > 10 else value
        else:
            display = value
        print(f"ℹ️  {var}: {display}")
        print(f"   {description}")
    else:
        print(f"⚠️  {var}: не установлен (опционально)")
        print(f"   {description}")

print("\n=== Проверка зависимостей ===\n")

dependencies = [
    ("psycopg2", "PostgreSQL adapter"),
    ("openai", "OpenAI API client"),
    ("google.generativeai", "Google Generative AI (для совместимости)"),
    ("langchain_text_splitters", "Text splitting utilities"),
    ("requests", "HTTP library")
]

for module, description in dependencies:
    try:
        __import__(module)
        print(f"✅ {module}: установлен")
        print(f"   {description}")
    except ImportError:
        print(f"❌ {module}: НЕ УСТАНОВЛЕН")
        print(f"   {description}")
        all_ok = False

print("\n=== Резюме ===\n")

if all_ok:
    print("🎉 Все обязательные компоненты настроены правильно!")
    print("\nМожно запускать:")
    print("  python3 indexer.py  # Индексация")
    print("  python3 rag.py 'Ваш вопрос'  # Поиск")
else:
    print("❌ Некоторые компоненты не настроены.")
    print("\nДля установки зависимостей:")
    print("  pip install -r requirements.txt")
    print("\nДля настройки переменных окружения:")
    print("  cp .env.example .env")
    print("  # Отредактируйте .env файл")

sys.exit(0 if all_ok else 1)