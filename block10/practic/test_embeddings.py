#!/usr/bin/env python3
"""Тестовый скрипт для проверки генерации 3072-мерных эмбеддингов"""

import os
import sys

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("❌ Необходимо установить openai: pip install openai")
    sys.exit(1)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://llm.rebrainme.study")

if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY не установлен")
    sys.exit(1)

# Initialize client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

def test_embedding():
    """Тест генерации 3072-мерного эмбеддинга с gemini-embedding-001"""
    test_text = "Kubernetes API changes in v1.35"
    
    try:
        print(f"🧪 Тестируем эмбеддинг для текста: '{test_text}'")
        print(f"📝 Используем модель: gemini-embedding-001")
        
        response = client.embeddings.create(
            model="gemini-embedding-001",
            input=test_text
        )
        
        embedding = response.data[0].embedding
        
        print(f"✅ Эмбеддинг создан!")
        print(f"📏 Размерность: {len(embedding)}")
        print(f"📊 Первые 5 значений: {embedding[:5]}")
        print(f"📊 Последние 5 значений: {embedding[-5:]}")
        
        if len(embedding) == 3072:
            print("🎉 Тест ПРОЙДЕН! Размерность корректна: 3072")
            print("✅ Модель: gemini-embedding-001")
            return True
        else:
            print(f"❌ Тест ПРОВАЛЕН! Ожидалось 3072, получено {len(embedding)}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при генерации эмбеддинга: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Тестирование 3072-мерных эмбеддингов ===")
    success = test_embedding()
    sys.exit(0 if success else 1)