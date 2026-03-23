#!/usr/bin/env python3
"""
Тестирование API сервера пиратской модели
Запускается после run_server.sh
"""

import requests
import json

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "pirate-qwen-7b"

def test_chat(user_message: str):
    """Отправка запроса к API"""
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False
    }
    
    print(f"\n📝 Запрос: {user_message}")
    print("-"*60)
    
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"]
        
        print(f"🏴‍☠️ Ответ: {assistant_message}")
        print("-"*60)
        
        return assistant_message
        
    except requests.exceptions.ConnectionError:
        print("❌ Ошибка: Не удалось подключиться к серверу")
        print("   Убедитесь, что сервер запущен: ./run_server.sh")
        return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


if __name__ == "__main__":
    print("="*60)
    print("🏴‍☠️ Тестирование API Пиратской Модели")
    print("="*60)
    print(f"API URL: {API_URL}")
    print(f"Модель: {MODEL_NAME}")
    print("="*60)
    
    # Тестовые запросы
    test_prompts = [
        "Как дела?",
        "Расскажи мне о себе",
        "Что такое искусственный интеллект?",
    ]
    
    for prompt in test_prompts:
        test_chat(prompt)
    
    print("\n" + "="*60)
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("Введите 'exit' для выхода")
    print("="*60 + "\n")
    
    # Интерактивный режим
    while True:
        try:
            user_input = input("Вы: ").strip()
            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("👋 До свидания!")
                break
            
            if not user_input:
                continue
            
            response = test_chat(user_input)
            if response is None:
                break
                
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
