import os
from dotenv import load_dotenv
from litellm import completion  # Используем LiteLLM для обращения к Ollama

load_dotenv()

# Список моделей Ollama, которые будем тестировать (префикс ollama/)
models = [
    "ollama/gpt-oss:120b",  # та же модель, что в test2_main.py
    "ollama/qwen2.5:32b",   # дополнительная модель
]

# Сообщения-запросы - те же, что в universal_client.py
messages = [
    {"role": "user", "content": "Кто ты? Ответь одним словом."}
]

print(">>> Начинаем тестирование моделей Ollama через LiteLLM...\n")

for model_name in models:
    try:
        print(f"Запрос к модели: {model_name}")
        
        # МАГИЯ ЗДЕСЬ:
        # Используем litellm для подключения к локальному Ollama
        response = completion(
            model=model_name,
            messages=messages,
            api_base="http://127.0.0.1:11434"  # БЕЗ /v1 - litellm сам добавит нужные пути
        )
        
        # Формат ответа ВСЕГДА одинаковый (как у OpenAI)
        answer = response.choices[0].message.content
        print(f"Ответ: {answer}\n")
        
    except Exception as e:
        print(f"Ошибка при вызове {model_name}: {e}\n")
