import os
from dotenv import load_dotenv
from litellm import completion  # Импортируем универсальную функцию

load_dotenv()

# Список моделей для теста
models = [
    "gpt-5-mini",     # OpenAI
    "claude-4.5-opus" # Anthropic
]

messages = [
    {"role": "user", "content": "Кто ты? Ответь одним словом."}
]

print(">>> Начинаем тестирование моделей через LiteLLM...\n")

for model_name in models:
    try:
        print(f"Запрос к модели: {model_name}")
        
        # МАГИЯ ЗДЕСЬ:
        # Мы используем одну и ту же функцию completion для разных провайдеров.
        # LiteLLM сама найдёт нужный ключ в env (ANTHROPIC_API_KEY или OPENAI_API_KEY)
        response = completion(
            model=model_name,
            messages=messages
        )
        
        # Формат ответа ВСЕГДА одинаковый (как у OpenAI)
        answer = response.choices[0].message.content
        print(f"Ответ: {answer}\n")
        
    except Exception as e:
        print(f"Ошибка при вызове {model_name}: {e}")

