from openai import OpenAI

# Инициализация клиента
# ВАЖНО: Мы подменяем base_url на наш локальный адрес.
# /v1 в конце обязательно, так требует стандарт OpenAI.
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-secret-token",  # Тот же ключ, что и при запуске сервера
)

print("Отправляю запрос к локальной модели...")

try:
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "Ты — эксперт по Linux."},
            {"role": "user", "content": "Как узнать список запущенных процессов?"}
        ],
        temperature=0.7,
    )

    # Получаем ответ
    answer = response.choices[0].message.content
    print("\n--- Ответ модели ---")
    print(answer)
    
    # Смотрим статистику токенов (vLLM её тоже возвращает!)
    print("\n--- Статистика ---")
    print(f"Tokens used: {response.usage.total_tokens}")

except Exception as e:
    print(f"Ошибка: {e}")
