import os
import json
from openai import OpenAI

# Инициализация клиента
# Если вы используете vLLM или другой совместимый сервер, укажите base_url
client = OpenAI(
    api_key="API_TOKEN",
    base_url="https://llm.rebrainme.study"
)

# 1. SYSTEM PROMPT: Задаём роль и жёсткую схему
system_instruction = """
Role: Ты — генератор синтетических данных (Faker) для тестирования Enterprise CRM.
Task: Генерируй профили пользователей на основе запроса. Данные должны выглядеть реалистично.
Output Format: ВСЕГДА отвечай валидным JSON объектом с ключом "users".
"""

# 2. FEW-SHOT: Показываем пример (One-shot)
# Это помогает модели понять стиль bio и формат UUID
example_input = "Сгенерируй одного заблокированного менеджера."
example_output = json.dumps({
    "users": [
        {
            "id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
            "full_name": "Джон Доу",
            "email": "john.doe@example-corp.com",
            "role": "MANAGER",
            "status": "SUSPENDED",
            "bio": "Бывший менеджер продаж, учётная запись приостановлена из-за нарушения политик."
        }
    ]
})

# Основной запрос
user_request = "Сгенерируй 2 активных пользователей-разработчиков из России."

try:
    print("[INFO] Генерирую данные...")

    # 3. API CALL
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": example_input},       # Пример запроса
            {"role": "assistant", "content": example_output}, # Пример ответа
            {"role": "user", "content": user_request}         # Реальная задача
        ],
        # 4. JSON MODE: важно для интеграции
        response_format={"type": "json_object"},
        temperature=0.7  # Немного вариативности для генерации разных имён
    )

    # Получаем сырой текст (строку)
    raw_content = response.choices[0].message.content

    # Парсим строку в Python-словарь
    data = json.loads(raw_content)

    # Выводим результат красиво
    print("[OK] Успешно сгенерировано:\n")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    # Пример использования данных в коде:
    print(f"\n[INFO] Первый пользователь: {data['users'][0]['email']}")

except Exception as e:
    print(f"[ERROR] Произошла ошибка: {e}")
