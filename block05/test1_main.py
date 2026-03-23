import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Загружаем переменные из .env
load_dotenv()

# 2. Инициализируем клиента
# Указываем адрес нашего LiteLLM прокси.  
# api_key обязателен. Если он не указан - библиотека будет искать его по ключу OPENAI_API_KEY в os.environ.

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",  # CHANGED TO USER'S REQUESTED ENDPOINT
    api_key=os.getenv("OPENAI_API_KEY")
)

print(">>> Отправляем запрос к LLM...")

# 3. Формируем запрос
try:
    response = client.chat.completions.create(
        model="gpt-oss:120b",  # CHANGED TO USER'S REQUESTED MODEL
        messages=[
            {
                "role": "system",
                "content": "Ты — лаконичный научный ассистент. Отвечай максимально кратко."
            },
            {
                "role": "user",
                "content": "Объясни квантовую физику в 5 словах."
            }
        ]
    )
except Exception as e:
    print(f"Ошибка при запросе к LLM: {e}")
    exit(1)

# 4. Получаем и выводим ответ
if response.choices and response.choices[0].message and response.choices[0].message.content:
    answer = response.choices[0].message.content
    print(f"Ответ модели: {answer}")
else:
    print("Не удалось получить ответ от модели")
    exit(1)

# 5. Проверка успешного запуска
print(">>> Запрос к LLM успешно завершен")
