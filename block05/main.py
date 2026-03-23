import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Загружаем переменные из .env
load_dotenv()

# 2. Инициализируем клиента
# Указываем адрес нашего LiteLLM прокси.
# api_key обязателен. Если он не указан - библиотека 
# будет искать его по ключу OPENAI_API_KEY в os.environ.
client = OpenAI(
    base_url="https://llm.rebrainme.study",
    api_key=os.getenv("OPENAI_API_KEY")
)
print(">>> Отправляем запрос к LLM...")

# 3. Формируем запрос
response = client.chat.completions.create(
    model="gemini-2.5-flash",
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

# 4. Получаем и выводим ответ
# Ответ приходит в сложной структуре, нам нужен только текст сообщения
answer = response.choices[0].message.content

print(f"\nОтвет модели:\n{answer}")