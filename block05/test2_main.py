import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)
print(">>> Отправляем запрос к LLM...")

response = client.chat.completions.create(
    model="gpt-oss:120b",
    messages=[
        {"role": "system", "content": "Ты — лаконичный научный ассистент."},
        {"role": "user", "content": "Объясни квантовую физику в 5 словах."}
    ]
)

# 1. Извлекаем ответ
content = response.choices[0].message.content

# 2. Извлекаем метаданные
usage = response.usage
if usage is None:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
else:
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
finish_reason = response.choices[0].finish_reason

print("-" * 30)
print(f"Ответ модели:\n{content}")
print("-" * 30)

# 3. Выводим техническую сводку
print(f"Причина остановки: {finish_reason}")
print(f"Токены промпта (Input):  {prompt_tokens}")
print(f"Токены ответа (Output):  {completion_tokens}")
print(f"Всего токенов:           {usage.total_tokens}")
