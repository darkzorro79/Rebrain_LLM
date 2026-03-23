import os
import sys
import argparse
from dotenv import load_dotenv
from openai import OpenAI

def main():
    # Загружаем переменные из .env
    load_dotenv()
    
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description='Отправка промпта к LLM модели')
    parser.add_argument('prompt', type=str, help='Текстовый промпт для модели')
    args = parser.parse_args()
    
    # Получаем API-ключ из переменной окружения
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Ошибка: переменная окружения OPENAI_API_KEY не установлена", file=sys.stderr)
        sys.exit(1)
    
    # Инициализируем клиента OpenAI с указанием base_url
    client = OpenAI(
        base_url="https://llm.rebrainme.study",
        api_key=api_key
    )
    
    # Отправляем запрос к модели
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": args.prompt
            }
        ]
    )
    
    # Выводим только текст ответа модели
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
