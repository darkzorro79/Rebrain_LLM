#!/usr/bin/env python3
"""
Log Parser Client
Загружает логи и парсит их с помощью LLM API
"""

import os
import sys
import requests
import json
from openai import OpenAI


def download_logs(url):
    """Скачивает файл логов по URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Ошибка при скачивании логов: {e}", file=sys.stderr)
        sys.exit(1)


def parse_logs_with_llm(logs_content, api_key, api_url, model):
    """Парсит логи с помощью LLM"""
    # Инициализация клиента OpenAI с кастомным API URL
    client = OpenAI(
        api_key=api_key,
        base_url=api_url
    )
    
    # System prompt задает роль Log Parser
    system_prompt = """Ты - Log Parser. Твоя задача - анализировать логи и структурировать их в JSON формат.
Извлекай следующую информацию:
- service_name: название сервиса из лога
- error_code: числовой код ошибки (0 если отсутствует)
- severity: уровень критичности (CRITICAL, WARNING, INFO)
- message: очищенный текст сообщения

Правила определения severity:
- CRITICAL: системные сбои, аварии, недоступность сервисов
- WARNING: предупреждения, потенциальные проблемы
- INFO: информационные сообщения, обычные события"""

    # Few-Shot Prompting - пример преобразования нестандартного лога
    few_shot_example = """Пример:
Входной лог: "SYSTEM DOWN - Database connection lost"
Выходной JSON:
{
  "entries": [
    {
      "service_name": "system",
      "error_code": 0,
      "severity": "CRITICAL",
      "message": "Database connection lost"
    }
  ]
}"""
    
    # Формируем сообщения для API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_example},
        {"role": "assistant", "content": '{"entries": [{"service_name": "system", "error_code": 0, "severity": "CRITICAL", "message": "Database connection lost"}]}'},
        {"role": "user", "content": f"Проанализируй следующие логи и верни результат в JSON формате:\n\n{logs_content}"}
    ]
    
    try:
        # Отправляем запрос к LLM с форматированным выходом
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ошибка при обращении к LLM API: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Основная функция"""
    # Получаем API ключ из переменной окружения
    api_key = os.getenv('REBRAIN_API_KEY')
    if not api_key:
        print("Ошибка: переменная окружения REBRAIN_API_KEY не установлена", file=sys.stderr)
        sys.exit(1)
    
    # Параметры API
    api_url = "https://llm.rebrainme.study"
    model = "gemini-2.5-flash"
    logs_url = "http://91.219.24.207:3333/logs.txt"
    
    # Скачиваем логи
    print("Скачиваем логи...", file=sys.stderr)
    logs_content = download_logs(logs_url)
    
    # Парсим логи с помощью LLM
    print("Анализируем логи с помощью LLM...", file=sys.stderr)
    result = parse_logs_with_llm(logs_content, api_key, api_url, model)
    
    # Выводим результат на stdout
    print(result)


if __name__ == "__main__":
    main()