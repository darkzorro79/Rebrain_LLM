#!/usr/bin/env python3
"""
Скрипт поиска и ответа для системы техподдержки.
Использует RAG (Retrieval-Augmented Generation) для ответов на вопросы пользователей.
"""

import os
import sys
import argparse
import logging
import requests
import json
import psycopg2

# Настройка логирования (только для ошибок, чтобы не засорять stdout)
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Конфигурация базы данных
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'support_db'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'secretpassword')
}

# API конфигурация
API_KEY = os.getenv('API_KEY')
API_BASE_URL = 'https://llm.rebrainme.study/v1'
EMBEDDING_URL = f'{API_BASE_URL}/embeddings'
CHAT_URL = f'{API_BASE_URL}/chat/completions'

def get_embedding(text):
    """Получает эмбеддинг текста через OpenRouter прокси API."""
    if not API_KEY:
        logger.error("API_KEY не установлен")
        sys.exit(1)
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    data = {
        'model': 'google/gemini-pro',
        'input': text
    }
    
    try:
        response = requests.post(
            EMBEDDING_URL,
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        embedding = result['data'][0]['embedding']
        return embedding
        
    except requests.RequestException as e:
        logger.error(f"Ошибка при запросе эмбеддинга: {e}")
        sys.exit(1)
    except (KeyError, ValueError) as e:
        logger.error(f"Ошибка парсинга ответа API: {e}")
        sys.exit(1)

def connect_to_db():
    """Подключение к базе данных PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        sys.exit(1)

def find_most_relevant_context(conn, question_embedding):
    """Находит наиболее релевантный контекст в базе данных."""
    try:
        with conn.cursor() as cur:
            # Используем оператор <-> для поиска по косинусному расстоянию
            cur.execute("""
                SELECT content, embedding <-> %s::vector AS distance
                FROM knowledge_base
                ORDER BY distance ASC
                LIMIT 1;
            """, (question_embedding,))
            
            result = cur.fetchone()
            
            if result:
                return result[0]  # возвращаем только текст, не расстояние
            else:
                logger.error("Нет данных в базе знаний")
                sys.exit(1)
                
    except psycopg2.Error as e:
        logger.error(f"Ошибка поиска в базе данных: {e}")
        sys.exit(1)

def generate_answer(context, question):
    """Генерирует ответ используя LLM с найденным контекстом."""
    if not API_KEY:
        logger.error("API_KEY не установлен")
        sys.exit(1)
    
    prompt = f"Используя данный текст: {context}, ответь на вопрос: {question}"
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    data = {
        'model': 'google/gemini-pro',
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'max_tokens': 1000,
        'temperature': 0.7
    }
    
    try:
        response = requests.post(
            CHAT_URL,
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Извлекаем текст ответа в формате OpenAI API
        if 'choices' in result and result['choices']:
            choice = result['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content'].strip()
        
        logger.error("Неожиданный формат ответа от LLM")
        sys.exit(1)
        
    except requests.RequestException as e:
        logger.error(f"Ошибка при запросе к LLM: {e}")
        sys.exit(1)
    except (KeyError, ValueError) as e:
        logger.error(f"Ошибка парсинга ответа LLM: {e}")
        sys.exit(1)

def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description='RAG система ответов техподдержки',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python rag_chat.py "Как восстановить пароль?"
  python rag_chat.py "Проблемы с принтером"
  python rag_chat.py "Часы работы офиса"
        """
    )
    
    parser.add_argument(
        'question',
        help='Вопрос пользователя'
    )
    
    args = parser.parse_args()
    question = args.question.strip()
    
    if not question:
        logger.error("Вопрос не может быть пустым")
        sys.exit(1)
    
    # Подключение к базе данных
    conn = connect_to_db()
    
    try:
        # Проверяем наличие данных в базе
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_base;")
            count = cur.fetchone()[0]
            
            if count == 0:
                logger.error("База знаний пуста. Запустите indexer.py для индексации данных.")
                sys.exit(1)
        
        # Генерируем эмбеддинг вопроса
        question_embedding = get_embedding(question)
        
        # Ищем наиболее релевантный контекст
        context = find_most_relevant_context(conn, question_embedding)
        
        # Генерируем ответ
        answer = generate_answer(context, question)
        
        # Выводим ответ в stdout
        print(answer)
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()