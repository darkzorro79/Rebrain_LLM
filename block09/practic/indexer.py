#!/usr/bin/env python3
"""
Скрипт индексации для системы техподдержки.
Подключается к PostgreSQL и индексирует 5 записей базы знаний.
Использует модель gemini-embedding-001 для генерации эмбеддингов.
"""

import os
import sys
import logging
import requests
import json
import psycopg2
from psycopg2.extras import execute_values

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# База знаний техподдержки
KNOWLEDGE_BASE = [
    "Чтобы восстановить доступ к корпоративной учётной записи, воспользуйтесь порталом self-service или отправьте запрос на сброс пароля через SMS-верификацию в приложении Authenticator.",
    "Инструкция по подключению к удалённому рабочему столу: используйте клиент AnyConnect, адрес сервера https://www.google.com/search?q=vpn.company.com и выберите протокол TLS. Убедитесь, что установлены актуальные корневые сертификаты.",
    "Гостевой Wi-Fi доступен в зоне кофе-поинта. Имя сети: Guest_Free_Internet. Пароль меняется каждый понедельник, актуальный код можно найти на доске объявлений у ресепшена.",
    "Если принтер на втором этаже выдаёт ошибку 'Paper Jam' или 'Missing Toner', проверьте лоток №2 или обратитесь в сервисный отдел через тикет-систему с пометкой 'Hardware'.",
    "Официальные часы работы офиса: с 10:00 до 19:00. Заявки на ежегодный оплачиваемый отпуск должны быть поданы в системе HR-Portal не менее чем за две недели до предполагаемой даты начала."
]

# Конфигурация
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'support_db'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'secretpassword')
}

API_KEY = os.getenv('API_KEY')
API_URL = 'https://llm.rebrainme.study/v1/embeddings'

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
            API_URL,
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        embedding = result['data'][0]['embedding']
        
        logger.info(f"Получен эмбеддинг размерности {len(embedding)}")
        return embedding
        
    except requests.RequestException as e:
        logger.error(f"Ошибка при запросе к API: {e}")
        sys.exit(1)
    except (KeyError, ValueError) as e:
        logger.error(f"Ошибка парсинга ответа API: {e}")
        sys.exit(1)

def connect_to_db():
    """Подключение к базе данных PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Успешное подключение к базе данных")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        sys.exit(1)

def check_table_exists(conn):
    """Проверяет существование таблицы knowledge_base."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'knowledge_base'
                );
            """)
            exists = cur.fetchone()[0]
            if exists:
                logger.info("Таблица knowledge_base уже существует")
            else:
                logger.error("Таблица knowledge_base не найдена. Убедитесь, что база данных инициализирована.")
                sys.exit(1)
    except psycopg2.Error as e:
        logger.error(f"Ошибка проверки таблицы: {e}")
        sys.exit(1)

def check_existing_data(conn):
    """Проверяет наличие данных в таблице."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_base;")
            count = cur.fetchone()[0]
            
            if count > 0:
                logger.info(f"В таблице уже есть {count} записей")
                # Для автоматического режима - не перезаписываем данные
                logger.info("Пропускаем индексацию, данные уже существуют")
                return False
            else:
                logger.info("Таблица пуста, начинаем индексацию")
                return True
                
    except psycopg2.Error as e:
        logger.error(f"Ошибка проверки данных: {e}")
        sys.exit(1)

def index_documents(conn):
    """Индексирует документы базы знаний."""
    logger.info("Начало индексации документов...")
    
    indexed_data = []
    
    for i, document in enumerate(KNOWLEDGE_BASE, 1):
        logger.info(f"Обработка документа {i}/5...")
        
        # Получаем эмбеддинг
        embedding = get_embedding(document)
        
        # Проверяем размерность эмбеддинга
        if len(embedding) != 3072:
            logger.error(f"Неожиданная размерность эмбеддинга: {len(embedding)}, ожидается 3072")
            sys.exit(1)
        
        indexed_data.append((document, embedding))
        logger.info(f"Документ {i} готов к сохранению")
    
    # Сохраняем все документы в базу
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO knowledge_base (content, embedding) VALUES %s",
                indexed_data,
                template=None
            )
        conn.commit()
        logger.info("Все документы успешно сохранены в базу данных")
        
        # Проверяем количество записей
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_base;")
            count = cur.fetchone()[0]
            logger.info(f"В базе данных находится {count} записей")
            
    except psycopg2.Error as e:
        logger.error(f"Ошибка сохранения в базу данных: {e}")
        sys.exit(1)

def main():
    """Основная функция."""
    logger.info("Запуск скрипта индексации...")
    
    # Подключение к базе данных
    conn = connect_to_db()
    
    try:
        # Проверка существования таблицы
        check_table_exists(conn)
        
        # Проверка существующих данных
        if check_existing_data(conn):
            # Индексация документов только если таблица пуста
            index_documents(conn)
            logger.info("Индексация завершена успешно!")
        else:
            logger.info("Индексация пропущена - данные уже существуют")
        
    finally:
        conn.close()
        logger.info("Соединение с базой данных закрыто")

if __name__ == "__main__":
    main()