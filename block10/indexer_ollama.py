import os
import json
import psycopg2
from openai import OpenAI
# Импортируем функцию из прошлого примера для парсинга pdf
from chunk_pdf import process_pdf

# 1. Настройка клиента Ollama (через OpenAI совместимый API)
# Ollama работает по адресу http://localhost:11434/v1
ollama_client = OpenAI(
    api_key="dummy",  # Ollama не требует реального ключа, но параметр обязателен
    base_url="http://localhost:11434/v1"
)


def get_db_connection():
    """Создаёт подключение к PostgreSQL, использующее параметры из docker‑compose"""
    conn = psycopg2.connect(
        dbname="myvectordb",
        user="user",
        password="secretpassword",
        host="localhost",
        port=5432,
    )
    return conn


def setup_database():
    """Подготавливает БД: включает расширение pgvector и создаёт таблицу с размерностью 4096"""
    conn = get_db_connection()
    cur = conn.cursor()

    # Включаем расширение pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Создаём таблицу
    # embedding VECTOR(4096) – размерность модели qwen3‑embedding:8b‑q4_K_M
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding VECTOR(4096)
        );
        """
    )

    conn.commit()
    cur.close()
    conn.close()
    print("База данных готова (Ollama, 4096‑dim).")


# 2. Функция получения эмбеддинга от Ollama
def get_embedding(text: str) -> list:
    """Получает вектор размером 4096 от модели qwen3‑embedding:8b‑q4_K_M"""
    # Убираем переносы строк, чтобы они не влияли на вектор
    cleaned = text.replace("\n", " ")
    response = ollama_client.embeddings.create(
        input=cleaned,
        model="qwen3-embedding:8b-q4_K_M"
    )
    # Ollama возвращает список float‑ов в поле `embedding`
    return response.data[0].embedding


# 3. Основной цикл индексации
def index_chunks(chunks):
    """Итерирует чанки, получает эмбеддинги и сохраняет их в БД"""
    conn = get_db_connection()
    cur = conn.cursor()

    print(f"Начинаем индексацию {len(chunks)} чанков...")

    for i, chunk in enumerate(chunks):
        # A. Генерируем вектор
        vector = get_embedding(chunk.page_content)

        # B. Подготавливаем метаданные (JSON‑строка)
        meta_json = json.dumps(chunk.metadata)

        # C. Сохраняем в БД
        cur.execute(
            """
            INSERT INTO rag_documents (content, metadata, embedding)
            VALUES (%s, %s, %s)
            """,
            (chunk.page_content, meta_json, vector)
        )

        if (i + 1) % 5 == 0:
            print(f"Обработано {i + 1}/{len(chunks)}...")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Индексация завершена успешно!")


# --- Точка входа ---
if __name__ == "__main__":
    # 1. Создаём/обновляем схему БД
    setup_database()

    # 2. Получаем чанки из PDF (файл должен находиться в корне проекта)
    try:
        raw_chunks = process_pdf("policy.pdf")
        # 3. Запускаем процесс индексации
        index_chunks(raw_chunks)
    except Exception as e:
        print(f"Ошибка: {e}")
