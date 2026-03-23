import os
import json
import psycopg2
from openai import OpenAI
# Импортируем функцию из прошлого примера для парсинга pdf
from chunk_pdf import process_pdf

# 1. Настройка клиентов
openai_client = OpenAI(
    api_key="API_KEY", # Для локального сервера можно вписать любой текст
    base_url="https://llm.rebrainme.study/" # Либо адрес вашего провайдера/локального хоста
)

def get_db_connection():
    """Создаёт подключение к PostgreSQL"""
    conn = psycopg2.connect(
        dbname="app",
        user="app",
        password="password",
        host="localhost"
    )
    return conn

def setup_database():
    """
    Подготавливает БД: включает расширение и создаёт таблицу.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Включаем расширение pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Создаём таблицу
    # embedding vector(3072) -> 3072 это размерность модели gemini-embedding-001
    # metadata JSONB -> позволяет гибко хранить номер страницы, имя файла и т.д.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rag_documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding VECTOR(3072)
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("База данных готова.")

# 2. Функция получения эмбеддинга
def get_embedding(text: str) -> list:
    """
    Превращает текст в вектор (список из 3072 чисел).
    """
    # Убираем переносы строк, так как они могут влиять на вектор (best practice)
    text = text.replace("\n", " ")

    response = openai_client.embeddings.create(
        input=text,
        model="gemini-embedding-001"
    )
    return response.data[0].embedding

# 3. Основной цикл индексации
def index_chunks(chunks):
    """
    Принимает список чанков (из прошлого шага), векторизует и сохраняет в БД.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    print(f"Начинаем индексацию {len(chunks)} чанков...")

    for i, chunk in enumerate(chunks):
        # А. Генерируем вектор
        vector = get_embedding(chunk.page_content)

        # Б. Подготавливаем метаданные (превращаем dict в json строку)
        meta_json = json.dumps(chunk.metadata)

        # В. Сохраняем в БД
        cur.execute(
            """
            INSERT INTO rag_documents (content, metadata, embedding)
            VALUES (%s, %s, %s)
            """,
            (chunk.page_content, meta_json, vector)
        )

        # Визуализация прогресса
        if (i + 1) % 5 == 0:
            print(f"Обработано {i + 1}/{len(chunks)}...")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Индексация завершена успешно!")

# --- Точка входа ---
if __name__ == "__main__":
    # 1. Сначала создаём таблицу
    setup_database()

    # 3. Получаем чанки (убедитесь, что файл sample.pdf существует)
    try:
        raw_chunks = process_pdf("policy.pdf")

        # 4. Запускаем индексацию
        index_chunks(raw_chunks)

    except Exception as e:
        print(f"Ошибка: {e}")
