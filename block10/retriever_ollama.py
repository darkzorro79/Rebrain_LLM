import psycopg2
# Импортируем функции из файла indexer_ollama, который уже настроен под Ollama и pgvector
from indexer_ollama import get_db_connection, get_embedding


def search_context(query_text: str, k: int = 3):
    """Ищет k наиболее релевантных чанков для заданного вопроса.
    Возвращает список словарей с полями: content, metadata, similarity.
    """
    # 1️⃣ Превращаем запрос в эмбеддинг
    query_vector = get_embedding(query_text)

    conn = get_db_connection()
    cur = conn.cursor()

    # 2️⃣ Выполняем поиск по векторному индексу (cosine distance через оператор <=>)
    sql = """
        SELECT 
            content,
            metadata,
            1 - (embedding <=> %s::vector) AS similarity
        FROM rag_documents
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
    """
    cur.execute(sql, (query_vector, query_vector, k))
    results = cur.fetchall()

    cur.close()
    conn.close()

    # 3️⃣ Форматируем результат в удобный вид
    formatted_results = []
    for row in results:
        formatted_results.append({
            "content": row[0],
            "metadata": row[1],
            "similarity": round(row[2], 4),
        })

    return formatted_results


# --- Демонстрация работы ---
if __name__ == "__main__":
    # Пример вопроса к базе, которую наполнили через indexer_ollama.py
    user_question = "Как часто нужно менять пароль и какой длины он должен быть?"
    print(f"❓ Вопрос: {user_question}\n")
    print("🔍 Ищу в базе знаний...\n")
    matches = search_context(user_question, k=3)
    for i, match in enumerate(matches):
        print(f"--- Результат #{i+1} (Сходство: {match['similarity']}) ---")
        # metadata обычно хранит номер страницы и имя файла
        page = match["metadata"].get("page") if isinstance(match["metadata"], dict) else "?"
        print(f"📄 Источник: Страница {page}")
        print(f"📝 Текст: {match['content'][:200]}...\n")
        print("-" * 30)
