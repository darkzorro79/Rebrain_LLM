import psycopg2
# Импортируем функции из предыдушего файла - indexer
from indexer import get_db_connection, get_embedding

def search_context(query_text: str, k: int = 3):
    """
    Ищет k наиболее релевантных чанков для заданного вопроса.
    """
    # 1. Превращаем вопрос пользователя в вектор
    query_vector = get_embedding(query_text)

    conn = get_db_connection()
    cur = conn.cursor()

    # 2. Выполняем SQL-запрос
    # Обратите внимание: мы передаём вектор дважды (для SELECT и для ORDER BY)
    # Оператор <=> означает "Cosine Distance"
    sql = """
        SELECT 
            content, 
            metadata, 
            1 - (embedding <=> %s::vector) as similarity
        FROM rag_documents
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
    """

    cur.execute(sql, (query_vector, query_vector, k))
    results = cur.fetchall()

    cur.close()
    conn.close()

    # Возвращаем список словарей для удобства
    formatted_results = []
    for row in results:
        formatted_results.append({
            "content": row[0],
            "metadata": row[1],
            "similarity": round(row[2], 4) # Округляем score
        })

    return formatted_results

# --- Демонстрация работы ---
if __name__ == "__main__":
    # Предположим, мы индексировали PDF про безопасность из примера выше
    user_question = "Как часто нужно менять пароль и какой длины он должен быть?"

    print(f"❓ Вопрос: {user_question}\n")
    print("🔍 Ищу в базе знаний...\n")

    matches = search_context(user_question, k=3)

    for i, match in enumerate(matches):
        print(f"--- Результат #{i+1} (Сходство: {match['similarity']}) ---")
        print(f"📄 Источник: Страница {match['metadata'].get('page', '?')}")
        print(f"📝 Текст: {match['content'][:200]}...") # Показываем начало текста
        print("-" * 30)
