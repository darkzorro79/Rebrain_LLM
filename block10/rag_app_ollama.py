import psycopg2
from openai import OpenAI
# Импортируем функции, уже настроенные под Ollama и pgvector
from indexer_ollama import get_db_connection, get_embedding

# 1️⃣ Настройка клиента Ollama (через совместимый OpenAI API)
ollama_client = OpenAI(
    api_key="dummy",  # ключ не нужен, но обязателен
    base_url="http://localhost:11434/v1"
)

# 2️⃣ Поиск в базе
def search_db(query_vector: list, k: int = 3):
    """Ищет k наиболее похожих чанков в PostgreSQL/pgvector."""
    conn = get_db_connection()
    cur = conn.cursor()
    sql = """
        SELECT content, metadata, 1 - (embedding <=> %s::vector) AS similarity
        FROM rag_documents
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
    """
    cur.execute(sql, (query_vector, query_vector, k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [{"content": r[0], "metadata": r[1], "score": r[2]} for r in results]

# 3️⃣ Основная функция‑оркестратор
def ask_rag(question: str):
    """Получает ответ на вопрос, используя Retrieval‑Augmented Generation."""
    print(f"\n🤖 Пользователь: {question}")

    # Шаг 1 – векторизация вопроса
    print("   ... Векторизую вопрос")
    q_vector = get_embedding(question)

    # Шаг 2 – поиск релевантных фрагментов
    print("   ... Ищу информацию в базе знаний")
    context_items = search_db(q_vector, k=3)
    if not context_items:
        return "База знаний пуста или недоступна."

    print(f"   ... Найдено {len(context_items)} фрагментов. Топ‑score: {context_items[0]['score']:.4f}")

    # Шаг 3 – формируем промпт
    context_str = "\n\n---\n\n".join([item["content"] for item in context_items])
    system_prompt = (
        "Ты — корпоративный AI‑ассистент. Отвечай на вопросы, используя ТОЛЬКО предоставленный ниже контекст. "
        "Если информации недостаточно, ответь: \"Я не нашёл информации в базе знаний\". "
        "Не придумывай факты, ссылайся на разделы документов, если это возможно."
    )
    user_prompt = f"""
    CONTEXT:
    {context_str}

    QUESTION:
    {question}
    """

    # Шаг 4 – генерация ответа через Ollama
    print("   ... Генерирую ответ")
    response = ollama_client.chat.completions.create(
        model="gpt-oss:120b-cloud",  # пример модели, установленной в Ollama
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content

# --- Демонстрация работы ---
if __name__ == "__main__":
    ans1 = ask_rag("Какие требования к длине пароля?")
    print(f"\n💬 Ответ RAG:\n{ans1}\n")
    print("=" * 50)
    ans2 = ask_rag("Как приготовить лазанью?")
    print(f"\n💬 Ответ RAG:\n{ans2}\n")
